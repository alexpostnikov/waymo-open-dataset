from math import pi, log
from functools import wraps

import torch
import torch.distributions as D
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

from typing import List, Mapping, Optional, Tuple




# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = False
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)
        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        return_embeddings = False
    ):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            data = torch.cat((data, enc_pos), dim = -1)

        # concat to channels of data and flatten axis

        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)

        # layers

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)




class Decoder(nn.Module):

    def __init__(self, num_modes=5, n_heads=8, num_encoder_layers=8, dim_ff=1024, d_model=512, num_latents=64):
        super().__init__()
        # self.transformer_model = nn.Transformer(nhead=n_heads, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_ff,
        #                            d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.transf_to_traj_mlp = nn.Linear(d_model * num_latents, 50 * num_modes)
        self.transf_to_prob_mlp = nn.Linear(d_model * num_latents, 1 * num_modes)
        self.transf_to_unc_mlp = nn.Sequential(nn.Linear(d_model * num_latents, 128),
                                               # nn.BatchNorm1d(128),
                                               nn.ReLU(),
                                               nn.Linear(128, 1))

        self.num_modes = num_modes

    def forward(self, scene_encoded):
        if scene_encoded.ndim == 2:
            scene_encoded = scene_encoded.unsqueeze(0)
        bs = scene_encoded.shape[0]
        target_encoded = self.transformer_encoder(scene_encoded.permute(1, 0, 2)).permute(1, 0, 2)
        bs = target_encoded.shape[0]
        target_encoded = target_encoded.reshape(bs , -1)
        trajs = self.transf_to_traj_mlp(target_encoded)
        traj_weights = self.transf_to_prob_mlp(target_encoded)
        weights = torch.nn.functional.softmax(traj_weights, dim=1)
        unc = self.transf_to_unc_mlp(target_encoded.reshape(bs, -1))
        trajs = trajs.unsqueeze(-1).reshape(bs, self.num_modes, 25, 2)
        trajs = torch.cumsum(trajs, dim=-2)

        ###
        # likelihood = [D.MultivariateNormal(
        #     loc=trajs[:, i].reshape(-1, 50),
        #     scale_tril=torch.stack([
        #         torch.diag(torch.ones(50)).to(trajs.device)
        #         for _ in range(bs)], dim=0)) for i in range(self.num_modes)]
        ##
        return trajs, weights, unc[:, 0] #, likelihood




class InitEmbedding(nn.Module):
    def __init__(self, inp_dim=2, out_dim=64, dr_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inp_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(dr_rate),
            nn.LayerNorm(32),
            nn.Linear(32, out_dim)
        )
        self.silly_masking = True

    def masking(self, agent_h, agent_h_avail):
        if self.silly_masking:
            # neighb_h[neighb_h_avail==0] = -100
            agent_h[agent_h_avail == 0] = -100
        return agent_h

    def forward(self, agent_h):
        return self.layers(agent_h)

class PercieverClassical(nn.Module):
    def __init__(self, in_channels: int = 17, dim_hidden: int = 128,
                 output_shape: Tuple[int, int] = (25, 2), scale_eps: float = 1e-7, num_modes: int = 5,
                 num_classes: int = 10, positional_enc_dim=16, num_blocks=4, num_latents=64,
                 device="cpu") -> None:
        super().__init__()
        self.device = device
        self._output_shape = output_shape
        self.num_modes = num_modes
        self.num_classes = num_classes
        self.embeder = InitEmbedding(out_dim=17)
        self.encoder = Perceiver(num_freq_bands=12, depth=num_blocks, max_freq=10, latent_dim=dim_hidden,
                                 num_latents=num_latents, input_channels=17, final_classifier_head=False)
        self.decoder = Decoder(num_modes=5, n_heads=4, num_encoder_layers=4,
                               dim_ff=256, d_model=dim_hidden, num_latents=num_latents)

    def forward(self, environment: torch.tensor, self_poses: torch.tensor,
                neighb_poses: torch.tensor):
        bs = environment.shape[0]
        device = environment.device
        # Encodes the visual input.
        image_emb = environment.permute(0, 2, 3, 1).reshape(bs,-1,17)
        agent_h_emb = self.embeder(self_poses[:, :, :2])
        neighb_h_emb = self.embeder(neighb_poses[:, :, :, :2]).reshape(bs, -1, agent_h_emb.shape[2])
        x = torch.cat([image_emb, agent_h_emb, neighb_h_emb], dim=1).unsqueeze(1)
        x = self.encoder(x.repeat(1,2,1,1)) #, image_emb, neighb_h_emb, agent_h_emb)
        trajs, weights, err_pred = self.decoder(x)
        return trajs, weights, err_pred


def train_step_percv(
        model: PercieverClassical,
        optimizer: torch.optim.Optimizer,
        batch: Mapping[str, torch.Tensor],
        clip: bool = False,
) -> Mapping[str, torch.Tensor]:
    """Performs a single gradient-descent optimization step."""
    # Resets optimizer's gradients.
    optimizer.zero_grad()

    env = batch.get("feature_maps")
    poses = batch.get("poses").to(env.device)
    self_poses = poses[:, 0]
    neighb_poses = poses[:, 1:]
    predictions, unc, err_predicted, likelihood = model(env, self_poses, neighb_poses)

    # Compute ADE loss
    y = batch["ground_truth_trajectory"]

    # ades = [batch_mean_metric_torch(
    #     base_metric=average_displacement_error_torch,
    #     predictions=predictions[:, i],
    #     ground_truth=y) for i in range(model.num_modes)]
    ades = torch.stack([unc[:, i] * average_displacement_error_torch(y, predictions[:, i]) for i in range(5)], dim=1)
    weight_ade = torch.sum(ades, dim=1)
    bs = y.shape[0]
    nll = torch.sum((-torch.stack([unc[:, i] * likelihood[i].log_prob(y.reshape(bs, -1)) for i in range(5)], dim=1)), 1)
    delta_err_predicted = torch.mean(torch.sqrt((err_predicted - nll) ** 2 + 1e-6))
    loss = torch.mean(nll) + 0.2 * delta_err_predicted
    loss.backward()

    del ades

    # Clips gradients norm.
    clip = 1
    if clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Performs a gradient descent step.
    optimizer.step()

    fdes = torch.stack([unc[:, i] * final_displacement_error_torch(y, predictions[:, i]) for i in range(5)], dim=1)
    weight_fde = torch.mean(torch.sum(fdes, dim=1))
    weight_ade = torch.mean(weight_ade)

    loss_dict = {
        'ade': weight_ade.detach(),
        'fde': weight_fde.detach(),

        "delta_err_predicted": delta_err_predicted.detach(),
        "nll": torch.mean(nll).detach(),
    }

    return loss_dict


def evaluate_step_percv(
        model: PercieverClassical,
        batch: Mapping[str, torch.Tensor],
) -> Tuple[Mapping[str, torch.Tensor], Optional[Mapping[str, torch.Tensor]]]:
    """Performs a single gradient-descent optimization step."""
    # Resets optimizer's gradients.

    env = batch.get("feature_maps")
    poses = batch.get("poses").to(env.device)
    self_poses = poses[:, 0]
    neighb_poses = poses[:, 1:]
    predictions, unc, err_predicted, likelihood = model(env, self_poses, neighb_poses)

    # Compute ADE loss
    y = batch["ground_truth_trajectory"]
    ades = torch.stack([unc[:, i] * average_displacement_error_torch(y, predictions[:, i]) for i in range(5)], dim=1)
    weight_ade = torch.sum(ades, dim=1)
    bs = y.shape[0]
    nll = torch.sum((-torch.stack([unc[:, i] * likelihood[i].log_prob(y.reshape(bs, -1)) for i in range(5)], dim=1)), 1)

    delta_err_predicted = torch.mean(torch.sqrt((err_predicted - nll) ** 2 + 1e-6))
    del ades
    fdes = torch.stack([unc[:, i] * final_displacement_error_torch(y, predictions[:, i]) for i in range(5)], dim=1)
    weight_fde = torch.mean(torch.sum(fdes, dim=1))
    weight_ade = torch.mean(weight_ade)

    loss_dict = {
        'ade': weight_ade.detach(),
        'fde': weight_fde.detach(),

        "delta_err_predicted": delta_err_predicted.detach(),
        "nll": torch.mean(nll).detach(),
    }

    model_outp = None
    if "return_predictions" in batch.keys() and batch["return_predictions"] is True:
        model_outp = {"predictions": predictions.cpu(), "unc": unc.cpu(), "err_predicted": err_predicted.cpu()}
        return loss_dict, model_outp
    return loss_dict



if __name__ == "__main__":
    model = PercieverClassical()
    bs = 2
    self_poses = torch.rand(bs, 8, 2)
    n_poses = torch.rand(bs, 10, 8, 2)
    env = torch.rand(bs, 17, 32, 32)
    model(env, self_poses, n_poses)

