import torch
from torch import nn as nn
import os
from einops import rearrange, repeat
from numpy import pi
import torch.nn.functional
from test.perciever import Perceiver


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


class PercieverBased(nn.Module):
    def __init__(self):
        super().__init__()

        self.perc = Perceiver(num_freq_bands=12, depth=4, max_freq=10, latent_dim=161 * 6,  # (80*6*2)+6,
                              num_latents=128, input_channels=54, final_classifier_head=False)

        self.perc_xyz = Perceiver(num_freq_bands=12, depth=4, max_freq=10, latent_dim=32,  # (80*6*2)+6,
                                  num_latents=128, input_channels=200, final_classifier_head=False)
        self.xyz_lin_0 = nn.Linear(80, 128)
        self.xyz_lin_1 = nn.Linear(10, 11)

    def forward(self, data):
        bs = data["roadgraph_samples/xyz"].shape[0]
        xyz = data["roadgraph_samples/xyz"].reshape(bs, 20000, 3).reshape(bs, -1, 200).cuda()
        xyz = self.perc_xyz(xyz.unsqueeze(2))
        # xyz = self.xyz_lin_0(xyz.reshape(bs, 500, 80)).permute(0, 2, 1).unsqueeze(2)
        # xyz = self.xyz_lin_1(xyz.reshape(bs, 128, 50, 10)).permute(0, 1, 3, 2)
        cur = torch.cat(
            [data["state/current/x"].reshape(-1, 128, 1, 1), data["state/current/y"].reshape(-1, 128, 1, 1)], -1)
        past = torch.cat(
            [data["state/past/x"].reshape(-1, 128, 10, 1), data["state/past/y"].reshape(-1, 128, 10, 1)],
            -1)
        x = torch.cat([xyz, torch.cat([cur, past], dim=2).reshape(bs, 128, -1).cuda()], dim=2)
        out = self.perc(x.unsqueeze(2))
        poses = out[..., :-6].reshape(bs, 128, 80, 6, 2)
        confs = torch.nn.functional.softmax(out[..., -6:].reshape(bs, 128, 6), -1)
        poses_cum = torch.cumsum(poses, 2)
        return poses_cum, confs


class MultyModel(nn.Module):
    def __init__(self, use_every_nth_prediction):
        super().__init__()
        self.use_every_nth_prediction = use_every_nth_prediction
        self.tr = nn.Transformer(d_model=500 + 25, nhead=5, num_encoder_layers=4)  # .cuda()
        self.lin_xyz = nn.Linear(3, 2)  # .cuda()
        # self.lin_xyz_post = nn.Linear(self.tr.d_model, 64)
        self.lin_xyz_post = nn.Sequential(nn.Linear(self.tr.d_model, 128),
                                          nn.ReLU(),
                                          nn.LayerNorm(128),
                                          nn.Linear(128, 64))
        self.hist_tr = nn.Transformer(d_model=24 + 9, nhead=3, num_encoder_layers=4)  # .cuda()
        self.hist_tr_tgt = nn.Parameter(torch.rand(128, self.hist_tr.d_model).cuda(), requires_grad=True)
        # self.lin_hist = nn.Linear(22, 24)  # .cuda()
        self.lin_hist = nn.Sequential(nn.Linear(22, 64),
                                      nn.ReLU(),
                                      nn.LayerNorm(64),
                                      nn.Linear(64, 24))
        self.future_tr = nn.Transformer(d_model=256, nhead=16, num_encoder_layers=4)  # .cuda()
        self.lin_fut = nn.Linear(97, 256)
        self.dec = nn.Sequential(nn.Linear(289, 512),
                                 nn.ReLU(),
                                 nn.LayerNorm(512),
                                 nn.Linear(512, 256))

        self.dec_f = nn.Sequential(nn.Linear(22 + 256, 512),
                                   nn.ReLU(),
                                   nn.LayerNorm(512),
                                   nn.Linear(512, ((160 // self.use_every_nth_prediction) + 1) * 6))
        self.fourier_encode_data = True
        self.tgt_xyz = nn.Parameter(torch.rand(128, self.tr.d_model).cuda(), requires_grad=True)
        self.tgt_fut = nn.Parameter(torch.rand(128, 256).cuda(), requires_grad=True)

    def forward(self, data):
        bs = data["roadgraph_samples/xyz"].shape[0]
        xyz = data["roadgraph_samples/xyz"].reshape(bs, -1, 3).cuda()
        xyz_mean = data["roadgraph_samples/xyz"].reshape(bs, -1, 3).mean(1).reshape(bs, 1, 3).cuda()
        src = self.lin_xyz(xyz - xyz_mean).reshape(bs, -1, 500)
        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device="cuda"), src.shape[1:2]))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, 10, 12)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=bs)

            src = torch.cat((src, enc_pos.to(src.device)), dim=-1)

        tgt = self.tgt_xyz.reshape(1, 128, -1).repeat(bs, 1, 1)

        out_0 = self.tr(src.permute(1, 0, 2), tgt.permute(1, 0, 2)).permute(1, 0, 2)
        out_0 = self.lin_xyz_post(out_0)
        cur = torch.cat(
            [data["state/current/x"].reshape(-1, 128, 1, 1), data["state/current/y"].reshape(-1, 128, 1, 1)], -1)

        past = torch.cat([data["state/past/x"].reshape(-1, 128, 10, 1), data["state/past/y"].reshape(-1, 128, 10, 1)],
                         -1)
        # state = torch.cat([cur, past - cur], 1).permute(0, 2, 1, 3).reshape(-1, 128, 11 * 2).cuda()
        state = (torch.cat([cur, past], 2).cuda() - xyz_mean[:, :, :2].reshape(bs, 1, 1, 2))
        state = rearrange(state, "bs nump time datadim -> bs nump (time datadim) ")  # .reshape(-1, 128, 11 * 2)
        state_emb = self.lin_hist(state)
        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device="cuda"), state_emb.shape[1:-1]))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, 12)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=bs)
            state_emb = torch.cat((state_emb, enc_pos.to(src.device)), dim=-1)

        tgt_hist = self.hist_tr_tgt.reshape(1, 128, -1).repeat(bs, 1, 1)
        out_1 = self.hist_tr(state_emb.permute(1, 0, 2), tgt_hist.permute(1, 0, 2)).permute(1, 0, 2)
        out_2 = self.lin_fut(torch.cat([out_0, out_1], -1))
        # future_tgt = cur.reshape(-1, 1, 128, 2).repeat(1, 80, 1, 1).permute(0, 2, 1, 3).cuda()
        # future_tgt -= xyz_mean[:, :, :2].reshape(bs, 1, 1, 2)
        # future_tgt = future_tgt.reshape(-1, 128, 160)
        # future_tgt = torch.rand(bs, 128, 256).cuda()
        future_tgt = self.tgt_fut.reshape(1, 128, -1).repeat(bs, 1, 1)

        # cur.reshape(-1, 1, 128, 2).repeat(1, 80, 1, 1).permute(0, 2, 1, 3).reshape(-1, 128, 160).cuda()
        out_3 = self.future_tr(out_2.permute(1, 0, 2), future_tgt.permute(1, 0, 2))
        out_3 = out_3.permute(1, 0, 2).reshape(bs, 128, -1)  # bs, 128, 80 ,2

        fin_input = torch.cat([state_emb, out_3], -1)
        out_dec = self.dec(fin_input)  # .reshape(-1, 128, 80, 2)
        cur = torch.cat(
            [data["state/current/x"].reshape(-1, 1, 128, 1), data["state/current/y"].reshape(-1, 1, 128, 1)], -1)
        past = torch.cat([data["state/past/x"].reshape(-1, 128, 10, 1), data["state/past/y"].reshape(-1, 128, 10, 1)],
                         -1).permute(0, 2, 1, 3)
        state = (torch.cat([cur, past], 1) - cur).permute(0, 2, 1, 3).reshape(-1, 128, 11 * 2).cuda()
        out = self.dec_f(torch.cat([out_dec, state], dim=2))
        poses = out[..., :-6].reshape(bs, 128, -1, 6, 2)
        confs = torch.nn.functional.softmax(out[..., -6:].reshape(bs, 128, 6), -1)
        poses_cum = torch.cumsum(poses, 2)
        return poses_cum, confs  # out.reshape(-1, 128, 80, 2)


class MapLess(nn.Module):
    def __init__(self, use_every_nth_prediction, data_dim = 128):
        super().__init__()
        self.data_dim = data_dim
        self.use_every_nth_prediction = use_every_nth_prediction

        self.temporal_tr_after = nn.Sequential(nn.Linear(data_dim+25, 128),
                                          nn.ReLU(),
                                          nn.LayerNorm(128),
                                          nn.Linear(128, data_dim))

        self.spatial_lin = nn.Sequential(nn.Linear(data_dim, data_dim//2),
                                         nn.ReLU(),
                                         nn.LayerNorm(data_dim//2),
                                         nn.Linear(data_dim//2, data_dim-25))

        self.spatial_tgt = nn.Parameter(torch.rand(128, data_dim).cuda(), requires_grad=True)
        num_l = 8
        self.spatial_tr = nn.Transformer(d_model=data_dim, nhead=4, num_encoder_layers=num_l, batch_first=True,
                                         dim_feedforward=data_dim, dropout=0.01)
        self.spatial_tr_after = nn.Sequential(nn.Linear(data_dim, 128),
                                               nn.ReLU(),
                                               nn.LayerNorm(128),
                                               nn.Linear(128, data_dim))

        self.temporal_lin = nn.Sequential(nn.Linear(data_dim, 8),
                                          nn.ReLU(),
                                          nn.LayerNorm(8),
                                          nn.Linear(8, data_dim))
        self.temporal_tgt = nn.Parameter(torch.rand(11, data_dim+25).cuda(), requires_grad=True)
        self.temporal_tr = nn.Transformer(d_model=data_dim+25, nhead=1, num_encoder_layers=num_l, batch_first=True,
                                         dim_feedforward=64, dropout=0.01)


        # self.lstm = nn.LSTM(66, 48, 1, batch_first=True)
        self.confs = nn.Linear(12 * 80 // use_every_nth_prediction, 6)
        self.state_lin = nn.Linear(20,  80//use_every_nth_prediction * 2)

        self.out_lin = nn.Sequential(nn.Linear(8, 4),
                                     nn.ReLU(),
                                     nn.Linear(4, 2))

        self.lstm_pre = nn.Sequential(nn.Linear(2, 8),
                                   nn.ReLU(),
                                   nn.LayerNorm(8),
                                   # nn.BatchNorm1d(512),
                                   nn.Linear(8, data_dim))

        self.lstm = nn.LSTM(data_dim, 64, 1, batch_first=True)
        self.lstm_past = nn.Sequential(nn.Linear(64*11, 512),
                                   nn.ReLU(),
                                   nn.LayerNorm(512),
                                    nn.Linear(512, ((160 // self.use_every_nth_prediction) + 1) * 6))

    def forward(self, data):
        bs = data["roadgraph_samples/xyz"].shape[0]


        cur = torch.cat(
            [data["state/current/x"].reshape(-1, 128, 1, 1), data["state/current/y"].reshape(-1, 128, 1, 1)], -1)

        past = torch.cat([data["state/past/x"].reshape(-1, 128, 10, 1), data["state/past/y"].reshape(-1, 128, 10, 1)],
                         -1)  # .permute(0, 2, 1, 3)
        state_emb = self.lstm_pre(torch.cat([cur, past - cur], dim=2).reshape(bs * 128, 11, -1).cuda())

        spatial_outs = torch.rand(bs, 128, 0, self.data_dim).cuda()
        for timestemp in range(11):
            state_spatial = state_emb.reshape(bs, 128, 11, -1)[:, :, timestemp, :] + 0
            tgt_spatial = self.spatial_tgt.unsqueeze(0).repeat(bs, 1, 1)
            state_spatial = self.spatial_lin(state_spatial)
            axis_pos = list(
                map(lambda size: torch.linspace(-1., 1., steps=size, device="cuda"), state_spatial.shape[1:2]))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, 10, 12)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=bs)
            state_spatial = torch.cat((state_spatial, enc_pos.to(state_spatial.device)), dim=-1)

            spatial_out = self.spatial_tr(state_spatial, tgt_spatial)
            spatial_out = rearrange(spatial_out, "bs nump datadim -> bs nump 1 datadim")
            spatial_outs = torch.cat([spatial_outs, spatial_out], dim=2)

        # out = rearrange(out, "bs nump time datadim -> (bs nump) time datadim")
        # print(spatial_outs.shape)
        spatial_outs = self.spatial_tr_after(spatial_outs)
        spatial_outs = rearrange(spatial_outs, "bs nump time datadim -> (bs nump) time datadim")
        # state_emb -> bs, 128, 11, 32
        state = state_emb + 0 # torch.cat([cur, past], 2).cuda() - cur.cuda()
        state_temporal = self.temporal_lin(state) #rearrange(state, "bs nump time datadim -> (bs nump) time datadim"))
        tgt_temporal = self.temporal_tgt.unsqueeze(0).repeat(bs * 128, 1, 1)

        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device="cuda"), state_temporal.shape[1:2]))
        pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
        enc_pos = fourier_encode(pos, 10, 12)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b=bs * 128)

        state_temporal = torch.cat((state_temporal, enc_pos.to(state_temporal.device)), dim=-1)

        temporal_out = self.temporal_tr(state_temporal, tgt_temporal)
        temporal_out = self.temporal_tr_after(temporal_out)
        # temporal_out = rearrange(temporal_out, "(bs nump) time datadim -> bs nump time datadim", nump=128)


        out, (_, _) = self.lstm(state_emb + temporal_out + spatial_outs)
        out = self.lstm_past(out.reshape(bs*128, -1))
        # state = torch.cat([cur, past - cur], dim=2).reshape(bs*128, -1).cuda()
        # out = self.SIMPL(state)
        poses = out[..., :-6].reshape(bs, 128, -1, 6, 2)
        confs = torch.nn.functional.softmax(out[..., -6:].reshape(bs, 128, 6), -1)
        poses_cum = torch.cumsum(poses, 2)
        return poses_cum, confs  # out.reshape(-1, 128, 80, 2)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tr = nn.Transformer(d_model=500, nhead=4, num_encoder_layers=4)  # .cuda()
        self.lin_xyz = nn.Linear(3, 2)  # .cuda()
        self.lin_xyz_post = nn.Linear(500, 64)
        self.hist_tr = nn.Transformer(d_model=24, nhead=12, num_encoder_layers=4)  # .cuda()
        # self.lin_hist = nn.Linear(22, 24)  # .cuda()
        self.lin_hist = nn.Sequential(nn.Linear(22, 64),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(128),
                                      nn.Linear(64, 24))
        self.future_tr = nn.Transformer(d_model=160, nhead=16, num_encoder_layers=4)  # .cuda()
        self.lin_fut = nn.Linear(88, 160)
        self.dec = nn.Sequential(nn.Linear((24 + 160), 160),
                                 nn.ReLU(),
                                 nn.Linear(160, 64))

        self.dec_f = nn.Sequential(nn.Linear(22 + 64, 64),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(128),
                                   nn.Linear(64, 160))

    def forward(self, data):
        bs = data["roadgraph_samples/xyz"].shape[0]

        xyz = data["roadgraph_samples/xyz"].reshape(bs, -1, 3).cuda()
        xyz_mean = data["roadgraph_samples/xyz"].reshape(bs, -1, 3).mean(1).reshape(bs, 1, 3).cuda()
        src = self.lin_xyz(xyz - xyz_mean).reshape(bs, -1, 500)
        tgt = torch.rand(bs, 128, 500).cuda()
        out_0 = self.tr(src.permute(1, 0, 2), tgt.permute(1, 0, 2)).permute(1, 0, 2)
        out_0 = self.lin_xyz_post(out_0)
        cur = torch.cat(
            [data["state/current/x"].reshape(-1, 1, 128, 1), data["state/current/y"].reshape(-1, 1, 128, 1)], -1)
        # cur = torch.cat(
        #     [data["state/current/x"].reshape(-1, 128, 1,  1), data["state/current/y"].reshape(-1, 128,1,  1)], -1).permute(0,2,1,3)
        past = torch.cat([data["state/past/x"].reshape(-1, 128, 10, 1), data["state/past/y"].reshape(-1, 128, 10, 1)],
                         -1).permute(0, 2, 1, 3)
        # state = torch.cat([cur, past - cur], 1).permute(0, 2, 1, 3).reshape(-1, 128, 11 * 2).cuda()
        state = (torch.cat([cur, past], 1).permute(0, 2, 1, 3).cuda() - xyz_mean[:, :, :2].reshape(bs, 1, 1,
                                                                                                   2)).reshape(-1, 128,
                                                                                                               11 * 2)

        state_emb = self.lin_hist(state)
        tgt = torch.rand(bs, 128, 24).cuda()
        out_1 = self.hist_tr(state_emb.permute(1, 0, 2), tgt.permute(1, 0, 2)).permute(1, 0, 2)
        out_2 = self.lin_fut(torch.cat([out_0, out_1], -1))
        # future_tgt = cur.reshape(-1, 1, 128, 2).repeat(1, 80, 1, 1).permute(0, 2, 1, 3).cuda()
        # future_tgt -= xyz_mean[:, :, :2].reshape(bs, 1, 1, 2)
        # future_tgt = future_tgt.reshape(-1, 128, 160)
        future_tgt = torch.rand(bs, 128, 160).cuda()
        # cur.reshape(-1, 1, 128, 2).repeat(1, 80, 1, 1).permute(0, 2, 1, 3).reshape(-1, 128, 160).cuda()
        out_3 = self.future_tr(out_2.permute(1, 0, 2), future_tgt.permute(1, 0, 2))
        out_3 = out_3.permute(1, 0, 2).reshape(bs, 128, -1)  # bs, 128, 80 ,2

        fin_input = torch.cat([state_emb, out_3], -1)
        out_dec = self.dec(fin_input)  # .reshape(-1, 128, 80, 2)
        cur = torch.cat(
            [data["state/current/x"].reshape(-1, 1, 128, 1), data["state/current/y"].reshape(-1, 1, 128, 1)], -1)
        past = torch.cat([data["state/past/x"].reshape(-1, 128, 10, 1), data["state/past/y"].reshape(-1, 128, 10, 1)],
                         -1).permute(0, 2, 1, 3)
        state = (torch.cat([cur, past], 1) - cur).permute(0, 2, 1, 3).reshape(-1, 128, 11 * 2).cuda()
        out = self.dec_f(torch.cat([out_dec, state], dim=2))

        return out.reshape(-1, 128, 80, 2)

    def get_mask(self, data):
        mask = data["state/tracks_to_predict"]
        return mask

    def get_n_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


class SimplModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(nn.Linear(22, 64),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(128),
                                 nn.Linear(64, 160))

    def forward(self, data):
        cur = torch.cat(
            [data["state/current/x"].reshape(-1, 1, 128, 1), data["state/current/y"].reshape(-1, 1, 128, 1)], -1)
        past = torch.cat([data["state/past/x"].reshape(-1, 128, 10, 1), data["state/past/y"].reshape(-1, 128, 10, 1)],
                         -1).permute(0, 2, 1, 3)
        state = (torch.cat([cur, past], 1) - cur).permute(0, 2, 1, 3).reshape(-1, 128, 11 * 2).cuda()
        out = self.dec(state).reshape(-1, 128, 80, 2)
        return out


class Checkpointer:
    """A simple `PyTorch` model load/save wrapper."""

    def __init__(
            self,
            model: nn.Module,
            torch_seed: int,
            ckpt_dir: str,
            checkpoint_frequency: int
    ) -> None:
        """Constructs a simple load/save checkpointer."""
        self._model = model
        self._torch_seed = torch_seed
        self._ckpt_dir = ckpt_dir
        self._best_validation_loss = float('inf')

        # The number of times validation loss must improve prior to our
        # checkpointing of the model
        if checkpoint_frequency == -1:
            self.checkpoint_frequency = float('inf')  # We will never checkpoint
        else:
            self.checkpoint_frequency = checkpoint_frequency

        self._num_improvements_since_last_ckpt = 0

        os.makedirs(self._ckpt_dir, exist_ok=True)

    def save(
            self,
            epoch: int,
            new_validation_loss: float
    ) -> str:
        """Saves the model to the `ckpt_dir/epoch/model.pt` file."""
        model_file_name = f'model-seed-{self._torch_seed}-epoch-{epoch}.pt'

        if new_validation_loss < self._best_validation_loss:
            self._num_improvements_since_last_ckpt += 1
            self._best_validation_loss = new_validation_loss

            if self._num_improvements_since_last_ckpt >= self.checkpoint_frequency:
                print(
                    f'Validation loss has improved '
                    f'{self._num_improvements_since_last_ckpt} times since last ckpt, '
                    f'storing checkpoint {model_file_name}.')
        ckpt_path = os.path.join(self._ckpt_dir, model_file_name)
        torch.save(self._model.state_dict(), ckpt_path)
        self._num_improvements_since_last_ckpt = 0
        return ckpt_path

    def load(
            self,
            epoch: int,
    ) -> nn.Module:
        """Loads the model from the `ckpt_dir/epoch/model.pt` file."""
        if epoch >= 0:
            model_file_name = f'model-seed-{self._torch_seed}-epoch-{epoch}.pt'
            ckpt_path = os.path.join(self._ckpt_dir, model_file_name)
            self._model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
            print(f"loading model with  path {ckpt_path}")
        else:
            print(f"training from scratch")
        return self._model
