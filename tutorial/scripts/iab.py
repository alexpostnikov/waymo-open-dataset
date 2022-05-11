import math
import torch
import torch.nn as nn
import torch.distributions as D
from scripts.pointNet import PointNetfeat
from scripts.rasterization import rasterize_batch
from scripts.rgb_loader import RgbLoader
import timm
import numpy as np
from einops import rearrange
import torchvision.models as models

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nns = 1
        for s in list(p.size()):
            nns = nns * s
        pp += nns
    return pp


class InitEmbedding(nn.Module):
    def __init__(self, inp_dim=2, out_dim=64, dr_rate=0.1, use_recurrent=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inp_dim, 32),
            nn.ReLU(),
            # nn.LayerNorm(8),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(dr_rate),
            # nn.BatchNorm2d(11),
            # nn.LayerNorm(32),
            nn.Linear(128, out_dim)
        )
        self.silly_masking = False
        self.use_recurrent = use_recurrent
        if self.use_recurrent:
            self.rec = nn.GRU(out_dim, out_dim)


    def masking(self, agent_h, agent_h_avail):
        if self.silly_masking:
            agent_h[agent_h_avail == 0] = -100
        return agent_h

    def forward(self, agent_h, agent_h_avail=None):
        agent_h = self.masking(agent_h, agent_h_avail)
        out = self.layers(agent_h)
        if self.use_recurrent:
            out, (_) = self.rec(out.permute(1,0,2))
            out = out.permute(1,0,2) 
        return out


def positionalencoding1d(d_model, length, device="cuda"):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, device=device).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def time_encoding_sin(neighb, added_emb_dim=16):
    device = neighb.device
    if neighb.ndim == 4:
        bs, num_n, length, _ = neighb.shape
        pe = positionalencoding1d(added_emb_dim, length, device) #.to(device)
        pose_enc = pe.unsqueeze(0).unsqueeze(0).repeat(bs, num_n, 1, 1)
        neighb = torch.cat((neighb, pose_enc), dim=-1)
        return neighb
    else:
        bs, length, _ = neighb.shape
        pe = positionalencoding1d(added_emb_dim, length, device) #.to(device)
        pose_enc = pe.unsqueeze(0).repeat(bs, 1, 1)
        neighb = torch.cat((neighb, pose_enc), dim=-1)
        return neighb


class NeighbourAttentionTimeEncoding(nn.Module):
    def __init__(self, inp_dim=64, embed_dim=64, pose_emb_dim=16, num_heads=16, dr_rate=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pose_emb_dim = pose_emb_dim
        self.att = nn.MultiheadAttention(embed_dim + pose_emb_dim, num_heads, dropout=dr_rate)
        self.q_mlp = nn.Linear(inp_dim + pose_emb_dim, embed_dim + pose_emb_dim)
        self.k_mlp = nn.Linear(inp_dim + pose_emb_dim, embed_dim + pose_emb_dim)
        self.v_mlp = nn.Linear(inp_dim + pose_emb_dim, embed_dim + pose_emb_dim)

        self.silly_masking = True
        self.layer_norm = nn.LayerNorm(embed_dim + pose_emb_dim)

    def masking(self, neighb_h, agent_h, neighb_h_avail, agent_h_avail):
        if self.silly_masking:
            neighb_h[neighb_h_avail == 0] = -100
            agent_h[agent_h_avail == 0] = -100
        return neighb_h, agent_h

    def forward(self, neighb_h, x):
        # DO pose(time) embedding
        neighb_h_te = time_encoding_sin(neighb_h, added_emb_dim=self.pose_emb_dim)

        bs, num_n, seq_len, data_dim = neighb_h_te.shape

        # flatten num_n and history_horizon
        neighb_h_te = neighb_h_te.reshape(bs, num_n * seq_len, data_dim)

        value = self.v_mlp(neighb_h_te).permute(1, 0, 2)
        key = self.k_mlp(neighb_h_te).permute(1, 0, 2)
        query = self.q_mlp(x).permute(1, 0, 2)
        out, _ = (self.att(query, key, value))
        out = self.layer_norm(out.permute(1, 0, 2))
        # out shape bs, history_horizon, embed_dim

        # value = self.v_mlp_fin(out)  # .permute(1, 0, 2)
        # key = self.k_mlp_fin(out)  # .permute(1, 0, 2)
        # query = self.q_mlp_fin(agent_h).permute(1, 0, 2)
        # out, _ = (self.att_fin(query, key, value))
        # out = self.layer_norm(out)
        return out


class PoseAttention(nn.Module):
    def __init__(self, inp_dim=64, embed_dim=64, data_dim=32, num_heads=8, dr_rate=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pose_emb_dim = 16
        self.att = nn.MultiheadAttention(embed_dim + self.pose_emb_dim, num_heads, dropout=dr_rate)
        self.q_mlp = nn.Linear(embed_dim + self.pose_emb_dim, embed_dim + self.pose_emb_dim)
        self.k_mlp = nn.Linear(embed_dim + 32, embed_dim + self.pose_emb_dim)
        self.v_mlp = nn.Linear(embed_dim + 32, embed_dim + self.pose_emb_dim)
        self.silly_masking = True
        self.layer_norm = nn.LayerNorm(embed_dim + self.pose_emb_dim)

    def masking(self, agent_h, agent_h_avail):
        if self.silly_masking:
            agent_h[agent_h_avail == 0] = -100
        return agent_h

    def forward(self, x, agent_h):
        agent_h_te = time_encoding_sin(agent_h, added_emb_dim=32)
        value = self.v_mlp(agent_h_te).permute(1, 0, 2)
        key = self.k_mlp(agent_h_te).permute(1, 0, 2)

        query = self.q_mlp(x).permute(1, 0, 2)
        out, _ = self.att(query, key, value)
        out = self.layer_norm(out.permute(1, 0, 2))
        return out


class SelfAttention(nn.Module):
    def __init__(self, inp_dim=64, embed_dim=64, num_heads=8, dr_rate=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.att = nn.MultiheadAttention(embed_dim + 16, num_heads, dropout=dr_rate)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim + 16, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.q_mlp = nn.Linear(embed_dim + 16 + 32, embed_dim + 16)
        # self.k_mlp = nn.Linear(embed_dim + 16 + 128, embed_dim + 16)
        # self.v_mlp = nn.Linear(embed_dim + 16 + 128, embed_dim + 16)
        self.silly_masking = True
        self.layer_norm = nn.LayerNorm(embed_dim + 16)

    def masking(self, agent_h, agent_h_avail):
        if self.silly_masking:
            agent_h[agent_h_avail == 0] = -100
        return agent_h

    def forward(self, agent_h, agent_h_avail, masking=0):
        if masking:
            agent_h = self.masking(agent_h, agent_h_avail)
        agent_h_te = time_encoding_sin(agent_h, added_emb_dim=32)
        src = self.q_mlp(agent_h_te).permute(1, 0, 2)
        out = self.transformer_encoder(src)
        return out.permute(1, 0, 2)


class VisualAttentionTransformer(nn.Module):
    def __init__(self, inp_dim=64, embed_dim=64, num_heads=16, dr_rate=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.att = nn.MultiheadAttention(embed_dim + 16, num_heads, dropout=dr_rate)
        self.v_mlp = nn.Linear(inp_dim, embed_dim + 16)
        self.k_mlp = nn.Linear(inp_dim, embed_dim + 16)
        self.q_mlp = nn.Linear(embed_dim + 16, embed_dim + 16)
        self.silly_masking = True
        self.layer_norm = nn.LayerNorm(embed_dim + 16)

    def masking(self, agent_h, agent_h_avail):
        if self.silly_masking:
            agent_h[agent_h_avail == 0] = -100
        return agent_h

    def forward(self, resnet_out, agent_h, masking=0):
        # print(self.v_mlp.in_features, resnet_out.shape)
        value = self.v_mlp(resnet_out).unsqueeze(1).permute(1, 0, 2)
        key = self.k_mlp(resnet_out).unsqueeze(1).permute(1, 0, 2)
        query = self.q_mlp(agent_h).permute(1, 0, 2)
        out, _ = self.att(query, key, value)
        out = self.layer_norm(out.permute(1, 0, 2))
        return out


class PoseEncoderBlock(nn.Module):
    def __init__(self, inp_dim=64, embed_dim=64, use_vis=True, use_segm=False, use_points=True, dr_rate=0):
        super().__init__()
        self.use_vis = use_vis
        self.use_points = use_points
        self.use_segm = use_segm
        self.sat = SelfAttention(inp_dim, embed_dim, dr_rate=dr_rate)
        self.pat = PoseAttention(inp_dim, embed_dim, data_dim=inp_dim, dr_rate=dr_rate)
        if use_vis:
            self.va = VisualAttentionTransformer(inp_dim, embed_dim, dr_rate=dr_rate)
        if use_points:
            self.pa = VisualAttentionTransformer(inp_dim, embed_dim, dr_rate=dr_rate)

    def forward(self, x, image_emb, agent_h_emb, points_emb):

        x = x + self.pat(x, agent_h_emb)

        if self.use_vis:
            x = x + self.va(image_emb, x)

        if self.use_points:
            x = x + self.pa(points_emb, x)

        # x = x + self.sat(x, None)
        x = self.sat(x, None)
        return x


class Encoder(nn.Module):
    def __init__(self, inp_dim=64, embed_dim=64, num_blocks=16, use_vis=True, use_points=True, dr_rate=0):
        super().__init__()
        self.layers = nn.ModuleList(
            [PoseEncoderBlock(inp_dim, embed_dim, use_vis=use_vis, use_points=use_points, dr_rate=dr_rate) for _ in
             range(num_blocks)])

    def forward(self, x, image_emb, agent_h_emb, points_emb):
        for pose_encoder in self.layers:
            x = pose_encoder(x, image_emb, agent_h_emb, points_emb)
        return x


class DecoderGoals(nn.Module):
    def __init__(self, inp_dim=32, inp_hor=6, out_modes=20, out_dim=2, dr_rate=0.1, use_recurrent=False):
        super().__init__()
        self.out_modes, self.out_dim = out_modes, out_dim
        self.layers = nn.Sequential(
            nn.Linear((inp_dim + 16) * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dr_rate),
            nn.LayerNorm(512),
            nn.Linear(512, out_dim)
        )
        self.use_recurrent = use_recurrent
        if self.use_recurrent:
            self.rec = nn.GRU(inp_dim + 16, inp_dim + 16)



    def forward(self, hist_enc):
        bs = hist_enc.shape[0]
        if self.use_recurrent:
            hist_enc_rec, (_) = self.rec(hist_enc.permute(1, 0, 2))  # .reshape(bs, -1)
            # predictions shape is (seq_len, bs, out_dim)
            hist_enc_rec = hist_enc_rec.permute(1, 0, 2)

            predictions = self.layers(hist_enc_rec).reshape(bs, -1)
        else:
            predictions = self.layers(hist_enc).reshape(bs, -1)
        return predictions


class DecoderTraj(nn.Module):

    def __init__(self, inp_dim=32, inp_hor=12, out_modes=1, out_dim=2, out_horiz=12, dr_rate=0.3):
        super().__init__()
        self.out_modes, self.out_dim, self.out_horiz = out_modes, out_dim, out_horiz
        self.goal_embeder = nn.Sequential(nn.Linear(2, 8),
                                          nn.ReLU(),
                                          nn.Linear(8, 32))
        self.layers = nn.Sequential(
            nn.Linear(inp_dim * inp_hor + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dr_rate),
            nn.LayerNorm(64),
            nn.Linear(64, out_dim * out_modes * out_horiz + self.out_modes)
        )

    def forward(self, hist_enc, goal):
        bs = hist_enc.shape[0]
        goal_embedded = self.goal_embeder(goal)
        inp = torch.cat((hist_enc.reshape(bs, -1), goal_embedded), dim=1)
        predictions = self.layers(inp)
        confidences = torch.softmax(predictions[:, -self.out_modes:], dim=-1)
        trajectories = predictions[:, :-self.out_modes].reshape(bs, self.out_modes, self.out_horiz, self.out_dim)
        return trajectories, confidences


class DecoderTraj3(nn.Module):

    def __init__(self, inp_dim=32, inp_hor=12, out_modes=1, out_dim=2, out_horiz=12, dr_rate=0.1, use_recurrent=False):
        super().__init__()
        self.out_modes, self.out_dim, self.out_horiz = out_modes, out_dim, out_horiz
        self.goal_embeder = nn.Sequential(nn.Linear(2, 8),
                                          nn.ReLU(),
                                          nn.Linear(8, inp_dim//2))
        out_shape = out_dim * out_modes * out_horiz + self.out_modes
        self.out_shape = out_shape

        self.outlayers = nn.Sequential(
            nn.Linear(512, out_shape * 4),
            nn.ReLU(),

            nn.Linear(out_shape * 4, out_shape * 2),
            nn.ReLU(),
            nn.Dropout(dr_rate),
            nn.LayerNorm(out_shape * 2),
            nn.Linear(out_shape * 2, out_shape)
        )
        # self.att = nn.MultiheadAttention(out_shape * 2, out_modes, dropout=0.)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.q_mlp = nn.Linear((inp_dim + 16) + inp_dim//2 + 32, 512)

        self.conf_mlp = nn.Sequential(nn.Linear(self.out_shape, self.out_shape * 2), nn.ReLU(),
                                      nn.Linear(self.out_shape * 2, 1))
        self.traj_mlp = nn.Sequential(nn.Linear(self.out_shape - self.out_modes, out_shape * 2 ), nn.ReLU(),
                                      nn.Linear(out_shape * 2, self.out_dim * out_horiz))
        self.use_recurrent = use_recurrent

    def forward(self, hist_enc, goal):
        bs = hist_enc.shape[0]
        goal_embedded = self.goal_embeder(goal.reshape(bs, -1, 2))
        inp = torch.cat((hist_enc, goal_embedded), dim=2)
        inp = time_encoding_sin(inp, added_emb_dim=32)
        agent_h_te = self.q_mlp(inp)


        predictions = self.transformer_encoder(agent_h_te.permute(1, 0, 2)).permute(1, 0, 2)
        predictions = self.outlayers(predictions)
        confidences = torch.softmax(self.conf_mlp(predictions).reshape(bs, self.out_modes), dim=-1)
        trajectories = self.traj_mlp(predictions[:, :, :self.out_shape - self.out_modes]).reshape(bs, self.out_modes,
                                                                                                  self.out_horiz,
                                                                                                  self.out_dim)
        # trajectories = trajectories.cumsum(2)
        return trajectories, confidences




class DecoderTraj2(nn.Module):

    def __init__(self, inp_dim=32, inp_hor=12, out_modes=1, out_dim=2, out_horiz=12, dr_rate=0.1, use_recurrent=False):
        super().__init__()
        self.out_modes, self.out_dim, self.out_horiz = out_modes, out_dim, out_horiz
        self.goal_embeder = nn.Sequential(nn.Linear(2, 8),
                                          nn.ReLU(),
                                          nn.Linear(8, inp_dim//2))
        out_shape = out_dim * out_modes * out_horiz + self.out_modes
        self.out_shape = out_shape

        self.outlayers = nn.Sequential(
            nn.Linear(out_shape * 2 + 32, out_shape * 4),
            nn.ReLU(),

            nn.Linear(out_shape * 4, out_shape * 2),
            nn.ReLU(),
            nn.Dropout(dr_rate),
            nn.LayerNorm(out_shape * 2),
            nn.Linear(out_shape * 2, out_shape)
        )
        # self.att = nn.MultiheadAttention(out_shape * 2, out_modes, dropout=0.)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=out_shape * 2 + 32, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.q_mlp = nn.Linear(inp_dim + inp_dim//2 + 16, out_shape * 2)

        self.conf_mlp = nn.Sequential(nn.Linear(self.out_modes, self.out_modes), nn.ReLU(),
                                      nn.Linear(self.out_modes, self.out_modes))
        self.use_recurrent = use_recurrent
#         if use_recurrent:
#             self.conf_mlp = nn.Sequential(nn.Linear(out_shape - self.out_modes, self.out_modes), nn.ReLU(),
#                                           nn.Linear(self.out_modes, self.out_modes))

#             self.rec = nn.GRU(out_shape * 2 + 32, out_dim * out_horiz)
#             self.singl_mode = nn.GRU(2, 2)

    def forward(self, hist_enc, goal):
        bs = hist_enc.shape[0]
        goal_embedded = self.goal_embeder(goal.reshape(bs, -1, 2))
        inp = torch.cat((hist_enc, goal_embedded), dim=2)
        inp_q = self.q_mlp(inp)

        agent_h_te = time_encoding_sin(inp_q, added_emb_dim=32)
        predictions = self.transformer_encoder(agent_h_te.permute(1, 0, 2)).permute(1, 0, 2)
#         if self.use_recurrent:
#             out, (_) = self.rec(predictions.permute(1, 0, 2))
#             out = out.permute(1, 0, 2)

#             confidences = torch.softmax(self.conf_mlp(out.reshape(bs, -1)), dim=-1)
#             out = rearrange(out, "bs num_modes (time data) -> (bs num_modes) time data", time=80, data=2)
#             out, (_) = self.singl_mode(out.permute(1, 0, 2))
#             out = out.permute(1, 0, 2)
#             out = rearrange(out, "(bs num_modes) time data -> bs num_modes time data", bs=bs, num_modes=hist_enc.shape[1])
#             trajectories = out.reshape(bs, self.out_modes,
#                                                  self.out_horiz,
#                                                  self.out_dim)
#             # trajectories = trajectories.cumsum(2)
#             return trajectories, confidences

        predictions = self.outlayers(predictions)

        confidences = torch.softmax(self.conf_mlp(predictions[:,:,-1].reshape(bs, -1)), dim=-1)
        trajectories = predictions[:, -1, :self.out_shape - self.out_modes].reshape(bs, self.out_modes, self.out_horiz,
                                                                                    self.out_dim)
        # trajectories = trajectories.cumsum(2)
        return trajectories, confidences

def gmm_to_mean_goal(gmm):
    return gmm.mean


def gmm_covariances(gmm):
    return gmm.component_distribution.covariance_matrix


def gmm_means(gmm):
    return gmm.component_distribution.loc



class AttPredictorPecNet(nn.Module):
    def __init__(self, inp_dim=32, embed_dim=128, num_blocks=8, out_modes=1, out_dim=2, out_horiz=12,
                 dr_rate=0.0, use_vis=True, use_points=True, use_map=True, use_rec=False):
        super().__init__()
        self.use_gt_goals = False
        self.use_points = use_points
        if use_points:
            self.pointNet = PointNetfeat(global_feat=True)
        self.use_vis = use_vis
        if use_vis:
            self.rgb_loader = RgbLoader(index_path="rendered/train/index.pkl")
            self.visual = models.resnet18(pretrained=True)
            self.visual.fc = torch.nn.Linear(512, 1024)
        self.latent = nn.Parameter(torch.rand(out_modes, embed_dim + 16), requires_grad=True)
        self.embeder = InitEmbedding(inp_dim=4, out_dim=embed_dim, use_recurrent=use_rec)
        self.encoder = Encoder(inp_dim, embed_dim, num_blocks, use_vis=use_vis, use_points=use_points, dr_rate=dr_rate)
        self.decoder_goals = DecoderGoals(embed_dim, 12, out_modes, dr_rate=dr_rate, use_recurrent=use_rec)
        self.decoder_trajs = DecoderTraj2(embed_dim, 12, out_modes, out_dim, out_horiz, dr_rate, use_recurrent=use_rec)

    def forward(self, batch_unpacked):
        ## xyz emb
        masks, rot_mat, rot_mat_inv, state_masked, xyz_personal, maps = batch_unpacked
        bsr = state_masked.shape[0]
        agent_h_emb = self.embeder(state_masked[:,:,:4])
        # pointnet embedder
        xyz_emb = None
        if self.use_points:
            xyz_emb, _, _ = self.pointNet(xyz_personal.permute(0, 2, 1))

        img_emb = None
        if self.use_vis:
            maps = maps[:, :3].to(self.latent.device) #cuda()
            img_emb = self.visual(maps) #.to(self.latent.device).cuda()
        x = self.latent.unsqueeze(0).repeat(bsr, 1, 1)
        x = self.encoder(x, img_emb, agent_h_emb, xyz_emb)
        goal_vector = self.decoder_goals(x)
        predictions, confidences = self.decoder_trajs(x, goal_vector)

        return predictions.permute(0, 2, 1, 3), confidences, goal_vector,  rot_mat, rot_mat_inv

class AttPredictorPecNetWithTypeD3(nn.Module):
    def __init__(self, inp_dim=32, embed_dim=128, num_blocks=8, out_modes=1, out_dim=2, out_horiz=12,
                 dr_rate=0.0, use_vis=True, use_points=True, use_map=True, use_rec=False):
        super().__init__()
        self.use_gt_goals = False
        self.use_points = use_points
        if use_points:
            self.pointNet = PointNetfeat(global_feat=True)
        self.use_vis = use_vis
        if use_vis:
            self.rgb_loader = RgbLoader(index_path="rendered/train/index.pkl")
            self.visual = models.resnet18(pretrained=True)
            self.visual.fc = torch.nn.Linear(512, 1024)
        self.latent = nn.Parameter(torch.rand(out_modes, embed_dim + 16), requires_grad=True)
        self.embeder = InitEmbedding(inp_dim=5, out_dim=embed_dim, use_recurrent=use_rec)
        self.encoder = Encoder(inp_dim, embed_dim, num_blocks, use_vis=use_vis, use_points=use_points, dr_rate=dr_rate)
        self.decoder_goals = DecoderGoals(embed_dim, 12, out_modes, dr_rate=dr_rate, use_recurrent=use_rec)
        self.decoder_trajs = DecoderTraj3(embed_dim, 12, out_modes, out_dim, out_horiz, dr_rate, use_recurrent=use_rec)

    def forward(self, batch_unpacked):
        ## xyz emb
        masks, rot_mat, rot_mat_inv, state_masked, xyz_personal, maps = batch_unpacked
        bsr = state_masked.shape[0]
        agent_h_emb = self.embeder(state_masked)
        # pointnet embedder
        xyz_emb = None
        if self.use_points:
            xyz_emb, _, _ = self.pointNet(xyz_personal.permute(0, 2, 1))

        img_emb = None
        if self.use_vis:
            maps = maps[:, :3].to(self.latent.device) #cuda()
            img_emb = self.visual(maps) #.to(self.latent.device).cuda()
        x = self.latent.unsqueeze(0).repeat(bsr, 1, 1)
        x = self.encoder(x, img_emb, agent_h_emb, xyz_emb)
        goal_vector = self.decoder_goals(x)
        predictions, confidences = self.decoder_trajs(x, goal_vector)

        return predictions.permute(0, 2, 1, 3), confidences, goal_vector,  rot_mat, rot_mat_inv


class AttPredictorPecNetWithType(nn.Module):
    def __init__(self, inp_dim=32, embed_dim=128, num_blocks=8, out_modes=1, out_dim=2, out_horiz=12,
                 dr_rate=0.0, use_vis=True, use_points=True, use_map=True, use_rec=False):
        super().__init__()
        self.use_gt_goals = False
        self.use_points = use_points
        if use_points:
            self.pointNet = PointNetfeat(global_feat=True)
        self.use_vis = use_vis
        if use_vis:
            self.rgb_loader = RgbLoader(index_path="rendered/train/index.pkl")
            self.visual = models.resnet18(pretrained=True)
            self.visual.fc = torch.nn.Linear(512, 1024)
        self.latent = nn.Parameter(torch.rand(out_modes, embed_dim + 16), requires_grad=True)
        self.embeder = InitEmbedding(inp_dim=5, out_dim=embed_dim, use_recurrent=use_rec)
        self.encoder = Encoder(inp_dim, embed_dim, num_blocks, use_vis=use_vis, use_points=use_points, dr_rate=dr_rate)
        self.decoder_goals = DecoderGoals(embed_dim, 12, out_modes, dr_rate=dr_rate, use_recurrent=use_rec)
        self.decoder_trajs = DecoderTraj2(embed_dim, 12, out_modes, out_dim, out_horiz, dr_rate, use_recurrent=use_rec)

    def forward(self, batch_unpacked):
        ## xyz emb
        masks, rot_mat, rot_mat_inv, state_masked, xyz_personal, maps = batch_unpacked
        bsr = state_masked.shape[0]
        agent_h_emb = self.embeder(state_masked)
        # pointnet embedder
        xyz_emb = None
        if self.use_points:
            xyz_emb, _, _ = self.pointNet(xyz_personal.permute(0, 2, 1))

        img_emb = None
        if self.use_vis:
            maps = maps[:, :3].to(self.latent.device) #cuda()
            img_emb = self.visual(maps) #.to(self.latent.device).cuda()
        x = self.latent.unsqueeze(0).repeat(bsr, 1, 1)
        x = self.encoder(x, img_emb, agent_h_emb, xyz_emb)
        goal_vector = self.decoder_goals(x)
        predictions, confidences = self.decoder_trajs(x, goal_vector)

        return predictions.permute(0, 2, 1, 3), confidences, goal_vector,  rot_mat, rot_mat_inv




class CovNet(nn.Module):
    def __init__(self, scene_inp_dim=32, out_modes=1, out_dim=2, out_horiz=12, inp_hor=8, traj_dim=2, goals_dim=5,
                 dr_rate=0.3):
        super().__init__()
        self.traj_emb = nn.Sequential(nn.Linear(traj_dim * out_horiz, 64),
                                      nn.ReLU(),
                                      nn.LayerNorm(64),
                                      nn.Dropout(dr_rate),
                                      nn.Linear(64, 32))

        self.goal_emb = nn.Sequential(nn.Linear(goals_dim, 64),
                                      nn.ReLU(),
                                      nn.LayerNorm(64),
                                      nn.Dropout(dr_rate),
                                      nn.Linear(64, 32))

        self.layers = nn.Sequential(
            nn.Linear(scene_inp_dim * inp_hor + 32 + 32, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dr_rate),
            nn.LayerNorm(512),
            nn.Linear(512, out_horiz * out_modes * 4)
        )

    def forward(self, scene_emb, trajectory, goals):
        bs, in_hor, scene_dim = scene_emb.shape
        trajectory_embedded = self.traj_emb(trajectory.reshape(bs, -1))
        goals_embedded = self.goal_emb(goals)
        covs = self.layers(torch.cat((scene_emb.reshape(bs, in_hor * scene_dim),
                                      trajectory_embedded, goals_embedded), dim=1))
        gmm = traj_pred_to_gmm(trajectory, covs)
        return gmm


def traj_pred_to_gmm(means, covs, horizon=12, num_modes=1):
    bs = means.shape[0]
    mean_predictions = means.reshape(bs, horizon, num_modes, 2)

    probs_predictions = torch.ones(bs, horizon, num_modes).to(means.device)
    probs_predictions = torch.softmax(probs_predictions, dim=1).reshape(bs, horizon, num_modes)

    #########

    mix = D.Categorical(probs_predictions)
    scale_tril = torch.tril(covs.reshape(bs, horizon, 1, 2, 2)) ** 2
    scale_tril[:, :, :, 1, 0] = covs.reshape(bs, horizon, 1, 2, 2)[:, :, :, 1, 0]
    bi_gaus = D.multivariate_normal.MultivariateNormal(mean_predictions, scale_tril=scale_tril)
    gmm = D.MixtureSameFamily(mix, bi_gaus)
    return gmm


class AttPredictorPecNetCovNet(nn.Module):
    def __init__(self, inp_dim=32, embed_dim=32, num_blocks=8, in_hor=8, out_modes=1, out_dim=2, out_horiz=12,
                 dr_rate=0.3, use_vis=True, use_segm=False, gt_goals=None):
        super().__init__()
        self.mean_predicotr = AttPredictorPecNet(inp_dim, embed_dim, num_blocks, in_hor, out_modes, out_dim, out_horiz,
                                                 dr_rate, use_vis, use_segm)
        self.cov_pred = CovNet(embed_dim, out_modes, out_dim, out_horiz, inp_hor=8, traj_dim=2, goals_dim=7 * 20,
                               dr_rate=0.3)

    def forward(self, image, neighb_h, agent_h, neighb_h_avail, agent_h_avail, segm=None, goals=None):
        trajectories, confidences, gmm, scene_emb, goal_vector = self.mean_predicotr(image, neighb_h, agent_h,
                                                                                     neighb_h_avail, agent_h_avail,
                                                                                     segm, goals=goals)
        with torch.no_grad():
            real_trajs, _ = self.mean_predicotr.decoder_trajs(scene_emb, gmm.mean)

        traj_gmm = self.cov_pred(scene_emb, real_trajs, goal_vector)

        return trajectories, confidences, gmm, traj_gmm, None


if __name__ == "__main__":
    model = AttPredictorPecNet(embed_dim=1024, out_modes=6, out_horiz=20).cuda()
    bs = 2
    to_predict = torch.zeros(bs, 128).cuda()
    to_predict[0, :7] = 1
    to_predict[0, 67] = 1
    to_predict[1, 10:17] = 1
    to_predict[1, 77] = 1
    data = {"roadgraph_samples/xyz": torch.rand(bs, 20000, 3).cuda(),
            "state/current/x": torch.rand(bs, 128).cuda(),
            "state/current/y": torch.rand(bs, 128).cuda(),
            "state/past/x": torch.rand(bs, 128, 10).cuda(),
            "state/past/y": torch.rand(bs, 128, 10).cuda(),
            "state/tracks_to_predict": to_predict
            }
    out = model(data)
    print(1)
