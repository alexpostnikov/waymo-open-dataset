import math
import torch
import torch.nn as nn
from torchvision import models
import torch.distributions as D
# import time
# import numpy as np
# import timm
from test.pointNet import PointNetfeat
import timm


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nns = 1
        for s in list(p.size()):
            nns = nns * s
        pp += nns
    return pp


class InitEmbedding(nn.Module):
    def __init__(self, inp_dim=2, out_dim=64, dr_rate=0.1):
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

    def masking(self, agent_h, agent_h_avail):
        if self.silly_masking:
            agent_h[agent_h_avail == 0] = -100
        return agent_h

    def forward(self, agent_h, agent_h_avail=None):
        agent_h = self.masking(agent_h, agent_h_avail)
        return self.layers(agent_h)


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def time_encoding_sin(neighb, added_emb_dim=16):
    device = neighb.device
    if neighb.ndim == 4:
        bs, num_n, length, _ = neighb.shape
        pe = positionalencoding1d(added_emb_dim, length).to(device)
        pose_enc = pe.unsqueeze(0).unsqueeze(0).repeat(bs, num_n, 1, 1)
        # print(pose_enc.shape)
        neighb = torch.cat((neighb, pose_enc), dim=-1)
        return neighb
    else:
        bs, length, _ = neighb.shape
        pe = positionalencoding1d(added_emb_dim, length).to(device)
        pose_enc = pe.unsqueeze(0).repeat(bs, 1, 1)
        # print(pose_enc.shape)
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
        return out


class PoseAttention(nn.Module):
    def __init__(self, inp_dim=64, embed_dim=64, data_dim=32, num_heads=8, dr_rate=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pose_emb_dim = 16
        self.att = nn.MultiheadAttention(embed_dim + self.pose_emb_dim, num_heads, dropout=dr_rate)
        self.q_mlp = nn.Linear(embed_dim + self.pose_emb_dim, embed_dim + self.pose_emb_dim)
        self.k_mlp = nn.Linear(embed_dim + 128, embed_dim + self.pose_emb_dim)
        self.v_mlp = nn.Linear(embed_dim + 128, embed_dim + self.pose_emb_dim)
        self.silly_masking = True
        self.layer_norm = nn.LayerNorm(embed_dim + self.pose_emb_dim)

    def masking(self, agent_h, agent_h_avail):
        if self.silly_masking:
            agent_h[agent_h_avail == 0] = -100
        return agent_h

    def forward(self, x, agent_h):
        agent_h_te = time_encoding_sin(agent_h, added_emb_dim=128)
        value = self.v_mlp(agent_h_te).permute(1, 0, 2)
        key = self.k_mlp(agent_h_te).permute(1, 0, 2)

        query = self.q_mlp(x).permute(1, 0, 2)
        out, _ = self.att(query, key, value)
        out = self.layer_norm(out)
        return out.permute(1, 0, 2)


class SelfAttention(nn.Module):
    def __init__(self, inp_dim=64, embed_dim=64, num_heads=8, dr_rate=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.att = nn.MultiheadAttention(embed_dim + 16, num_heads, dropout=dr_rate)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim + 16, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.q_mlp = nn.Linear(embed_dim + 16 + 128, embed_dim + 16)
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
        agent_h_te = time_encoding_sin(agent_h, added_emb_dim=128)
        src = self.q_mlp(agent_h_te).permute(1, 0, 2)
        out = self.transformer_encoder(src)

        return out.permute(1, 0, 2)


class VisualAttentionTransformer(nn.Module):
    def __init__(self, inp_dim=64, embed_dim=64, num_heads=16, dr_rate=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.att = nn.MultiheadAttention(embed_dim + 16, num_heads, dropout=dr_rate)
        self.q_mlp = nn.Linear(embed_dim + 16, embed_dim + 16)
        self.k_mlp = nn.Linear(embed_dim, embed_dim + 16)
        self.v_mlp = nn.Linear(embed_dim, embed_dim + 16)
        self.silly_masking = True
        self.layer_norm = nn.LayerNorm(embed_dim + 16)

    def masking(self, agent_h, agent_h_avail):
        if self.silly_masking:
            agent_h[agent_h_avail == 0] = -100
        return agent_h

    def forward(self, resnet_out, agent_h, masking=0):
        value = self.v_mlp(resnet_out).unsqueeze(1).permute(1, 0, 2)
        key = self.k_mlp(resnet_out).unsqueeze(1).permute(1, 0, 2)
        query = self.q_mlp(agent_h).permute(1, 0, 2)
        out, _ = self.att(query, key, value)
        out = self.layer_norm(out)
        return out.permute(1, 0, 2)


class PoseEncoderBlock(nn.Module):
    def __init__(self, inp_dim=64, embed_dim=64, use_vis=True, use_segm=False, dr_rate=0):
        super().__init__()
        self.use_vis = use_vis
        self.use_segm = use_segm
        self.sat = SelfAttention(inp_dim, embed_dim, dr_rate=dr_rate)
        self.pat = PoseAttention(inp_dim, embed_dim, data_dim=inp_dim, dr_rate=dr_rate)
        if use_vis:
            self.va = VisualAttentionTransformer(inp_dim, embed_dim, dr_rate=dr_rate)
            self.va1 = VisualAttentionTransformer(inp_dim, embed_dim, dr_rate=dr_rate)

    def forward(self, x, image_emb, agent_h_emb, me):

        x = x + self.pat(x, agent_h_emb)

        if self.use_vis:
            x = x + self.va(image_emb, x)
            x = x + self.va1(me, x)

        x = x + self.sat(x, None)

        return x


class Encoder(nn.Module):
    def __init__(self, inp_dim=64, embed_dim=64, num_blocks=16, use_vis=True, dr_rate=0):
        super().__init__()
        self.layers = nn.ModuleList(
            [PoseEncoderBlock(inp_dim, embed_dim, use_vis=use_vis, dr_rate=dr_rate) for _ in
             range(num_blocks)])

    def forward(self, x, image_emb, agent_h_emb, me):
        for pose_encoder in self.layers:
            x = x + pose_encoder(x, image_emb, agent_h_emb, me)
        return x


class DecoderGoals(nn.Module):
    def __init__(self, inp_dim=32, inp_hor=12, out_modes=20, out_dim=2 + 4 + 1, dr_rate=0.3):
        super().__init__()
        self.out_modes, self.out_dim = out_modes, out_dim
        self.layers = nn.Sequential(
            nn.Linear((inp_dim + 16) * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dr_rate),
            nn.LayerNorm(512),
            nn.Linear(512, out_dim )
        )

    def pred_to_gmm(self, predictions):
        bs = predictions.shape[0]
        mean_predictions = predictions[:, :2 * self.out_modes].reshape(bs, self.out_modes, 2)
        # cov_predictions = predictions[:, 2 * self.out_modes:2 * self.out_modes + 4 * self.out_modes].reshape(bs,
        #                                                                                                      self.out_modes,
        #                                                                                                      2, 2)
        cov_predictions = 3 * torch.eye(2).unsqueeze(0).unsqueeze(0).repeat(bs, self.out_modes, 1, 1).to(
            mean_predictions.device)
        probs_predictions = predictions[:, -self.out_modes:]

        probs_predictions = torch.sigmoid(probs_predictions)
        mix = torch.distributions.Categorical(probs_predictions)

        scale_tril = torch.tril(((cov_predictions ** 2) + 1e-6))
        scale_tril[:, :, 1, 0] = cov_predictions[:, :, 1, 0]

        distr = torch.distributions.multivariate_normal.MultivariateNormal(mean_predictions, scale_tril=scale_tril)

        gmm = torch.distributions.MixtureSameFamily(mix, distr)
        return gmm

    def forward(self, hist_enc):
        bs = hist_enc.shape[0]
        predictions = self.layers(hist_enc).reshape(bs, -1)
        gmm = self.pred_to_gmm(predictions)
        return gmm, predictions


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


class DecoderTraj2(nn.Module):

    def __init__(self, inp_dim=32, inp_hor=12, out_modes=1, out_dim=2, out_horiz=12, dr_rate=0.1):
        super().__init__()
        self.out_modes, self.out_dim, self.out_horiz = out_modes, out_dim, out_horiz
        self.goal_embeder = nn.Sequential(nn.Linear(2, 8),
                                          nn.ReLU(),
                                          nn.Linear(8, inp_dim//2))
        out_shape = out_dim * out_modes * out_horiz + self.out_modes
        self.out_shape = out_shape
        # self.layers = nn.Sequential(
        #     nn.Linear(inp_dim + inp_dim//2 + 16, inp_dim * 2),
        #     nn.ReLU(),
        #     nn.Dropout(dr_rate),
        #     nn.LayerNorm(inp_dim * 2),
        #     nn.Linear(inp_dim * 2, inp_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dr_rate),
        #     nn.LayerNorm(inp_dim),
        #     nn.Linear(inp_dim, out_shape * 2)
        # )

        self.outlayers = nn.Sequential(
            nn.Linear(out_shape * 2 + 128, out_shape * 4),
            nn.ReLU(),

            nn.Linear(out_shape * 4, out_shape * 2),
            nn.ReLU(),
            nn.Dropout(dr_rate),
            nn.LayerNorm(out_shape * 2),
            nn.Linear(out_shape * 2, out_shape)
        )
        # self.att = nn.MultiheadAttention(out_shape * 2, out_modes, dropout=0.)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=out_shape * 2 + 128, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.q_mlp = nn.Linear(inp_dim + inp_dim//2 + 16, out_shape * 2)
        self.conf_mlp = nn.Sequential(nn.Linear(self.out_modes, self.out_modes), nn.ReLU(), nn.Linear(self.out_modes, self.out_modes))

    def forward(self, hist_enc, goal):
        bs = hist_enc.shape[0]
        goal_embedded = self.goal_embeder(goal)
        inp = torch.cat((hist_enc, goal_embedded), dim=2)
        inp_q = self.q_mlp(inp)

        agent_h_te = time_encoding_sin(inp_q, added_emb_dim=128)
        # src = self.q_mlp(agent_h_te).permute(1, 0, 2)
        predictions = self.transformer_encoder(agent_h_te.permute(1, 0, 2)).permute(1, 0, 2)


        predictions = self.outlayers(predictions)

        confidences = torch.softmax(self.conf_mlp(predictions[:, -self.out_modes:, -1]), dim=-1)
        trajectories = predictions[:, -1, :self.out_shape - self.out_modes].reshape(bs, self.out_modes, self.out_horiz,
                                                                                    self.out_dim)
        # trajectories = predictions[:, :-self.out_modes].reshape(bs, self.out_modes, self.out_horiz, self.out_dim)
        # trajectories = trajectories.cumsum(2)
        return trajectories, confidences


def gmm_to_mean_goal(gmm):
    return gmm.mean


def gmm_covariances(gmm):
    return gmm.component_distribution.covariance_matrix


def gmm_means(gmm):
    return gmm.component_distribution.loc


def create_rot_matrix(state_masked):
    cur_3d = torch.ones_like(state_masked[:, 0, :3])
    cur_3d[:, :2] = -state_masked[:, 0, :2].clone()
    T = torch.eye(3).unsqueeze(0).repeat(cur_3d.shape[0], 1, 1).to(cur_3d.device)
    T[:, :, 2] = cur_3d
    angles = torch.atan2(state_masked[:, 0, 2], state_masked[:, 0, 3])
    rot_mat = torch.eye(3).unsqueeze(0).repeat(cur_3d.shape[0], 1, 1).to(cur_3d.device)
    rot_mat[:, 0, 0] = torch.cos(angles)
    rot_mat[:, 1, 1] = torch.cos(angles)
    rot_mat[:, 0, 1] = -torch.sin(angles)
    rot_mat[:, 1, 0] = torch.sin(angles)
    transform = rot_mat @ T
    return transform


class AttPredictorPecNet(nn.Module):
    def __init__(self, inp_dim=32, embed_dim=128, num_blocks=8, out_modes=1, out_dim=2, out_horiz=12,
                 dr_rate=0.0, use_vis=True):
        super().__init__()
        self.pointNet = PointNetfeat(global_feat=True)
        self.visual = m = timm.create_model('efficientnetv2_rw_t', pretrained=True)
        self.visual.classifier = torch.nn.Identity()
        # self.visual.head = nn.Linear(self.visual.head.in_features, 1024)
        self.latent = nn.Parameter(torch.rand(out_modes, embed_dim + 16), requires_grad=True)
        self.embeder = InitEmbedding(inp_dim=4, out_dim=embed_dim)
        self.encoder = Encoder(inp_dim, embed_dim, num_blocks, use_vis=use_vis, dr_rate=dr_rate)
        self.decoder_goals = DecoderGoals(embed_dim, 12, out_modes, dr_rate=dr_rate)
        self.decoder_trajs = DecoderTraj2(embed_dim, 12, out_modes, out_dim, out_horiz, dr_rate)


    def forward(self, data):
        ### xyz emb
        bs = data["roadgraph_samples/xyz"].shape[0]
        masks = data["state/tracks_to_predict"].reshape(-1, 128) > 0
        bsr = masks.sum()  # num peds to predict, bs real
        rasters = rasterize_batch(data, True)
        maps = torch.tensor(np.concatenate(rasters)).permute(0, 3, 1, 2)/255.
        maps[:, :3] += maps[:, 3:].sum(1).unsqueeze(1)
        maps = maps[:, :3].cuda()
        me = self.visual(maps).cuda()
        # positional embedder
        # torch.tensor(np.stack(rasters[0] + rasters[1]))
        cur = torch.cat(
            [data["state/current/x"].reshape(-1, 128, 1, 1), data["state/current/y"].reshape(-1, 128, 1, 1)], -1)

        past = torch.cat([data["state/past/x"].reshape(-1, 128, 10, 1), data["state/past/y"].reshape(-1, 128, 10, 1)],
                         -1)  # .permute(0, 2, 1, 3)

        poses = torch.cat([cur, torch.flip(past, dims=[2])], dim=2).reshape(bs * 128, 11, -1).cuda()
        velocities = torch.zeros_like(poses)
        velocities[:, :-1] = poses[:, :-1] - poses[:, 1:]
        state = torch.cat([poses, velocities], dim=-1)
        state_masked = state.reshape(bs, 128, 11, -1)[masks]
        rot_mat = create_rot_matrix(state_masked)
        rot_mat_inv = torch.inverse(rot_mat)
        ### rotate cur state
        state_expanded = torch.cat([state_masked[:, :, :2], torch.ones_like(state_masked[:, :, :1])], -1)
        state_masked[:, :, :2] = torch.bmm(rot_mat, state_expanded.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :2]
        # state_masked[:, :-1, 2:] *= 0
        state_masked[:, :-1, 2:] = state_masked[:, :-1, :2] - state_masked[:, 1:, :2]
        agent_h_emb = self.embeder(state_masked)
        # pointnet embedder
        xyz = data["roadgraph_samples/xyz"].reshape(bs, -1, 3)[:, ::400].cuda()
        xyz_personal = torch.zeros([0, xyz.shape[1], xyz.shape[2]], device=xyz.device)
        ## rotate pointclouds
        # ...
        for i, index in enumerate(masks.nonzero()):
            xyz_p = torch.ones([xyz.shape[1], xyz.shape[2]], device=xyz.device)
            xyz_p[:, :2] = xyz[index[0], :, :2].clone()
            xyz_p = (rot_mat[i] @ xyz_p.T).T
            xyz_personal = torch.cat((xyz_personal, xyz_p.unsqueeze(0)), dim=0)
        xyz_emb, _, _ = self.pointNet(xyz_personal.permute(0, 2, 1))
        # xyz_emb = xyz_emb.repeat(8, 1)
        ####

        x = self.latent.unsqueeze(0).repeat(bsr, 1, 1)
        x = self.encoder(x, xyz_emb, agent_h_emb, me)
        gmm, goal_vector = self.decoder_goals(x)
        # goals = gmm.mean
        goals = gmm.component_distribution.mean.clone()
        # gmm.component_distribution.mean
        gmm.component_distribution.loc = torch.bmm(rot_mat_inv,
                          torch.cat([gmm.component_distribution.loc, torch.ones_like(gmm.component_distribution.mean[:, :, :1])], -1).permute(0, 2, 1)).permute(0, 2, 1)[:,
                :, :2]
        gmm.component_distribution.loc -= rot_mat_inv[:, :2, 2].unsqueeze(1)

        gt_goals = torch.cat([data["state/future/x"].reshape(-1, 128, 80, 1), data["state/future/y"].reshape(-1, 128, 80, 1)], -1)[data["state/tracks_to_predict"]>0][:, -1:].repeat(1, 6, 1)
        gt_goals = torch.cat([gt_goals, torch.ones_like(gt_goals[:, :, :1])], -1).to(goals.device)
        gt_goals = torch.bmm(rot_mat, gt_goals.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :2]

        ## find where no final goal at timestamp 80
        no_gt_future_indexes = data["state/future/valid"].reshape(-1,128,80)[data["state/tracks_to_predict"] > 0][:,-1] == 0
        gt_goals[no_gt_future_indexes] = gmm.component_distribution.loc[no_gt_future_indexes]

        predictions, confidences = self.decoder_trajs(x, gt_goals)
        ps = predictions.shape
        ## rotate goals back
        goals = torch.bmm(rot_mat_inv,
                          torch.cat([goals, torch.ones_like(goals[:, :, :1])], -1).permute(0, 2, 1)).permute(0, 2, 1)[:,
                :, :2]
        goals -= rot_mat_inv[:, :2, 2].unsqueeze(1)

        return predictions.permute(0, 2, 1, 3), confidences, goals, goal_vector,  rot_mat, rot_mat_inv


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

import cv2
import numpy as np
MAX_PIXEL_VALUE = 255
N_ROADS = 21
road_colors = [int(x) for x in np.linspace(1, MAX_PIXEL_VALUE, N_ROADS).astype("uint8")]

def rasterize_batch(parsed,
        validate,
        crop_size=512,
        raster_size=224,
        shift=2 ** 9,
        magic_const=3,
        n_channels=11,
):
    tracks_to_predict = parsed["state/tracks_to_predict"].cpu().numpy()
    past_x = parsed["state/past/x"].cpu().numpy().reshape(-1,128,10)
    past_y = parsed["state/past/y"].cpu().numpy().reshape(-1,128,10)
    current_x = parsed["state/current/x"].cpu().numpy()
    current_y = parsed["state/current/y"].cpu().numpy()
    current_yaw = parsed["state/current/bbox_yaw"].cpu().numpy()
    past_yaw = parsed["state/past/bbox_yaw"].cpu().numpy().reshape(-1,128,10)
    past_valid = parsed["state/past/valid"].cpu().numpy().reshape(-1,128,10)
    current_valid = parsed["state/current/valid"].cpu().numpy()
    agent_type = parsed["state/type"].cpu().numpy()
    roadlines_coords = parsed["roadgraph_samples/xyz"].cpu().numpy().reshape(-1,20000,3)
    roadlines_types = parsed["roadgraph_samples/type"].cpu().numpy()
    roadlines_valid = parsed["roadgraph_samples/valid"].cpu().numpy()
    roadlines_ids = parsed["roadgraph_samples/id"].cpu().numpy()
    widths = parsed["state/current/width"].cpu().numpy()
    lengths = parsed["state/current/length"].cpu().numpy()
    agents_ids = parsed["state/id"].cpu().numpy()
    tl_states = parsed["traffic_light_state/current/state"].cpu().numpy()
    tl_ids = np.ones([tl_states.shape[0]])
    tl_valids = parsed["traffic_light_state/current/valid"].cpu().numpy()
    future_x = parsed["state/future/x"].cpu().numpy().reshape(-1,128,80)
    future_y = parsed["state/future/y"].cpu().numpy().reshape(-1,128,80)
    future_valid = parsed["state/future/valid"].cpu().numpy().reshape(-1,128,80)
    #scenario_id = np.ones([tl_states.shape[0]])#.decode("utf-8")
    rasters = []
    for bn in range(future_y.shape[0]):
        raster = rasterize(
            tracks_to_predict[bn], past_x[bn], past_y[bn],
            current_x[bn], current_y[bn], current_yaw[bn],
            past_yaw[bn], past_valid[bn], current_valid[bn],
            agent_type[bn], roadlines_coords[bn], roadlines_types[bn],
            roadlines_valid[bn], roadlines_ids[bn], widths[bn],
            lengths[bn], agents_ids[bn], tl_states[bn],
            tl_ids[bn], tl_valids[bn], future_x[bn],
            future_y[bn], future_valid[bn], True,
            crop_size=512, raster_size=224, shift=2 ** 9,
            magic_const=3, n_channels=11,
        )
        rasters.append(raster)
    return rasters


def rasterize(tracks_to_predict, past_x, past_y, current_x, current_y, current_yaw, past_yaw, past_valid, current_valid,
              agent_type, roadlines_coords, roadlines_types, roadlines_valid, roadlines_ids, widths, lengths, agents_ids,
              tl_states, tl_ids, tl_valids, future_x, future_y, future_valid, validate, crop_size=512, raster_size=224,
              shift=2 ** 9, magic_const=3, n_channels=11):
    GRES = []
    displacement = np.array([[raster_size // 4, raster_size // 2]]) * shift
    tl_dict = {"green": set(), "yellow": set(), "red": set()}

    # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
    # Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
    for tl_state, tl_id, tl_valid in zip(
            tl_states.flatten(), tl_ids.flatten(), tl_valids.flatten()
    ):
        if tl_valid == 0:
            continue
        if tl_state in [1, 4, 7]:
            tl_dict["red"].add(tl_id)
        if tl_state in [2, 5, 8]:
            tl_dict["yellow"].add(tl_id)
        if tl_state in [3, 6]:
            tl_dict["green"].add(tl_id)

    XY = np.concatenate(
        (
            np.expand_dims(np.concatenate((past_x, current_x[:,np.newaxis]), axis=1), axis=-1),
            np.expand_dims(np.concatenate((past_y, current_y[:,np.newaxis]), axis=1), axis=-1),
        ),
        axis=-1,
    )

    GT_XY = np.concatenate(
        (np.expand_dims(future_x, axis=-1), np.expand_dims(future_y, axis=-1)), axis=-1
    )

    YAWS = np.concatenate((past_yaw, current_yaw[:,np.newaxis]), axis=1)

    agents_valid = np.concatenate((past_valid, current_valid[:,np.newaxis]), axis=1)

    roadlines_valid = roadlines_valid.reshape(-1)
    roadlines_coords = (
            roadlines_coords[:, :2][roadlines_valid > 0]
            * shift
            * magic_const
            * raster_size
            / crop_size
    )
    roadlines_types = roadlines_types[roadlines_valid > 0]
    roadlines_ids = roadlines_ids.reshape(-1)[roadlines_valid > 0]
    rasters = []
    for _, ( xy, current_val, val, _, yaw, agent_id, gt_xy, future_val, predict, ) \
            in enumerate(zip(XY, current_valid, agents_valid, agent_type, current_yaw.flatten(),
                              agents_ids, GT_XY, future_valid, tracks_to_predict.flatten())):
        if (not validate and future_val.sum() == 0) or (validate and predict == 0):
            continue
        if current_val == 0:
            continue

        RES_ROADMAP = (
                np.ones((raster_size, raster_size, 3), dtype=np.uint8) * MAX_PIXEL_VALUE
        )
        RES_EGO = [
            np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
            for _ in range(n_channels)
        ]
        RES_OTHER = [
            np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
            for _ in range(n_channels)
        ]

        xy_val = xy[val > 0]
        if len(xy_val) == 0:
            continue

        unscaled_center_xy = xy_val[-1].reshape(1, -1)
        center_xy = unscaled_center_xy * shift * magic_const * raster_size / crop_size
        rot_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ]
        )

        centered_roadlines = (roadlines_coords - center_xy) @ rot_matrix + displacement
        centered_others = (
                                  XY.reshape(-1, 2) * shift * magic_const * raster_size / crop_size
                                  - center_xy
                          ) @ rot_matrix + displacement
        centered_others = centered_others.reshape(128, n_channels, 2)
        centered_gt = (gt_xy - unscaled_center_xy) @ rot_matrix

        unique_road_ids = np.unique(roadlines_ids)
        for road_id in unique_road_ids:
            if road_id >= 0:
                roadline = centered_roadlines[roadlines_ids == road_id]
                road_type = roadlines_types[roadlines_ids == road_id].flatten()[0]

                road_color = road_colors[road_type]
                for c, rgb in zip(
                        ["green", "yellow", "red"],
                        [
                            (0, MAX_PIXEL_VALUE, 0),
                            (MAX_PIXEL_VALUE, 211, 0),
                            (MAX_PIXEL_VALUE, 0, 0),
                        ],
                ):
                    if road_id in tl_dict[c]:
                        road_color = rgb

                RES_ROADMAP = cv2.polylines(
                    RES_ROADMAP,
                    [roadline.astype(int)],
                    False,
                    road_color,
                    shift=9,
                )

        unique_agent_ids = np.unique(agents_ids)

        is_ego = False
        self_type = 0
        _tmp = 0
        for other_agent_id in unique_agent_ids:
            other_agent_id = int(other_agent_id)
            if other_agent_id < 1:
                continue
            if other_agent_id == agent_id:
                is_ego = True
                self_type = agent_type[agents_ids == other_agent_id]
            else:
                is_ego = False

            _tmp += 1
            agent_lane = centered_others[agents_ids == other_agent_id][0]
            agent_valid = agents_valid[agents_ids == other_agent_id]
            agent_yaw = YAWS[agents_ids == other_agent_id]

            agent_l = lengths[agents_ids == other_agent_id]
            agent_w = widths[agents_ids == other_agent_id]

            for timestamp, (coord, valid_coordinate, past_yaw,) in enumerate(
                    zip(
                        agent_lane,
                        agent_valid.flatten(),
                        agent_yaw.flatten(),
                    )
            ):
                if valid_coordinate == 0:
                    continue
                box_points = (
                        np.array(
                            [
                                -agent_l,
                                -agent_w,
                                agent_l,
                                -agent_w,
                                agent_l,
                                agent_w,
                                -agent_l,
                                agent_w,
                            ]
                        )
                        .reshape(4, 2)
                        .astype(np.float32)
                        * shift
                        * magic_const
                        / 2
                        * raster_size
                        / crop_size
                )

                box_points = (
                        box_points
                        @ np.array(
                    (
                        (np.cos(yaw - past_yaw), -np.sin(yaw - past_yaw)),
                        (np.sin(yaw - past_yaw), np.cos(yaw - past_yaw)),
                    )
                ).reshape(2, 2)
                )

                _coord = np.array([coord])

                box_points = box_points + _coord
                box_points = box_points.reshape(1, -1, 2).astype(np.int32)

                if is_ego:
                    cv2.fillPoly(
                        RES_EGO[timestamp],
                        box_points,
                        color=MAX_PIXEL_VALUE,
                        shift=9,
                    )
                else:
                    cv2.fillPoly(
                        RES_OTHER[timestamp],
                        box_points,
                        color=MAX_PIXEL_VALUE,
                        shift=9,
                    )

        raster = np.concatenate([RES_ROADMAP] + RES_EGO + RES_OTHER, axis=2)
        rasters.append(raster)
    return rasters

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
    rasterize(
        data["state/tracks_to_predict"].numpy(),
        data["state/past/x"].numpy(),
        data["state/past/y"].numpy(),
        data["state/current/x"].numpy(),
        data["state/current/y"].numpy(),
        # parsed["state/current/bbox_yaw"].numpy(),
        # parsed["state/past/bbox_yaw"].numpy(),
        parsed["state/past/valid"].numpy(),
        parsed["state/current/valid"].numpy(),
        parsed["state/type"].numpy(),
        parsed["roadgraph_samples/xyz"].numpy(),
        parsed["roadgraph_samples/type"].numpy(),
        parsed["roadgraph_samples/valid"].numpy(),
        parsed["roadgraph_samples/id"].numpy(),
        parsed["state/current/width"].numpy(),
        parsed["state/current/length"].numpy(),
        parsed["state/id"].numpy(),
        parsed["traffic_light_state/current/state"].numpy(),
        parsed["traffic_light_state/current/id"].numpy(),
        parsed["traffic_light_state/current/valid"].numpy(),
        parsed["state/future/x"].numpy(),
        parsed["state/future/y"].numpy(),
        parsed["state/future/valid"].numpy(),
        parsed["scenario/id"].numpy()[0].decode("utf-8"),
        validate=validate,)

    # out = model(data)
    print(1)
