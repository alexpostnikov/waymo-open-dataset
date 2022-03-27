import torch
from torch import nn as nn
import os

class MultyModel(nn.Module):
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
        self.future_tr = nn.Transformer(d_model=256, nhead=16, num_encoder_layers=4)  # .cuda()
        self.lin_fut = nn.Linear(88, 256)
        self.dec = nn.Sequential(nn.Linear((24 + 256), 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256))

        self.dec_f = nn.Sequential(nn.Linear(22+256, 512),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(128),
                                 nn.Linear(512, (160+1)*6))

    def forward(self, data):
        bs = data["roadgraph_samples/xyz"].shape[0]

        xyz = data["roadgraph_samples/xyz"].reshape(bs, -1, 3).cuda()
        xyz_mean = data["roadgraph_samples/xyz"].reshape(bs, -1, 3).mean(1).reshape(bs,1,3).cuda()
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
        state = (torch.cat([cur, past], 1).permute(0, 2, 1, 3).cuda() - xyz_mean[:, :, :2].reshape(bs, 1, 1, 2)).reshape(-1, 128, 11 * 2)

        state_emb = self.lin_hist(state)
        tgt = torch.rand(bs, 128, 24).cuda()
        out_1 = self.hist_tr(state_emb.permute(1, 0, 2), tgt.permute(1, 0, 2)).permute(1, 0, 2)
        out_2 = self.lin_fut(torch.cat([out_0, out_1], -1))
        # future_tgt = cur.reshape(-1, 1, 128, 2).repeat(1, 80, 1, 1).permute(0, 2, 1, 3).cuda()
        # future_tgt -= xyz_mean[:, :, :2].reshape(bs, 1, 1, 2)
        # future_tgt = future_tgt.reshape(-1, 128, 160)
        future_tgt = torch.rand(bs, 128, 256).cuda()
        # cur.reshape(-1, 1, 128, 2).repeat(1, 80, 1, 1).permute(0, 2, 1, 3).reshape(-1, 128, 160).cuda()
        out_3 = self.future_tr(out_2.permute(1, 0, 2), future_tgt.permute(1, 0, 2))
        out_3 = out_3.permute(1, 0, 2).reshape(bs, 128, -1)  # bs, 128, 80 ,2

        fin_input = torch.cat([state_emb, out_3], -1)
        out_dec = self.dec(fin_input)#.reshape(-1, 128, 80, 2)
        cur = torch.cat(
            [data["state/current/x"].reshape(-1, 1, 128, 1), data["state/current/y"].reshape(-1, 1, 128, 1)], -1)
        past = torch.cat([data["state/past/x"].reshape(-1, 128, 10, 1), data["state/past/y"].reshape(-1, 128, 10, 1)],
                         -1).permute(0, 2, 1, 3)
        state = (torch.cat([cur, past], 1) - cur).permute(0, 2, 1, 3).reshape(-1, 128, 11 * 2).cuda()
        out = self.dec_f(torch.cat([out_dec, state], dim=2))
        poses = out[..., :-6].reshape(bs, 128, 80, 6, 2)
        confs = torch.nn.functional.softmax(out[..., -6:].reshape(bs, 128, 6), -1)
        return poses, confs #out.reshape(-1, 128, 80, 2)

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

        self.dec_f = nn.Sequential(nn.Linear(22+64, 64),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(128),
                                 nn.Linear(64, 160))
    def forward(self, data):
        bs = data["roadgraph_samples/xyz"].shape[0]

        xyz = data["roadgraph_samples/xyz"].reshape(bs, -1, 3).cuda()
        xyz_mean = data["roadgraph_samples/xyz"].reshape(bs, -1, 3).mean(1).reshape(bs,1,3).cuda()
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
        state = (torch.cat([cur, past], 1).permute(0, 2, 1, 3).cuda() - xyz_mean[:, :, :2].reshape(bs, 1, 1, 2)).reshape(-1, 128, 11 * 2)

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
        out_dec = self.dec(fin_input)#.reshape(-1, 128, 80, 2)
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