import sys
import os
import torch
import numpy as np
from torch import nn

from paths import S3D_PATH

sys.path.append(S3D_PATH)
from s3dg import S3D


S3D_PATH = "/user/n.dvornik/Git/S3D_HowTo100M/"
global_device = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, device=None):
        super(Net, self).__init__()
        device = device if device is not None else global_device
        self.net = S3D(os.path.join(S3D_PATH, "s3d_dict.npy"))
        state_dict = torch.load(os.path.join(S3D_PATH, "s3d_howto100m.pth"))
        self.net.load_state_dict(state_dict)
        self.net.to(device)
        self.net.eval()

    @torch.no_grad()
    def retrieve(self, texts, videos):
        # video frames have to be normalized in [0, 1]
        video_descriptors = torch.cat([self.net(v[None, ...].cuda())["video_embedding"] for v in videos], 0)
        text_descriptors = self.net.text_module(texts)["text_embedding"]
        scores = text_descriptors @ video_descriptors.t()

        decr_sim_inds = torch.argsort(scores, descending=True, dim=1)
        outs = []
        for i in range(len(texts)):
            sorted_videos = [{"video_ind": j, "score": scores[i, j]} for j in decr_sim_inds[i]]
            outs.append(sorted_videos)
        return outs

    @torch.no_grad()
    def embed_full_video(self, frames):
        # assuming the video is at 10fps and that we take 32 frames
        num_frames = 32

        # frames is a tensor of size [T, W, H, 3]
        T, W, H, _ = frames.shape
        frames = frames.permute(3, 0, 1, 2)
        N_chunks = T // num_frames
        n_last_frames = T % num_frames
        if n_last_frames > 0:
            zeros = torch.zeros((3, num_frames - n_last_frames, W, H), dtype=torch.uint8)
            frames = torch.cat((frames, zeros), axis=1)
            N_chunks += 1

        # extract features
        chunk_features = []
        for i in range(0, N_chunks):
            chunk_frames = frames[:, i * num_frames : (i + 1) * num_frames, ...][None, ...]
            chunk_feat = self.net(chunk_frames.cuda())["video_embedding"]
            chunk_features.append(chunk_feat)

        chunk_features = torch.cat(chunk_features, 0)
        return chunk_features

    @torch.no_grad()
    def embed_full_subs(self, subs):
        clipped_subs = [" ".join(s.split(" ")[:30]) for s in subs]
        sub_features = self.net.text_module(clipped_subs)["text_embedding"]
        return sub_features


def get_text_encoder(device=None):
    model = Net(device)
    return model.embed_full_subs
