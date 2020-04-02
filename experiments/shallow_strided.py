from torchvision import datasets, transforms

from generative_model import HierarchicalGenerativeModel as HGM
from viz import viz_phis

import torch
import numpy as np

device = "cpu"

import tqdm

hgm = HGM(
    [
        torch.normal(torch.zeros(50, 1, 12, 12, 1), torch.ones(50, 1, 12, 12, 1)),
    ],
    [1], [1], [1],
    [1], [4/12], [4/12],
    device=device,
)

import pickle
videos = pickle.load(open("shallow/videos.pkl", "rb"))

warm_start = False
for (batch_idx, video) in tqdm.tqdm(enumerate(videos)):
    if batch_idx >= 0:
        video = torch.tensor(video, device=device, dtype=torch.float32).unsqueeze(0)

        print("batch_idx:", batch_idx)
        inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, use_sparse=True,
            lr=0.01, max_itr=100, vars_abs_stop_cond=0.001, vars_rel_stop_cond=0.001)
        hgm.update_phis(video, inferred_coefs_list, use_sparse=True,
            use_warm_start_optimizer=warm_start, lr=0.01)

        warm_start = True

    if batch_idx % 1000 == 0:
        torch.save(hgm.phis_list, "shallow/shallow_strided_phis_list_{}.torch".format(batch_idx))
