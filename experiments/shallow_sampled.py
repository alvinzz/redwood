from torchvision import datasets, transforms

from generative_model import HierarchicalGenerativeModel as HGM
from viz import viz_phis

import torch
import numpy as np

device = "cuda"

import tqdm

hgm = HGM(
    [
        torch.normal(torch.zeros(144, 1, 12, 12, 1), torch.ones(144, 1, 12, 12, 1)),
    ],
    [1], [1], [1],
    [1], [4/12], [4/12],
    device=device,
)

import pickle
#videos = pickle.load(open("shallow/videos.pkl", "rb"))
import glob
pictures = glob.glob("shallow/places365_standard/train/japanese_garden/*")
import cv2
pictures = [cv2.imread(picture, cv2.IMREAD_COLOR) for picture in pictures]

warm_start = False
load_idx = -1
if load_idx >= 0:
    hgm.phis_list = torch.load("shallow/shallow_sampled_phis_list_{}.torch".format(load_idx))
for l in range(hgm.n_layers):
    hgm.phis_list[l] = hgm.phis_list[l].to(device)

for (batch_idx, picture) in tqdm.tqdm(enumerate(pictures)):
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY) / 255.
    video = picture.reshape(1, 1, 256, 256, 1)

    if False and batch_idx == max(load_idx-1, 0):
        video = torch.tensor(video, device=device, dtype=torch.float32)
        start_x = np.random.randint(0, 256-12)
        start_y = np.random.randint(0, 256-12)
        video = video[:, :, start_y:start_y+12, start_x:start_x+12, :]
        pickle.dump(video[0, 0, :, :, 0], open("shallow/shallow_sampled_video_{}.pkl".format(load_idx), "wb"))
        inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, use_sparse=True,
            lr=0.001, max_itr=1000, abs_grad_stop_cond=0.0001, rel_grad_stop_cond=0.0001)
        torch.save(inferred_coefs_list, "shallow/shallow_sampled_inferred_coefs_list_{}.torch".format(load_idx))
        reconstr_video = hgm.generate_video(inferred_coefs_list, hgm.phis_list)[0]
        pickle.dump(reconstr_video[0, 0, :, :, 0], open("shallow/shallow_sampled_reconstr_video_{}.pkl".format(load_idx), "wb"))

    if batch_idx >= load_idx:
        video = torch.tensor(video, device=device, dtype=torch.float32)
        start_x = np.random.randint(0, 256-12)
        start_y = np.random.randint(0, 256-12)
        video = video[:, :, start_y:start_y+12, start_x:start_x+12, :]

        print("batch_idx:", batch_idx)
        #inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, use_sparse=True,
        #    lr=0.001, max_itr=100, abs_grad_stop_cond=0.0001, rel_grad_stop_cond=0.0001)
        inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, use_sparse=True,
            lr=0.01, max_itr=100, abs_grad_stop_cond=0.001, rel_grad_stop_cond=0.001)
        hgm.update_phis(video, inferred_coefs_list, use_sparse=True,
            use_warm_start_optimizer=warm_start, lr=0.01)

        warm_start = True

    if batch_idx % 100 == 0:
        torch.save(hgm.phis_list, "shallow/lr_01_shallow_sampled_phis_list_{}.torch".format(batch_idx))
