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
# shallow_: before 2000, coefs 0.001/100itr, phis 0.01: (too low frequency)
# shallow_: after 2000, coefs 0.001/1000itr, phis 0.01

# lr_01_: 0-5000, coefs 0.01/100itr, phis 0.01: (too high frequency)

# lr_01_: 0-16800, coefs 0.01/100itr, phis 0.001: (not converging)
# lr_01_: 16800-120000, coefs 0.01/100itr, phis 0.0001: smoother filters (not converging)
# lr_01_: 120000-200000, coefs 0.01/100itr, phis 0.00001: smoother filter (not changing)
# lr_01_: 200000-210000, coefs 0.01/100itr, phis 0.0001, batch_size(25): (not changing)
# lr_01_: 210000-, coefs 0.01/100itr, phis 0.001, batch_size(25):
load_idx = 210000
if load_idx >= 0:
    #hgm.phis_list = torch.load("shallow/shallow_sampled_phis_list_{}.torch".format(load_idx))
    hgm.phis_list = torch.load("shallow/lr_01_shallow_sampled_phis_list_{}.torch".format(load_idx))
for l in range(hgm.n_layers):
    hgm.phis_list[l] = hgm.phis_list[l].to(device)

import random
random.seed(0)
for epoch in range(1000):
    if epoch > 0:
        random.shuffle(pictures)

    mb_size = 25
    video = torch.zeros(mb_size, 1, 12, 12, 1, device=device, dtype=torch.float32)
    for (batch_idx, picture) in tqdm.tqdm(enumerate(pictures)):
        picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY) / 255.
        start_x = np.random.randint(0, 256-12)
        start_y = np.random.randint(0, 256-12)
        picture = picture[start_y:start_y+12, start_x:start_x+12]
        video[batch_idx % mb_size] = torch.tensor(picture.reshape(1, 12, 12, 1), device=device, dtype=torch.float32)

        if batch_idx % mb_size == mb_size-1:
            if False and 5000*epoch + batch_idx == max(load_idx-1, 0):
                pickle.dump(video[0, 0, :, :, 0], open("shallow/lr_01_shallow_sampled_video_{}.pkl".format(load_idx), "wb"))
                inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, use_sparse=False,
                    lr=0.001, max_itr=10000, vars_abs_stop_cond=0.0001, vars_rel_stop_cond=0.0001)
                torch.save(inferred_coefs_list, "shallow/lr_01_shallow_sampled_inferred_coefs_list_{}.torch".format(load_idx))
                reconstr_video = hgm.generate_video(inferred_coefs_list, hgm.phis_list, use_sparse=False)[0]
                pickle.dump(reconstr_video[0, 0, :, :, 0], open("shallow/lr_01_shallow_sampled_reconstr_video_{}.pkl".format(load_idx), "wb"))
                print(1/0)

            if 5000*epoch + batch_idx >= load_idx:
                print("batch_idx:", 5000*epoch + batch_idx)
                #inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, use_sparse=False,
                #    lr=0.001, max_itr=1000, vars_abs_stop_cond=0.0001, vars_rel_stop_cond=0.0001)
                inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, use_sparse=False,
                    lr=0.01, max_itr=100, vars_abs_stop_cond=0.001, vars_rel_stop_cond=0.001)
                #hgm.update_phis(video, inferred_coefs_list, use_sparse=False,
                #    use_warm_start_optimizer=warm_start, lr=0.01)
                hgm.update_phis(video, inferred_coefs_list, use_sparse=False,
                    use_warm_start_optimizer=warm_start, lr=0.001)

                warm_start = True

            if (5000*epoch + batch_idx+1) % 100 == 0:
                #torch.save(hgm.phis_list, "shallow/shallow_sampled_phis_list_{}.torch".format(5000*epoch + batch_idx+1))
                torch.save(hgm.phis_list, "shallow/lr_01_shallow_sampled_phis_list_{}.torch".format(5000*epoch + batch_idx+1))
