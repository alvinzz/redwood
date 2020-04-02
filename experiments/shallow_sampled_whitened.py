import numpy as np
import matplotlib.pyplot as plt

def whiten_picture(picture):
    # xs = np.linspace(0, 128)
    # ys = xs * np.exp(-(xs / 100)**4)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.plot(xs, ys)
    # plt.show()

    # im = np.zeros([256, 256])
    # for x in range(-128, 128):
    #    for y in range(-128, 128):
    #        dist = np.sqrt(x**2 + y**2)
    #        im[x+128, y+128] = dist * np.exp(-(dist / 100)**4)
    # plt.imshow(im, cmap="gray")
    # plt.show()

    f = np.fft.fftshift(np.fft.fft2(picture))
    for x in range(-128, 128):
       for y in range(-128, 128):
           dist = np.sqrt(x**2 + y**2)
           f[x+128, y+128] *= dist * np.exp(-(dist / 100)**4)
    whitened_picture = np.fft.ifft2(np.fft.ifftshift(f)).real
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(whitened_picture, cmap="gray")
    # ax = fig.add_subplot(1, 2, 2)
    # ax.imshow(picture, cmap="gray")
    # plt.show()

    return whitened_picture

from torchvision import datasets, transforms

from generative_model import HierarchicalGenerativeModel as HGM
from viz import viz_phis

import torch
import numpy as np

device = "cuda"

import tqdm

hgm = HGM(
    [
        torch.normal(torch.zeros(64, 1, 8, 8, 1), torch.ones(64, 1, 8, 8, 1)),
    ],
    [1], [1], [1],
    [1], [2/8], [2/8],
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
    hgm.phis_list = torch.load("shallow/shallow_sampled_whitened_phis_list_{}.torch".format(load_idx))
for l in range(hgm.n_layers):
    hgm.phis_list[l] = hgm.phis_list[l].to(device)

import random
random.seed(0)
for epoch in range(max(0, load_idx // 5000), 1000):
    if epoch > 0:
        random.shuffle(pictures)

    mb_size = 1
    video = torch.zeros(mb_size, 1, 8, 8, 1, device=device, dtype=torch.float32)
    for (batch_idx, picture) in tqdm.tqdm(enumerate(pictures)):
        picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY) / 255.
        picture = whiten_picture(picture)
        start_x = np.random.randint(0, 256-8)
        start_y = np.random.randint(0, 256-8)
        picture = picture[start_y:start_y+8, start_x:start_x+8]
        video[batch_idx % mb_size] = torch.tensor(picture.reshape(1, 8, 8, 1), device=device, dtype=torch.float32)

        if batch_idx % mb_size == mb_size-1:
            if False and 5000*epoch + batch_idx == max(load_idx-1, 0):
                pickle.dump(video[0, 0, :, :, 0], open("shallow/shallow_sampled_whitened_video_{}.pkl".format(load_idx), "wb"))
                inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, use_sparse=False,
                    lr=0.001, max_itr=10000, vars_abs_stop_cond=0.0001, vars_rel_stop_cond=0.0001)
                torch.save(inferred_coefs_list, "shallow/shallow_sampled_whitened_inferred_coefs_list_{}.torch".format(load_idx))
                reconstr_video = hgm.generate_video(inferred_coefs_list, hgm.phis_list, use_sparse=False)[0]
                pickle.dump(reconstr_video[0, 0, :, :, 0], open("shallow/shallow_sampled_whitened_reconstr_video_{}.pkl".format(load_idx), "wb"))
                print(1/0)

            if 5000*epoch + batch_idx >= load_idx:
                print("batch_idx:", 5000*epoch + batch_idx)
                inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(
                    video, hgm.phis_list,
                    max_itr=1000,
                    vars_rel_stop_cond=0.05,
                    vars_abs_stop_cond=0.0,
                    loss_rel_stop_cond=0.05,
                    loss_abs_stop_cond=0.0,
                    loss_stop_cond_window=10,
                    momentum_decay=0.9,
                    init_lr=0.1,
                    use_sparse=False)
                hgm.update_phis(video, inferred_coefs_list, use_sparse=False,
                    use_warm_start_optimizer=warm_start, lr=0.01)

                warm_start = True

            if (5000*epoch + batch_idx+1) % 1000 == 0:
                torch.save(hgm.phis_list, "shallow/shallow_sampled_whitened_phis_list_{}.torch".format(5000*epoch + batch_idx+1))
