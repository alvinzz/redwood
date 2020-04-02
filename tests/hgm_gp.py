from math import exp
import numpy as np
import matplotlib.pyplot as plt
import pickle

def rbf_kernel(x1, x2, variance = 2):
    return exp(-1 * ((x1-x2) ** 2) / (2*variance))

def gram_matrix(xs):
    return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]

#xs = np.arange(9)
#mean = [0 for x in xs]
#gram = gram_matrix(xs)

#data = []
#for i in range(100):
#    ys = np.random.multivariate_normal(mean, gram)
#    ys -= np.min(ys)
#    ys /= np.max(ys)
#    data.append(ys)
#data = np.array(data)

#import pickle
#pickle.dump(data, open("hgm_gp/data.pkl", "wb"))

from generative_model import HierarchicalGenerativeModel as HGM
from viz import viz_phis

import torch

device = "cuda"
hgm = HGM(
    [
        torch.normal(torch.zeros(20, 1, 1, 3, 1), torch.ones(20, 1, 1, 3, 1)),
        torch.normal(torch.zeros(50, 1, 1, 7, 20*8), torch.ones(50, 1, 1, 7, 20*8)),
    ],
    [1, 1], [1, 1], [1, 1],
    [1, 1], [1, 1], [1/3, 1/3],
    device=device
)

plt.imshow(pickle.load(open("hgm_gp/data.pkl", "rb"))[::10], cmap="gray", vmin=0, vmax=1)
plt.show()

videos = torch.tensor(pickle.load(open("hgm_gp/data.pkl", "rb")), dtype=torch.float32, device=device)
videos = videos.reshape(100, 1, 1, 9, 1)

if False:
    import tqdm
    for itr in range(1):
        print("Itr:", itr)
        for v in range(videos.shape[0]):
            print("Video:", v)
            video = videos[v:v+1]
            inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list,
                max_itr=100, init_lr=0.01,
                vars_rel_stop_cond=0.05,
                vars_abs_stop_cond=0.0,
                loss_rel_stop_cond=0.05,
                loss_abs_stop_cond=0.0,
                use_sparse=False)
            hgm.update_phis(video, inferred_coefs_list, use_warm_start_optimizer=(itr>0 or v>0), use_sparse=False)

    # inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(videos[0:1], hgm.phis_list)
    # torch.save(inferred_vars_list, "hgm_gp/rand_init/vars_list.torch")
    # torch.save(inferred_coefs_list, "hgm_gp/rand_init/coefs_list.torch")
    torch.save(hgm.phis_list, "hgm_gp/rand_init/100_01_01_phis_list_1.torch")
    # TODO: plot_coefs
    # viz_phis(hgm.phis_list[0], "hgm_gp/rand_init/phis0")
    # viz_phis(hgm.phis_list[1], "hgm_gp/rand_init/phis1")
    # hgm.generate_video(inferred_coefs_list, hgm.phis_list, plot_save_dir="hgm_gp/rand_init/video")

if False:
    hgm.phis_list = torch.load(open("hgm_gp/rand_init/100_01_01_phis_list_1.torch", "rb"))
    for i in range(2):
        hgm.phis_list[i] = hgm.phis_list[i].to(device)
    inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(videos[::10], hgm.phis_list,
        max_itr=1000, init_lr=0.001,
        vars_abs_stop_cond=0.0,
        vars_rel_stop_cond=0.0,
        loss_rel_stop_cond=0.0,
        loss_abs_stop_cond=0.0,
        use_sparse=False)
    inferred_coefs_list[0][0] = torch.zeros_like(inferred_coefs_list[0][0])
    inferred_coefs_list[0][1] = torch.zeros_like(inferred_coefs_list[0][1])
    inferred_coefs_list[0][2] = torch.zeros_like(inferred_coefs_list[0][2])
    inferred_coefs_list[0][3] = torch.zeros_like(inferred_coefs_list[0][3])
    inferred_coefs_list[0][4] = torch.ones_like(inferred_coefs_list[0][4])
    inferred_coefs_list[0][5] = torch.ones_like(inferred_coefs_list[0][5])
    inferred_coefs_list[0][6] = torch.ones_like(inferred_coefs_list[0][6])
    inferred_coefs_list[0][7] = torch.zeros_like(inferred_coefs_list[0][7])
    hgm.generate_video(inferred_coefs_list, hgm.phis_list, plot_save_dir="hgm_gp/rand_init/video_100_01_01_1")

if False:
    vs = []
    for ch in range(50):
        hgm.phis_list = torch.load(open("hgm_gp/rand_init/100_01_01_phis_list_1.torch", "rb"))
        for i in range(2):
            hgm.phis_list[i] = hgm.phis_list[i].to(device)
        inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(videos[0:1], hgm.phis_list,
            max_itr=1, init_lr=0., vars_abs_stop_cond=0., vars_rel_stop_cond=0., use_sparse=False)
        inferred_coefs_list[0][0] = torch.zeros_like(inferred_coefs_list[0][0])
        inferred_coefs_list[0][1] = torch.zeros_like(inferred_coefs_list[0][1])
        inferred_coefs_list[0][2] = torch.zeros_like(inferred_coefs_list[0][2])
        inferred_coefs_list[0][3] = torch.zeros_like(inferred_coefs_list[0][3])
        inferred_coefs_list[0][4] = torch.ones_like(inferred_coefs_list[0][4])
        inferred_coefs_list[0][5] = torch.ones_like(inferred_coefs_list[0][5])
        inferred_coefs_list[0][6] = torch.ones_like(inferred_coefs_list[0][6])
        inferred_coefs_list[0][7] = torch.zeros_like(inferred_coefs_list[0][7])

        inferred_coefs_list[1][0] = torch.zeros_like(inferred_coefs_list[1][0])
        inferred_coefs_list[1][1] = torch.zeros_like(inferred_coefs_list[1][1])
        inferred_coefs_list[1][2] = torch.zeros_like(inferred_coefs_list[1][2])
        inferred_coefs_list[1][3] = torch.zeros_like(inferred_coefs_list[1][3])
        inferred_coefs_list[1][4] = torch.ones_like(inferred_coefs_list[1][4])
        inferred_coefs_list[1][5] = torch.ones_like(inferred_coefs_list[1][5])
        inferred_coefs_list[1][6] = torch.ones_like(inferred_coefs_list[1][6])
        inferred_coefs_list[1][7] = torch.zeros_like(inferred_coefs_list[1][7])

        inferred_coefs_list[1][0][0, 0, 0, 0, 0, ch] = 1
        v = hgm.generate_video(inferred_coefs_list, hgm.phis_list)[0]
        v = np.squeeze(v.detach().cpu().numpy())
        v -= np.min(v)
        #v /= np.max(v)
        vs.append(v)

    vs = np.array(vs)
    plt.imshow(vs, cmap="gray")
    plt.show()

##### itr 1, coefs 100, 0.01, phis 0.01
#reconstr L_0 loss: tensor(0.6486, grad_fn=<SumBackward0>)
#reconstr L_1 loss: tensor(1.4253, grad_fn=<SumBackward0>)
#vars_0 sparsity loss: tensor(1.4253, grad_fn=<SumBackward0>)
#vars_1 sparsity loss: tensor(1.9787, grad_fn=<SumBackward0>)
# test on 10 images, coefs 1000, 0.001
#reconstr L_0 loss: tensor(0.4669, grad_fn=<SumBackward0>)
#reconstr L_1 loss: tensor(3.0783, grad_fn=<SumBackward0>)
#vars_0 sparsity loss: tensor(3.0783, grad_fn=<SumBackward0>)
#vars_1 sparsity loss: tensor(18.3885, grad_fn=<SumBackward0>)
# good deep reconstructions

##### itr 2, coefs 100, 0.001, phis 0.01
#reconstr L_0 loss: tensor(0.3075, grad_fn=<SumBackward0>)
#reconstr L_1 loss: tensor(1.6240, grad_fn=<SumBackward0>)
#vars_0 sparsity loss: tensor(1.6240, grad_fn=<SumBackward0>)
#vars_1 sparsity loss: tensor(1.4170, grad_fn=<SumBackward0>)
# test on 10 images, coefs 1000, 0.001
#reconstr L_0 loss: tensor(0.6972, grad_fn=<SumBackward0>)
#reconstr L_1 loss: tensor(4.2450, grad_fn=<SumBackward0>)
#vars_0 sparsity loss: tensor(4.2450, grad_fn=<SumBackward0>)
#vars_1 sparsity loss: tensor(11.9582, grad_fn=<SumBackward0>)
# good deep reconstructions

