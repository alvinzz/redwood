from generative_model import HierarchicalGenerativeModel as HGM
from viz import viz_phis

import torch

device = "cuda"

# a: [batch, T, N_y, N_x, [CH], n_philters]
# phis: [[N_y, N_x], [CH], n_philters, t, n_y, n_x, ch]
PHIS_0 = torch.tensor(
    [
        [0.1, 0.3, 0.6],
        [0.3, 0.6, 0.1],
        [0.6, 0.1, 0.3],
    ], dtype=torch.float32).reshape(
    3, 1, 1, 3, 1)
PHIS_1 = torch.tensor(
    [
        [
            [0, 0.5, 0.5],
            [0.33, 0.33, 0.33],
            [1, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ], dtype=torch.float32).reshape(
    8, 1, 1, 1, 3, 3)
coefs_list = [
    [
        torch.zeros(1, 1, 1, 3, 1, 3, device=device),
        torch.zeros(1, 1, 1, 3, 1, 3, device=device),
        torch.zeros(1, 1, 1, 3, 1, 3, device=device),
        torch.zeros(1, 1, 1, 3, 1, 3, device=device),
        torch.ones(1, 1, 1, 3, 1, 3, device=device),
        torch.ones(1, 1, 1, 3, 1, 3, device=device),
        torch.ones(1, 1, 1, 3, 1, 3, device=device),
        torch.zeros(1, 1, 1, 3, 1, 3, device=device),
    ],
    [
        3*torch.ones(1, 1, 1, 1, 8, 1, device=device),
        torch.zeros(1, 1, 1, 1, 8, 1, device=device),
        torch.zeros(1, 1, 1, 1, 8, 1, device=device),
        torch.zeros(1, 1, 1, 1, 8, 1, device=device),
        torch.ones(1, 1, 1, 1, 8, 1, device=device),
        torch.ones(1, 1, 1, 1, 8, 1, device=device),
        torch.ones(1, 1, 1, 1, 8, 1, device=device),
        torch.zeros(1, 1, 1, 1, 8, 1, device=device),
    ],
]

hgm = HGM([PHIS_0, PHIS_1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], device=device)
video = hgm.generate_video(coefs_list, hgm.phis_list, plot_save_dir="hgm/reference/video")[0]
# torch.save(coefs_list, "hgm/reference/coefs_list.torch")
# torch.save(hgm.phis_list, "hgm/reference/phis_list.torch")
# # TODO: plot_coefs
# viz_phis(PHIS_0, "hgm/reference/phis0")
# viz_phis(PHIS_1, "hgm/reference/phis1")

# ######### infer coefs + phis, phis starting from reference initialization ###########
# hgm = HGM([PHIS_0, PHIS_1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], device=device)
# inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=10)
# hgm.update_phis(video, inferred_coefs_list, use_warm_start_optimizer=False)

# for _ in range(50):
#     inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=10,
#         warm_start_vars_list=inferred_vars_list)
#     hgm.update_phis(video, inferred_coefs_list, use_warm_start_optimizer=True)

# inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=10,
#     warm_start_vars_list=inferred_vars_list)
# hgm.generate_video(inferred_coefs_list, hgm.phis_list, plot_save_dir="hgm/ref_init/video").detach()
# torch.save(inferred_coefs_list, "hgm/ref_init/coefs_list.torch")
# torch.save(hgm.phis_list, "hgm/ref_init/phis_list.torch")
# # TODO: plot_coefs
# viz_phis(hgm.phis_list[0], "hgm/ref_init/phis0")
# viz_phis(hgm.phis_list[1], "hgm/ref_init/phis1")

# TODO: fix alg so that this works

######### infer coefs + phis, phis starting from random initialization ###########
# phis: [[N_y, N_x], [CH], n_philters, t, n_y, n_x, ch]
hgm = HGM(
    [
        torch.normal(torch.zeros(20, 1, 1, 3, 1), torch.ones(20, 1, 1, 3, 1)),
        torch.normal(torch.zeros(50, 1, 1, 3, 20*8), torch.ones(50, 1, 1, 3, 20*8)),
    ],
    [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
    device=device
)
inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=100)
hgm.update_phis(video, inferred_coefs_list, use_warm_start_optimizer=False)

for _ in range(250):
    inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=100)
    hgm.update_phis(video, inferred_coefs_list, use_warm_start_optimizer=True)

inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=100)
hgm.generate_video(inferred_coefs_list, hgm.phis_list, plot_save_dir="hgm/rand_init/video")
torch.save(inferred_vars_list, "hgm/rand_init/vars_list.torch")
torch.save(inferred_coefs_list, "hgm/rand_init/coefs_list.torch")
torch.save(hgm.phis_list, "hgm/rand_init/phis_list.torch")
# TODO: plot_coefs
viz_phis(hgm.phis_list[0], "hgm/rand_init/phis0")
# viz_phis(hgm.phis_list[1], "hgm/rand_init/phis1")

# mean all, 250 itr, coefs no aux, phis aux
# tensor(15.3973, device='cuda:0', grad_fn=<SumBackward0>)
# tensor(0.7935, device='cuda:0', grad_fn=<SumBackward0>)

# 250 itr, coefs sum no aux, phis mean aux
# tensor(2.3659, device='cuda:0', grad_fn=<SumBackward0>)
# tensor(1.9482, device='cuda:0', grad_fn=<SumBackward0>)

# 250 itr, coefs sum (aux mean), phis mean aux
# tensor(2.1947, device='cuda:0', grad_fn=<SumBackward0>)
# tensor(1.6806, device='cuda:0', grad_fn=<SumBackward0>)

# 250 itr, coefs sum no aux, phis sum aux
# tensor(3.2806, device='cuda:0', grad_fn=<SumBackward0>)
# tensor(0.0504, device='cuda:0', grad_fn=<SumBackward0>)

# 250 itr, coefs sum no aux, phis mean aux, lr=0.01, stop_conds=0.005
# tensor(3.8113, device='cuda:0', grad_fn=<SumBackward0>)
# tensor(1.0491, device='cuda:0', grad_fn=<SumBackward0>)

# 250 itr, coefs sum no aux, phis mean aux, lr=0.001, stop_conds=0.0005
# tensor(2.0708, device='cuda:0', grad_fn=<SumBackward0>)
# tensor(2.0677, device='cuda:0', grad_fn=<SumBackward0>)
