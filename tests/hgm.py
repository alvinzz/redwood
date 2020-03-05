from generative_model import HierarchicalGenerativeModel as HGM
from viz import viz_phis

import torch

device = "cpu"

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
        torch.ones(1, 1, 1, 1, 8, 1, device=device),
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

######### infer coefs + phis, phis starting from reference initialization ###########
inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=1000, lr=0.01, abs_grad_stop_cond=0., rel_grad_stop_cond=0., use_sparse=False)
print("vars0_norm", torch.abs(torch.stack(inferred_vars_list[0])).sum())
print("vars1_norm", torch.abs(torch.stack(inferred_vars_list[1])).sum())
print(inferred_vars_list[1])

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
# inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=1000)
# hgm.update_phis(video, inferred_coefs_list, use_warm_start_optimizer=False)

# #hgm.phis_list = torch.load(open("hgm/rand_init/phis_list_001_0001.torch", "rb"))
# #for itr in range(100, 250):
# for itr in range(250):
#     print("Itr:", itr)

#     inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=1000)
#     hgm.update_phis(video, inferred_coefs_list, use_warm_start_optimizer=True)

#     print("vars0_norm", torch.abs(torch.stack(inferred_vars_list[0])).sum())
#     print("vars1_norm", torch.abs(torch.stack(inferred_vars_list[1])).sum())

# inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=1000)
# torch.save(inferred_vars_list, "hgm/rand_init/vars_list.torch")
# torch.save(inferred_coefs_list, "hgm/rand_init/coefs_list.torch")
# torch.save(hgm.phis_list, "hgm/rand_init/phis_list.torch")
# # TODO: plot_coefs
# viz_phis(hgm.phis_list[0], "hgm/rand_init/phis0")
# # viz_phis(hgm.phis_list[1], "hgm/rand_init/phis1")
# hgm.generate_video(inferred_coefs_list, hgm.phis_list, plot_save_dir="hgm/rand_init/video")

# hgm.phis_list = torch.load(open("hgm/rand_init/phis_list.torch", "rb"))
# for i in range(2):
#  hgm.phis_list[i] = hgm.phis_list[i].to(device)
# inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=1000, lr=0.001, abs_grad_stop_cond=0., rel_grad_stop_cond=0.)
# inferred_coefs_list[0][0] = torch.zeros_like(inferred_coefs_list[0][0])
# inferred_coefs_list[0][1] = torch.zeros_like(inferred_coefs_list[0][1])
# inferred_coefs_list[0][2] = torch.zeros_like(inferred_coefs_list[0][2])
# inferred_coefs_list[0][3] = torch.zeros_like(inferred_coefs_list[0][3])
# inferred_coefs_list[0][4] = torch.ones_like(inferred_coefs_list[0][4])
# inferred_coefs_list[0][5] = torch.ones_like(inferred_coefs_list[0][5])
# inferred_coefs_list[0][6] = torch.ones_like(inferred_coefs_list[0][6])
# inferred_coefs_list[0][7] = torch.zeros_like(inferred_coefs_list[0][7])
# hgm.generate_video(inferred_coefs_list, hgm.phis_list, plot_save_dir="hgm/rand_init/video2")

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

# 250 itr, coefs: sum (no aux), lr=0.001, stop_conds=0.0005, phis: mean (w/ aux), lr=0.001 (prev 0.01)
# tensor(1.9124, device='cuda:0', grad_fn=<SumBackward0>)
# tensor(2.7371, device='cuda:0', grad_fn=<SumBackward0>)

# 250 itr, coefs: sum (no aux), lr=0.001, stop_conds=0.0005, phis: mean (w/ aux (all weighted equally)), lr=0.001
# tensor(3.2752, grad_fn=<SumBackward0>)
# tensor(1.9326, grad_fn=<SumBackward0>)

# 250 itr, coefs: sum (no aux), lr=0.001, stop_conds=0.0005, phis: mean (no aux), lr=0.001
# tensor(2.1612, grad_fn=<SumBackward0>)
# tensor(2.9413, grad_fn=<SumBackward0>)

# 250 itr, coefs: sum (0.125 decay/layer) (no aux), lr=0.001, stop_conds=0.0005, phis: mean (no aux), lr=0.001
# tensor(5.2459, grad_fn=<SumBackward0>)
# tensor(1.9312, grad_fn=<SumBackward0>)

# 250 itr, coefs: sum (no aux), lr=0.001, stop_conds=0.0005, phis: mean (separate, greedy next layer), lr=0.001
# tensor(3.5792, grad_fn=<SumBackward0>)
# tensor(0.0447, grad_fn=<SumBackward0>)

# 1000 itr (250 coefs), coefs: sum (no aux), lr=0.001, stop_conds=0.0005, phis: mean (no aux), lr=0.001
# tensor(2.2435, grad_fn=<SumBackward0>)
# tensor(1.6505, grad_fn=<SumBackward0>)

# 250 itr (250 coefs), coefs: sum (no aux), lr=0.001, stop_conds=0.0005, phis: mean (no aux), lr=0.001
# tensor(1.9577, grad_fn=<SumBackward0>)
# tensor(2.6633, grad_fn=<SumBackward0>)

# 250 itr, coefs: sum (no aux), lr=0.001, stop_conds=0.0005, phis: mean l2 (w/ aux (all weighted equally)), lr=0.001
# tensor(3.2399, grad_fn=<SumBackward0>)
# tensor(1.8741, grad_fn=<SumBackward0>)

# 250 itr, coefs: sum (no aux), lr=0.001, stop_conds=0.0005, phis: mean l2 (w/ staircase aux (all weighted equally)), lr=0.001
# tensor(3.5998, grad_fn=<SumBackward0>)
# tensor(0.9724, grad_fn=<SumBackward0>)

# 25 itr, coefs: sum (no aux), lr=0.001, stop_conds=0.0005, phis: mean l2 (w/ staircase aux (all weighted equally)), lr=0.001
# tensor(3.2924, grad_fn=<SumBackward0>)
# tensor(2.3324, grad_fn=<SumBackward0>)

# BUGFIXED FOR LEARNING PHIS
# 100 itr, coefs: sum (no aux), lr=0.01, stop_conds=0.001, layer_decay_rate=10, phis: coefs loss, lr=0.01
# tensor(2.3349, grad_fn=<SumBackward0>)
# tensor(3.2531, grad_fn=<SumBackward0>)

# 100 itr, coefs: sum (no aux), lr=0.001, stop_conds=0.0001, layer_decay_rate=10, phis: coefs loss, lr=0.01
# tensor(2.7195, grad_fn=<SumBackward0>)
# tensor(2.1255, grad_fn=<SumBackward0>)

# 100 itr, coefs: sum (no aux), lr=0.01, stop_conds=0.001, layer_decay_rate=10, phis: coefs loss, lr=0.001
# tensor(0.9566, grad_fn=<SumBackward0>)
# tensor(7.5519, grad_fn=<SumBackward0>)

# 250 itr, coefs: sum (no aux), lr=0.01, stop_conds=0.001, layer_decay_rate=10, phis: coefs loss, lr=0.01

