from torchvision import datasets, transforms

from generative_model import HierarchicalGenerativeModel as HGM
from viz import viz_phis

import torch
import numpy as np

device = "cuda"

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=1, shuffle=False)

# phis: [[N_y, N_x], [CH], n_philters, t, n_y, n_x, ch]
# a: [batch, T, N_y, N_x, [CH], n_philters]
# hgm = HGM(
#     [
#         torch.normal(torch.zeros(50, 1, 10, 10, 1), torch.ones(50, 1, 10, 10, 1)),
#         torch.normal(torch.zeros(50, 1, 10, 10, 400), torch.ones(50, 1, 10, 10, 400)),
#         torch.normal(torch.zeros(50, 1, 10, 10, 400), torch.ones(50, 1, 10, 10, 400)),
#     ],
#     [1, 1, 1], [1, 1, 1], [1, 1, 1],
#     [1, 1, 1], [0.1, 0.1, 1], [0.1, 0.1, 1],
#     device=device,
# )

import tqdm

hgm = HGM(
    [
        torch.normal(torch.zeros(50, 1, 10, 10, 1), torch.ones(50, 1, 10, 10, 1)),
        torch.normal(torch.zeros(50, 1, 10, 10, 400), torch.ones(50, 1, 10, 10, 400)),
    ],
    [1, 1], [1, 1], [1, 1],
    [1, 1], [2/10, 1], [2/10, 1],
    device=device,
)

for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader)):

    data, target = data.to(device), target.to(device)
    video = data.permute(0, 2, 3, 1).unsqueeze(1) # [batch, T, N_y, N_x, CH]

    print("batch_idx:", batch_idx)
    inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=250, use_sparse=True)
    hgm.update_phis(video, inferred_coefs_list, use_warm_start_optimizer=(batch_idx > 1), use_sparse=True)

    torch.save(hgm.phis_list, "mnist/phis_list_{}.torch".format(batch_idx))
    torch.save(inferred_coefs_list, "mnist/inferred_coefs_list_{}.torch".format(batch_idx))

    continue

    if batch_idx < 3:
        pass

    if batch_idx == 3:
        hgm.phis_list = torch.load(open("mnist/phis_list_3.torch", "rb"))

    if batch_idx > 3:
        data, target = data.to(device), target.to(device)
        video = data.permute(0, 2, 3, 1).unsqueeze(1) # [batch, T, N_y, N_x, CH]

        print("batch_idx:", batch_idx)
        inferred_vars_list, inferred_coefs_list = hgm.infer_coefs(video, hgm.phis_list, max_itr=250, use_sparse=True)
        hgm.update_phis(video, inferred_coefs_list, use_warm_start_optimizer=(batch_idx > 3+1), use_sparse=True)

        torch.save(hgm.phis_list, "mnist/phis_list_{}.torch".format(batch_idx))
        torch.save(inferred_coefs_list, "mnist/inferred_coefs_list_{}.torch".format(batch_idx))
