from generative_model import generate_video
from viz import viz_phis

import numpy as np

# phis: [[N_y, N_x], [CH], n_philters, t, n_y, n_x, ch]
# a: [batch, T, N_y, N_x, [CH], n_philters]
import torch
PHIS_0 = torch.tensor([
    [0.3, 0., 0.1],
    [0., 0.5, 0.],
    [0.9, 0., 0.7]], dtype=torch.float32).reshape(
    1, 1, 1, 1, 3, 3, 1)
viz_phis(PHIS_0, "xy_rot/phis0")
PHIS_1 = torch.tensor([
    [
        [[[[0.3, 0], [0, 0.1], [0.1, 0]], [[0, 0.7], [0.5, 0.5], [0, 0.3]], [[0.9, 0], [0, 0.9], [0.7, 0]]],
        [[[0.7, 0], [0, 0.9], [0.9, 0]], [[0, 0.3], [0.5, 0.5], [0, 0.7]], [[0.1, 0], [0, 0.1], [0.3, 0]]],
        [[[0, 0.5], [0.5, 0], [0, 0.5]], [[0.5, 0], [0.5, 0.5], [0.5, 0]], [[0, 0.5], [0.5, 0], [0, 0.5]]]]
    ],
    [
        [[[[0, 0.1], [0.1, 0], [0, 0.3]], [[0.3, 0], [0.5, 0.5], [0.7, 0]], [[0, 0.7], [0.9, 0], [0, 0.9]]],
        [[[0, 0.9], [0.9, 0], [0, 0.7]], [[0.7, 0], [0.5, 0.5], [0.3, 0]], [[0, 0.3], [0.1, 0], [0, 0.1]]],
        [[[0.5, 0], [0, 0.5], [0.5, 0]], [[0, 0.5], [0.5, 0.5], [0, 0.5]], [[0.5, 0], [0, 0.5], [0.5, 0]]]]
    ]], dtype=torch.float32).reshape(
    [2, 3, 1, 3, 3, 2])
viz_phis(PHIS_1, "xy_rot/phis1")


def phis0_alpha1_stride1_a1_mux0_muy0_rot0_sigmax1_sigmay1():
    phis = PHIS_0

    alpha_X = alpha_Y = 1
    stride_X = stride_Y = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)
    mu_Y = torch.tensor([0], dtype=torch.float32)
    mu_Y = mu_Y.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)
    sigma_Y = torch.tensor([1], dtype=torch.float32)
    sigma_Y = sigma_Y.reshape(1, 1, 1, 1, 1, 1)

    rot = torch.tensor([0], dtype=torch.float32)
    rot = rot.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = 1
    stride_T = 1
    mu_T = torch.zeros_like(mu_X)
    sigma_T = torch.ones_like(sigma_X)

    plot_save_dir = "xy_rot/phis0_alpha1_stride1_a1_mux0_muy0_rot0_sigmax1_sigmay1"
    RGB = False

    generate_video(
    phis,
    alpha_T, alpha_Y, alpha_X,
    stride_T, stride_Y, stride_X,
    a,
    mu_T, mu_Y, mu_X,
    sigma_T, sigma_Y, sigma_X,
    rot,
    plot_save_dir, RGB,
)

def phis0_alpha8_stride1_a1_mux0_muy0_rot0_sigmax1_sigmay1():
    phis = PHIS_0

    alpha_X = alpha_Y = 8
    stride_X = stride_Y = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)
    mu_Y = torch.tensor([0], dtype=torch.float32)
    mu_Y = mu_Y.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)
    sigma_Y = torch.tensor([1], dtype=torch.float32)
    sigma_Y = sigma_Y.reshape(1, 1, 1, 1, 1, 1)

    rot = torch.tensor([0], dtype=torch.float32)
    rot = rot.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = 1
    stride_T = 1
    mu_T = torch.zeros_like(mu_X)
    sigma_T = torch.ones_like(sigma_X)

    plot_save_dir = "xy_rot/phis0_alpha8_stride1_a1_mux0_muy0_rot0_sigmax1_sigmay1"
    RGB = False

    generate_video(
    phis,
    alpha_T, alpha_Y, alpha_X,
    stride_T, stride_Y, stride_X,
    a,
    mu_T, mu_Y, mu_X,
    sigma_T, sigma_Y, sigma_X,
    rot,
    plot_save_dir, RGB,
)

def phis0_alpha8_stride1_a1_mux0_muy0_rotpi4_sigmax1_sigmay1():
    phis = PHIS_0

    alpha_X = alpha_Y = 8
    stride_X = stride_Y = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)
    mu_Y = torch.tensor([0], dtype=torch.float32)
    mu_Y = mu_Y.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)
    sigma_Y = torch.tensor([1], dtype=torch.float32)
    sigma_Y = sigma_Y.reshape(1, 1, 1, 1, 1, 1)

    rot = torch.tensor([np.pi/4], dtype=torch.float32)
    rot = rot.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = 1
    stride_T = 1
    mu_T = torch.zeros_like(mu_X)
    sigma_T = torch.ones_like(sigma_X)

    plot_save_dir = "xy_rot/phis0_alpha8_stride1_a1_mux0_muy0_rotpi4_sigmax1_sigmay1"
    RGB = False

    generate_video(
    phis,
    alpha_T, alpha_Y, alpha_X,
    stride_T, stride_Y, stride_X,
    a,
    mu_T, mu_Y, mu_X,
    sigma_T, sigma_Y, sigma_X,
    rot,
    plot_save_dir, RGB,
)

def phis0_alpha8_stride1_a05_mux0_muy0_rot0_sigmax1_sigmay05():
    phis = PHIS_0

    alpha_X = alpha_Y = 8
    stride_X = stride_Y = 1

    a = torch.tensor([0.5], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)
    mu_Y = torch.tensor([0], dtype=torch.float32)
    mu_Y = mu_Y.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)
    sigma_Y = torch.tensor([0.5], dtype=torch.float32)
    sigma_Y = sigma_Y.reshape(1, 1, 1, 1, 1, 1)

    rot = torch.tensor([0], dtype=torch.float32)
    rot = rot.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = 1
    stride_T = 1
    mu_T = torch.zeros_like(mu_X)
    sigma_T = torch.ones_like(sigma_X)

    plot_save_dir = "xy_rot/phis0_alpha8_stride1_a05_mux0_muy0_rot0_sigmax1_sigmay05"
    RGB = False

    generate_video(
    phis,
    alpha_T, alpha_Y, alpha_X,
    stride_T, stride_Y, stride_X,
    a,
    mu_T, mu_Y, mu_X,
    sigma_T, sigma_Y, sigma_X,
    rot,
    plot_save_dir, RGB,
)

def phis0_alpha8_stride1_a05_mux0_muy0_rotpi4_sigmax1_sigmay05():
    phis = PHIS_0

    alpha_X = alpha_Y = 8
    stride_X = stride_Y = 1

    a = torch.tensor([0.5], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)
    mu_Y = torch.tensor([0], dtype=torch.float32)
    mu_Y = mu_Y.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)
    sigma_Y = torch.tensor([0.5], dtype=torch.float32)
    sigma_Y = sigma_Y.reshape(1, 1, 1, 1, 1, 1)

    rot = torch.tensor([np.pi/4], dtype=torch.float32)
    rot = rot.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = 1
    stride_T = 1
    mu_T = torch.zeros_like(mu_X)
    sigma_T = torch.ones_like(sigma_X)

    plot_save_dir = "xy_rot/phis0_alpha8_stride1_a05_mux0_muy0_rotpi4_sigmax1_sigmay05"
    RGB = False

    generate_video(
    phis,
    alpha_T, alpha_Y, alpha_X,
    stride_T, stride_Y, stride_X,
    a,
    mu_T, mu_Y, mu_X,
    sigma_T, sigma_Y, sigma_X,
    rot,
    plot_save_dir, RGB,
)

def phis0_alpha8_stride1_a05_mux_0125_muy025_rotpi4_sigmax1_sigmay05():
    phis = PHIS_0

    alpha_X = alpha_Y = 8
    stride_X = stride_Y = 1

    a = torch.tensor([0.5], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([-0.125], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)
    mu_Y = torch.tensor([0.25], dtype=torch.float32)
    mu_Y = mu_Y.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)
    sigma_Y = torch.tensor([0.5], dtype=torch.float32)
    sigma_Y = sigma_Y.reshape(1, 1, 1, 1, 1, 1)

    rot = torch.tensor([np.pi/4], dtype=torch.float32)
    rot = rot.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = 1
    stride_T = 1
    mu_T = torch.zeros_like(mu_X)
    sigma_T = torch.ones_like(sigma_X)

    plot_save_dir = "xy_rot/phis0_alpha8_stride1_a05_mux_0125_muy025_rotpi4_sigmax1_sigmay05"
    RGB = False

    generate_video(
    phis,
    alpha_T, alpha_Y, alpha_X,
    stride_T, stride_Y, stride_X,
    a,
    mu_T, mu_Y, mu_X,
    sigma_T, sigma_Y, sigma_X,
    rot,
    plot_save_dir, RGB,
)

def phis0_alpha8_stride1_a05_mux025_muy_0125_rot_pi6_sigmax05_sigmay1():
    phis = PHIS_0

    alpha_X = alpha_Y = 8
    stride_X = stride_Y = 1

    a = torch.tensor([0.5], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0.25], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)
    mu_Y = torch.tensor([-0.125], dtype=torch.float32)
    mu_Y = mu_Y.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([0.5], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)
    sigma_Y = torch.tensor([1], dtype=torch.float32)
    sigma_Y = sigma_Y.reshape(1, 1, 1, 1, 1, 1)

    rot = torch.tensor([-np.pi/6], dtype=torch.float32)
    rot = rot.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = 1
    stride_T = 1
    mu_T = torch.zeros_like(mu_X)
    sigma_T = torch.ones_like(sigma_X)

    plot_save_dir = "xy_rot/phis0_alpha8_stride1_a05_mux025_muy_0125_rot_pi6_sigmax05_sigmay1"
    RGB = False

    generate_video(
    phis,
    alpha_T, alpha_Y, alpha_X,
    stride_T, stride_Y, stride_X,
    a,
    mu_T, mu_Y, mu_X,
    sigma_T, sigma_Y, sigma_X,
    rot,
    plot_save_dir, RGB,
)

def phis0_tiling():
    phis = PHIS_0

    alpha_X = alpha_Y = 8
    stride_X = stride_Y = 1

    a = torch.tensor([
        [0.5, 1/np.sqrt(2), 0.5],
        [1/np.sqrt(2), 1, 1/np.sqrt(2)],
        [0.5, 1/np.sqrt(2), 0.5]], dtype=torch.float32)
    a = a.reshape(1, 1, 3, 3, 1, 1)

    mu_X = torch.tensor([
        [-0.125, 0, 0.125],
        [-0.125, 0, 0.125],
        [-0.125, 0, 0.125]], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 3, 3, 1, 1)
    mu_Y = torch.tensor([
        [-0.125, -0.125, -0.125],
        [0, 0, 0],
        [0.125, 0.125, 0.125]], dtype=torch.float32)
    mu_Y = mu_Y.reshape(1, 1, 3, 3, 1, 1)

    sigma_X = torch.tensor([
        [0.5, 1, 1],
        [1/np.sqrt(2), 1, 1/np.sqrt(2)],
        [1, 1, 0.5]], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 3, 3, 1, 1)
    sigma_Y = torch.tensor([
        [1, 1/np.sqrt(2), 0.5],
        [1, 1, 1],
        [0.5, 1/np.sqrt(2), 1]], dtype=torch.float32)
    sigma_Y = sigma_Y.reshape(1, 1, 3, 3, 1, 1)

    rot = torch.tensor([
        [np.pi/4, 0, np.pi/4],
        [0, 0, 0],
        [np.pi/4, 0, np.pi/4]], dtype=torch.float32)
    rot = rot.reshape(1, 1, 3, 3, 1, 1)

    alpha_T = 1
    stride_T = 1
    mu_T = torch.zeros_like(mu_X)
    sigma_T = torch.ones_like(sigma_X)

    plot_save_dir = "xy_rot/phis0_tiling"
    RGB = False

    generate_video(
    phis,
    alpha_T, alpha_Y, alpha_X,
    stride_T, stride_Y, stride_X,
    a,
    mu_T, mu_Y, mu_X,
    sigma_T, sigma_Y, sigma_X,
    rot,
    plot_save_dir, RGB,
)

# phis: [[N_y, N_x], [CH], n_philters, t, n_y, n_x, ch]
# a: [batch, T, N_y, N_x, [CH], n_philters]
def phis1_batch_CH_Ny_Nx_nphilters():
    phis = PHIS_1

    alpha_X = alpha_Y = 8
    stride_X = stride_Y = 2

    a = torch.tensor([
        [
            [[[[1, 0, 0], [1, 0, 0]]], [[[0, 1, 0], [0, 1, 0]]], [[[0, 0, 1], [0, 0, 1]]]],
            [[[[1, 1, 0], [1, 1, 0]]], [[[0, 1, 1], [0, 1, 1]]], [[[1, 0, 1], [1, 0, 1]]]],
            [[[[1, 1, 2], [1, 1, 2]]], [[[1, 0, 0], [0, 1, 0]]], [[[0, 1, 0], [1, 0, 0]]]],
        ],
        [
            [[[[1, 1, 2], [1, 1, 2]]], [[[1, 0, 0], [0, 1, 0]]], [[[0, 1, 0], [1, 0, 0]]]],
            [[[[1, 0, 0], [1, 0, 0]]], [[[0, 1, 0], [0, 1, 0]]], [[[0, 0, 1], [0, 0, 1]]]],
            [[[[1, 1, 0], [1, 1, 0]]], [[[0, 1, 1], [0, 1, 1]]], [[[1, 0, 1], [1, 0, 1]]]],
        ],
    ], dtype=torch.float32)
    a = a.reshape(2, 1, 3, 3, 2, 3)

    mu_X = torch.tensor([
        [
            [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]],
            [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0.25, -0.25], [0, -0.25, 0.25]]], [[[-0.25, 0, 0.25], [0.25, 0, -0.25]]]],
            [[[[0, 0, 0], [0, 0, 0]]], [[[0.25, 0, 0], [0, -0.25, 0]]], [[[0, -0.25, 0], [0.25, 0, 0]]]],
        ],
        [
            [[[[0, 0, 0], [0, 0, 0]]], [[[0.25, 0, 0], [0, -0.25, 0]]], [[[0, -0.25, 0], [0.25, 0, 0]]]],
            [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]],
            [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0.25, -0.25], [0, -0.25, 0.25]]], [[[-0.25, 0, 0.25], [0.25, 0, -0.25]]]],
        ],
    ], dtype=torch.float32)
    mu_X = mu_X.reshape(2, 1, 3, 3, 2, 3)
    mu_Y = -mu_X.clone()

    sigma_X = torch.tensor([
        [
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]]],
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]]],
            [[[[1, 1, 1], [1, 1, 1]]], [[[0.5, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]]],
        ],
        [
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [0.5, 1, 1]]]],
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]]],
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]]],
        ],
    ], dtype=torch.float32)
    sigma_X = sigma_X.reshape(2, 1, 3, 3, 2, 3)
    sigma_Y = torch.tensor([
        [
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]]],
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]]],
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 0.5, 1]]], [[[1, 1, 1], [1, 1, 1]]]],
        ],
        [
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]], [[[1, 0.5, 1], [1, 1, 1]]]],
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]]],
            [[[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]], [[[1, 1, 1], [1, 1, 1]]]],
        ],
    ], dtype=torch.float32)
    sigma_Y = sigma_Y.reshape(2, 1, 3, 3, 2, 3)

    rot = torch.tensor([
        [
            [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]],
            [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]],
            [[[[0, 0, 0], [0, 0, 0]]], [[[np.pi/6, 0, 0], [0, -np.pi/6, 0]]], [[[0, -np.pi/4, 0], [np.pi/4, 0, 0]]]],
        ],
        [
            [[[[0, 0, 0], [0, 0, 0]]], [[[-np.pi/4, 0, 0], [0, np.pi/4, 0]]], [[[0, np.pi/6, 0], [-np.pi/6, 0, 0]]]],
            [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]],
            [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]],
        ],
    ], dtype=torch.float32)
    rot = rot.reshape(2, 1, 3, 3, 2, 3)

    alpha_T = 1
    stride_T = 1
    mu_T = torch.zeros_like(mu_X)
    sigma_T = torch.ones_like(sigma_X)

    plot_save_dir = "xy_rot/phis1_batch_CH_Ny_Nx_nphilters"
    RGB = False

    generate_video(
    phis,
    alpha_T, alpha_Y, alpha_X,
    stride_T, stride_Y, stride_X,
    a,
    mu_T, mu_Y, mu_X,
    sigma_T, sigma_Y, sigma_X,
    rot,
    plot_save_dir, RGB,
)

if __name__ == "__main__":
    # phis0_alpha1_stride1_a1_mux0_muy0_rot0_sigmax1_sigmay1()
    # phis0_alpha8_stride1_a1_mux0_muy0_rot0_sigmax1_sigmay1()
    # phis0_alpha8_stride1_a1_mux0_muy0_rotpi4_sigmax1_sigmay1()
    # phis0_alpha8_stride1_a05_mux0_muy0_rot0_sigmax1_sigmay05()
    # phis0_alpha8_stride1_a05_mux0_muy0_rotpi4_sigmax1_sigmay05()
    # phis0_alpha8_stride1_a05_mux_0125_muy025_rotpi4_sigmax1_sigmay05()
    phis0_alpha8_stride1_a05_mux025_muy_0125_rot_pi6_sigmax05_sigmay1()
    # phis0_tiling()

    # phis1_batch_CH_Ny_Nx_nphilters()