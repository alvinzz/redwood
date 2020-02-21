from generative_model import generate_video
from viz import viz_phis

# generate_video(
#     phis,
#     alpha_T, alpha_Y, alpha_X,
#     stride_T, stride_Y, stride_X,
#     a,
#     mu_T, mu_Y, mu_X,
#     sigma_T, sigma_Y, sigma_X,
#     rot,
#     plot_save_dir=None, RGB=False,
# )

# phis: [[N_y, N_x], [CH], n_philters, t, n_y, n_x, ch]
# a: [batch, T, N_y, N_x, [CH], n_philters]
import torch
PHIS_0 = torch.tensor([1], dtype=torch.float32).reshape(
    1, 1, 1, 1, 1)
viz_phis(PHIS_0, "x/phis0")
PHIS_1 = torch.tensor([0.9, 0.6, 0.3], dtype=torch.float32).reshape(
    1, 1, 1, 3, 1)
viz_phis(PHIS_1, "x/phis1")
PHIS_2 = torch.tensor([
    [0.7, 0.8, 0.9],
    [0.4, 0.5, 0.6],
    [0.1, 0.2, 0.3]], dtype=torch.float32).reshape(
    1, 3, 1, 1, 1, 3, 1)
viz_phis(PHIS_2, "x/phis2")
PHIS_3 = torch.tensor([
    [
        [[1, 0.7], [0, 1], [0.7, 0]],
        [[0.7, 0], [1, 0.7], [0, 1]],
        [[0, 1], [0.7, 0], [1, 0.7]]
    ],
    [
        [[1, 0.3], [0, 1], [0.3, 0]],
        [[0.3, 0], [1, 0.3], [0, 1]],
        [[0, 1], [0.3, 0], [1, 0.3]]
    ]], dtype=torch.float32).reshape(
    [2, 3, 1, 1, 3, 2])
viz_phis(PHIS_3, "x/phis3")

def phis0_alpha1_stride1_a1_mu0_sigma1():
    phis = PHIS_0

    alpha_X = 1
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis0_alpha1_stride1_a1_mu0_sigma1"
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

def phis0_alpha8_stride1_a1_mu0_sigma1():
    phis = PHIS_0

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis0_alpha8_stride1_a1_mu0_sigma1"
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

def phis0_alpha8_stride1_a2_mu0_sigma1():
    phis = PHIS_0

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([2], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis0_alpha8_stride1_a2_mu0_sigma1"
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

def phis0_alpha8_stride1_a1_mu025_sigma1():
    phis = PHIS_0

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0.25], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis0_alpha8_stride1_a1_mu025_sigma1"
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

def phis0_alpha8_stride1_a1_mu0_sigma075():
    phis = PHIS_0

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([0.75], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis0_alpha8_stride1_a1_mu0_sigma075"
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

def phis0_alpha8_stride1_a1_mu0_sigma133():
    phis = PHIS_0

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1.33], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis0_alpha8_stride1_a1_mu0_sigma133"
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

def phis0_alpha8_stride1_a1_mu_05_sigma133():
    phis = PHIS_0

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([-0.5], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1.33], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis0_alpha8_stride1_a1_mu_05_sigma133"
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

def phis0_alpha8_stride1_a1_mu_05_sigma075():
    phis = PHIS_0

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([-0.5], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([0.75], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis0_alpha8_stride1_a1_mu_05_sigma075"
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

def phis1_alpha1_stride1_a1_mu0_sigma1():
    phis = PHIS_1

    alpha_X = 1
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis1_alpha1_stride1_a1_mu0_sigma1"
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

def phis1_alpha8_stride1_a1_mu0_sigma1():
    phis = PHIS_1

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis1_alpha8_stride1_a1_mu0_sigma1"
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

def phis1_alpha8_stride1_a1_mu025_sigma1():
    phis = PHIS_1

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0.25], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis1_alpha8_stride1_a1_mu025_sigma1"
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

def phis1_alpha8_stride1_a1_mu0125_sigma1():
    phis = PHIS_1

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0.125], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis1_alpha8_stride1_a1_mu0125_sigma1"
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

def phis1_alpha8_stride1_a1_mu0_sigma075():
    phis = PHIS_1

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([0.75], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis1_alpha8_stride1_a1_mu0_sigma075"
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

def phis1_alpha8_stride1_a1_mu0_sigma125():
    phis = PHIS_1

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([1.25], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis1_alpha8_stride1_a1_mu0_sigma125"
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

def phis1_alpha8_stride1_a1_mu0125_sigma075():
    phis = PHIS_1

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 1, 1)

    mu_X = torch.tensor([0.125], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 1, 1)

    sigma_X = torch.tensor([0.75], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 1, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis1_alpha8_stride1_a1_mu0125_sigma075"
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

def phis1_alpha8_stride1_a20_mu0505_sigma22():
    phis = PHIS_1

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([2, 0], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 2, 1)

    mu_X = torch.tensor([0.5, 0.5], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 2, 1)

    sigma_X = torch.tensor([2, 2], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 2, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis1_alpha8_stride1_a20_mu0505_sigma22"
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

def phis2_alpha1_stride1_a111_mu000_sigma111():
    phis = PHIS_2

    alpha_X = 1
    stride_X = 1

    a = torch.tensor([1, 1, 1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 3, 1)

    mu_X = torch.tensor([0, 0, 0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 3, 1)

    sigma_X = torch.tensor([1, 1, 1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 3, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis2_alpha1_stride1_a111_mu000_sigma111"
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

def phis2_alpha8_stride1_a111_mu000_sigma111():
    phis = PHIS_2

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1, 1, 1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 3, 1)

    mu_X = torch.tensor([0, 0, 0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 3, 1)

    sigma_X = torch.tensor([1, 1, 1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 3, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis2_alpha8_stride1_a111_mu000_sigma111"
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

def phis2_alpha8_stride1_a111_mu_025000025_sigma111():
    phis = PHIS_2

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1, 1, 1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 3, 1)

    mu_X = torch.tensor([-0.25, 0, 0.25], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 3, 1)

    sigma_X = torch.tensor([1, 1, 1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 3, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis2_alpha8_stride1_a111_mu_025000025_sigma111"
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

def phis2_alpha8_stride2_a111_mu_000_sigma111():
    phis = PHIS_2

    alpha_X = 8
    stride_X = 2

    a = torch.tensor([1, 1, 1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 3, 1)

    mu_X = torch.tensor([0, 0, 0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 3, 1)

    sigma_X = torch.tensor([1, 1, 1], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 3, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis2_alpha8_stride2_a111_mu_000_sigma111"
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

def phis2_alpha8_stride2_a222_mu_000_sigma222():
    phis = PHIS_2

    alpha_X = 8
    stride_X = 2

    a = torch.tensor([2, 2, 2], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 3, 1)

    mu_X = torch.tensor([0, 0, 0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 3, 1)

    sigma_X = torch.tensor([2, 2, 2], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 3, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis2_alpha8_stride2_a222_mu_000_sigma222"
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

def phis2_alpha8_stride1_a121212_mu_020002_sigma121212():
    phis = PHIS_2

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1.2, 1.2, 1.2], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 3, 1)

    mu_X = torch.tensor([-0.2, 0, 0.2], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 3, 1)

    sigma_X = torch.tensor([1.2, 1.2, 1.2], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 3, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis2_alpha8_stride1_a121212_mu_020002_sigma121212"
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

def phis2_alpha8_stride1_a111_mu000_sigma201005():
    phis = PHIS_2

    alpha_X = 8
    stride_X = 1

    a = torch.tensor([1, 1, 1], dtype=torch.float32)
    a = a.reshape(1, 1, 1, 1, 3, 1)

    mu_X = torch.tensor([0, 0, 0], dtype=torch.float32)
    mu_X = mu_X.reshape(1, 1, 1, 1, 3, 1)

    sigma_X = torch.tensor([2, 1, 0.5], dtype=torch.float32)
    sigma_X = sigma_X.reshape(1, 1, 1, 1, 3, 1)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis2_alpha8_stride1_a111_mu000_sigma201005"
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
def phis3_batch_CH_Nx_nphilters():
    phis = PHIS_3

    alpha_X = 1
    stride_X = 1

    a = torch.tensor([
        [
            [[1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 0]],
        ],
        [
            [[0, 0, 1], [0.5, 0.5, 0.5]],
            [[1, 0, 0], [0.5, 0.5, 0.5]],
            [[0, 1, 0], [0.5, 0.5, 0.5]],
        ]], dtype=torch.float32)
    a = a.reshape(2, 1, 1, 3, 2, 3)

    mu_X = torch.zeros_like(a)

    sigma_X = torch.ones_like(a)

    alpha_T = alpha_Y = 1
    stride_T = stride_Y = 1
    mu_T = mu_Y = rot = torch.zeros_like(mu_X)
    sigma_T = sigma_Y = torch.ones_like(sigma_X)

    plot_save_dir = "x/phis3_batch_CH_Nx_nphilters"
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
    # phis0_alpha1_stride1_a1_mu0_sigma1()
    # phis0_alpha8_stride1_a1_mu0_sigma1()
    # phis0_alpha8_stride1_a2_mu0_sigma1()
    # phis0_alpha8_stride1_a1_mu025_sigma1()
    # phis0_alpha8_stride1_a1_mu0_sigma075()
    # phis0_alpha8_stride1_a1_mu0_sigma133()
    # phis0_alpha8_stride1_a1_mu_05_sigma133()
    # phis0_alpha8_stride1_a1_mu_05_sigma075()

    # phis1_alpha1_stride1_a1_mu0_sigma1()
    # phis1_alpha8_stride1_a1_mu0_sigma1()
    # phis1_alpha8_stride1_a1_mu025_sigma1()
    # phis1_alpha8_stride1_a1_mu0125_sigma1()
    # phis1_alpha8_stride1_a1_mu0_sigma075()
    # phis1_alpha8_stride1_a1_mu0_sigma125()
    # phis1_alpha8_stride1_a1_mu0125_sigma075()
    # phis1_alpha8_stride1_a20_mu0505_sigma22()

    # phis2_alpha1_stride1_a111_mu000_sigma111()
    # phis2_alpha8_stride1_a111_mu000_sigma111()
    # phis2_alpha8_stride1_a111_mu_025000025_sigma111()
    # phis2_alpha8_stride2_a111_mu_000_sigma111()
    # phis2_alpha8_stride2_a222_mu_000_sigma222()
    # phis2_alpha8_stride1_a121212_mu_020002_sigma121212()
    # phis2_alpha8_stride1_a111_mu000_sigma201005()

    # phis3_batch_CH_Nx_nphilters()
    pass