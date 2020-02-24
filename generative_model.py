import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2

def generate_video(
    phis,
    alpha_T, alpha_Y, alpha_X,
    stride_T, stride_Y, stride_X,
    a,
    mu_T, mu_Y, mu_X,
    sigma_T, sigma_Y, sigma_X,
    rot,
    plot_save_dir=None, RGB=False,
    use_sparse=True,
):
    """
    The model:
        A video is generated by the weighted sum of (n_philters) basis functions (also known as "kernels"
            and here referred to as "philters").
            The philters are 4-D and span the time, spatial Y, spatial X, and channel dimensions.
            In this implementation, each philter is restricted to explaining a small chunk of the video
                which is localized in the [time(t), spatial Y(n_y), spatial X(n_x), channel(ch)] dimensions.
                The full video is reconstructed by tiling a weighted sum of these philters across the
                time, spatial, and channel dimensions. The spacing of the tiling is controlled by "stride"
                terms, while the upsampling factor from the philters to the video is controlled by "alpha" 
                terms. If there are [T, N_y, N_x, CH] philters in the tiling, then the output video will
                have dimensions of approximately [T*t*alpha_T*stride_T, N_y*y*alpha_Y*stride_Y, 
                    N_x*x*alpha_X*stride_X, CH*ch].
            There may be a different philter sets for each spatial location and/or for each channel, or
                the basis functions (philter sets) can be shared across the spatial and/or channel dimensions.
                If the philter sets differ across spatial locations, there must be [N_y, N_x] of them.
                Similarly, if the philter sets differ across channels, there must be [CH] of them.
        Additionally, the philters can be scaled or shifted locally to accomodate for local deformations 
            to the philters/basis functions (this can be interpreted as performing dynamic routing).
            Deformations (shifting, scaling, and rotation) and upsampling are performed for the spatial 
                dimensions before the time dimension.
            This implementation does not support deformations along the channel (color) dimension.
                (Reasoning: Typically, the reflectance spectrum (color) of an object does
                not change over time or space. The illumination (shading) may change, but 
                this can already be modelled by changes in the philter coefficients rather 
                than deformations in the philters themselves.)
        A sparse prior is used for the philter coefficients and for the deformation coefficients.
        The model can be extended to incorporate hierarchical structure over the philter coefficients
            and the deformation coefficients. This may result in a more efficient, distributed coding
            for the video. Additionally, spatial and temporal structure may be captured at multiple
            scales, and certain deformations of an object's perspective projection can be modelled 
            as being more likely than others (e.g., a rotation vs. an inversion).

    #######################################################################

    INPUTS:
        PHILTERS (filter bank)
            phis: [[N_y, N_x], [CH], n_philters, t, n_y, n_x, ch]
                The last 4 dimensions (t, n_y, n_x, ch) define a philter.
                Indexing into phis to select a philter occurs as follows:
                TIME DIMENSION (no indexing):
                    Philters are shared across the time dimension (they do not change per-timestep,
                        so as to handle videos of arbitrary length and to reflect constant, continuous,
                        real-time processing), but they can be shifted & scaled along the time dimension.
                SPATIAL DIMENSION (N_y, N_x):
                    Philters can vary across spatial locations, but they may also be shared.
                    If philters are shared across space, this is referred to as the "convolutional" case.
                    Convolutional philters are specified by N_y = N_x = 1, 
                        or by omitting those dimensions from phis.
                CHANNEL DIMENSION (CH):
                    Philters may vary across the channel dimension, but they may also be shared.
                    If philters vary across the channel dimension, 
                        this is known as "depthwise", "separable", or "grouped" convolution.
                    This is specified by including the CH dimension in phis, or CH =/= 1.
        ALPHAS (upsampling factors)
            alpha_T:
                Upsampling in the time dimension.
                Applied AFTER spatial dimension transformations.
            alpha_Y:
                Upsampling in the spatial Y dimension.
            alpha_X:
                Upsampling in the spatial X dimension.
            NOTES:
                Fractional upsampling is allowed.
        STRIDES (philter spacing)
            stride_T:
                Philter spacing in the time dimension, in units of t*alpha_T.
            stride_Y:
                Philter spacing in the spatial Y dimension, in units of n_y*alpha_Y.
            stride_X:
                Philter spacing in the spatial X dimension, in units of n_x*alpha_X.
            NOTES:
                Fractional strides are allowed.
                A stride < 1 indicates overlap between the philters. 
                Strides > 1 are not recommended, since there will be empty space between the philters.
        A (philter coefficients)
            a: [batch, T, N_y, N_x, [CH], n_philters]
                Philter coefficients.
        MUS (philter shifts)
            mu_T: [batch, T, N_y, N_x, [CH], n_philters]
                Philter shifts in the time dimension, in units of t*alpha_T.
                Applied AFTER spatial dimension transformations.
            mu_Y: [batch, T, N_y, N_x, [CH], n_philters]
                Philter shifts in the spatial Y dimension, in units of n_y*alpha_Y.
            mu_X: [batch, T, N_y, N_x, [CH], n_philters]
                Philter shifts in the spatial X dimension, in units of n_x*alpha_X.
        SIGMAS (philter scaling)
            sigma_T: [batch, T, N_y, N_x, [CH], n_philters]
                Philter scaling in the time dimension, relative to alpha_T.
                Applied AFTER spatial dimension transformations.
            sigma_Y: [batch, T, N_y, N_x, [CH], n_philters]
                Philter scaling in the spatial Y dimension, relative to alpha_Y.
            sigma_X: [batch, T, N_y, N_x, [CH], n_philters]
                Philter scaling in the spatial X dimension, relative to alpha_X.
        ROT (philter rotations)
            rot: [batch, T, N_y, N_x, [CH], n_philters]
                Philter rotation in the spatial XY plane.
            NOTES:
                The rotation should be bounded by +/-(pi/4).
        PLOT (plotting settings)
            plot_save_dir:
                Directory to save the plots to.
            RGB:
                Whether or not to combine the first 3 channels into an RGB image instead of
                    plotting each channel separately.
    
    OUTPUTS:
        A video of dimension [batch, ~T*t*alpha_T*stride_T, ~N_y*y*alpha_Y*stride_Y, 
            ~N_x*x*alpha_X*stride_X, CH*ch]. (Approximate values due to rounding & stride.)
        Optionally, saves plots for the generated videos to plot_save_dir.
    """

    ### DIMENSION MUNGING ###
    if len(a.shape) == 5:
        a = a.unsqueeze(4)
        mu_T = mu_T.unsqueeze(4)
        mu_Y = mu_Y.unsqueeze(4)
        mu_X = mu_X.unsqueeze(4)
        sigma_T = sigma_T.unsqueeze(4)
        sigma_Y = sigma_Y.unsqueeze(4)
        sigma_X = sigma_X.unsqueeze(4)
        rot = rot.unsqueeze(4)
    batch_size, T, N_y, N_x, CH, _ = a.shape

    if len(phis.shape) == 5:
        phis = phis.unsqueeze(0).unsqueeze(1).unsqueeze(2)
    if len(phis.shape) == 6:
        phis = phis.unsqueeze(0).unsqueeze(1)
    if len(phis.shape) == 7:
        phis = phis.unsqueeze(2)
    if phis.shape[0] == 1 and N_y > 1:
        phis = phis.repeat(N_y, 1, 1, 1, 1, 1, 1, 1)
    if phis.shape[1] == 1 and N_x > 1:
        phis = phis.repeat(1, N_x, 1, 1, 1, 1, 1, 1)
    if phis.shape[2] == 1 and CH > 1:
        phis = phis.repeat(1, 1, CH, 1, 1, 1, 1, 1)
    _, _, _, n_philters, t, n_y, n_x, ch = phis.shape

    assert a.shape[5] == n_philters, \
        "a suggests n_philters is {} but phis suggests {}".format(a.shape[5], n_philters)

    ### GENERATE VIDEO ###
    T_max = int(np.ceil(t*alpha_T * (stride_T*(T-1) + 1)))
    Y_max = int(np.ceil(n_y*alpha_Y * (stride_Y*(N_y-1) + 1)))
    X_max = int(np.ceil(n_x*alpha_X * (stride_X*(N_x-1) + 1)))
    CH_max = ch*CH
    V = torch.zeros([batch_size, T_max, Y_max, X_max, CH_max], dtype=torch.float32, device=phis.device)

    # centers_T: [T]
    centers_T = t*alpha_T * \
        (stride_T*torch.arange(0, T, dtype=torch.float32, device=phis.device) + 0.5)
    # centers_N_y: [N_y]
    centers_N_y = n_y*alpha_Y * \
        (stride_Y*torch.arange(0, N_y, dtype=torch.float32, device=phis.device) + 0.5)
    # centers_N_x: [N_x]
    centers_N_x = n_x*alpha_X * \
        (stride_X*torch.arange(0, N_x, dtype=torch.float32, device=phis.device) + 0.5)

    # gauss_mus_T: [batch, T, N_y, N_x, [CH], n_philters]
    gauss_mus_T = centers_T.view(1, T, 1, 1, 1, 1) + \
        t*alpha_T*mu_T
    # gauss_mus_Y: [batch, T, N_y, N_x, [CH], n_philters]
    gauss_mus_Y = centers_N_y.view(1, 1, N_y, 1, 1, 1) + \
        n_y*alpha_Y*mu_Y
    # gauss_mus_X: [batch, T, N_y, N_x, [CH], n_philters]
    gauss_mus_X = centers_N_x.view(1, 1, 1, N_x, 1, 1) + \
        n_x*alpha_X*mu_X

    # offset_t: [batch, T, N_y, N_x, CH, n_philters, t]
    offset_t = alpha_T*(sigma_T.unsqueeze(6) * \
        (torch.arange(0, t, dtype=torch.float32, device=phis.device).view(
            1, 1, 1, 1, 1, 1, t)+0.5 - t/2))
    # offset_y: [batch, T, N_y, N_x, CH, n_philters, n_y]
    offset_y = alpha_Y*(sigma_Y.unsqueeze(6) * \
        (torch.arange(0, n_y, dtype=torch.float32, device=phis.device).view(
            1, 1, 1, 1, 1, 1, n_y)+0.5 - n_y/2))
    # offset_x: [batch, T, N_y, N_x, CH, n_philters, n_x]
    offset_x = alpha_X*(sigma_X.unsqueeze(6) * \
        (torch.arange(0, n_x, dtype=torch.float32, device=phis.device).view(
            1, 1, 1, 1, 1, 1, n_x)+0.5 - n_x/2))

    # gauss_mus_t: [batch, T, N_y, N_x, CH, n_philters, t]
    gauss_mus_t = gauss_mus_T.unsqueeze(6) + offset_t
    # gauss_sigmas_t: [batch, T, N_y, N_x, CH, n_philters]
    gauss_sigmas_t = alpha_T * sigma_T / 2

    # gauss_mus_y: [batch, T, N_y, N_x, CH, n_philters, n_y, n_x]
    gauss_mus_y = gauss_mus_Y.unsqueeze(6).unsqueeze(7) + \
        torch.sin(rot).unsqueeze(6).unsqueeze(7)*offset_x.unsqueeze(6) + \
        torch.cos(rot).unsqueeze(6).unsqueeze(7)*offset_y.unsqueeze(7)
    # gauss_sigmas_y: [batch, T, N_y, N_x, CH, n_philters]
    gauss_sigmas_y = (alpha_Y * sigma_Y / 2)

    # gauss_mus_x: [batch, T, N_y, N_x, CH, n_philters, n_y, n_x]
    gauss_mus_x = gauss_mus_X.unsqueeze(6).unsqueeze(7) + \
        torch.cos(rot).unsqueeze(6).unsqueeze(7)*offset_x.unsqueeze(6) - \
        torch.sin(rot).unsqueeze(6).unsqueeze(7)*offset_y.unsqueeze(7)
    # gauss_sigmas_x: [batch, T, N_y, N_x, CH, n_philters]
    gauss_sigmas_x = (alpha_X * sigma_X / 2)

    # gauss_rot: [batch, T, N_y, N_x, CH, n_philters]
    gauss_rots = rot

    if use_sparse:
        # get dense_inds
        b_inds, T_inds, N_y_inds, N_x_inds, CH_inds, phis_inds, t_inds, n_y_inds, n_x_inds = torch.meshgrid(
            torch.arange(0, batch_size),
            torch.arange(0, T),
            torch.arange(0, N_y),
            torch.arange(0, N_x),
            torch.arange(0, CH),
            torch.arange(0, n_philters),
            torch.arange(0, t),
            torch.arange(0, n_y),
            torch.arange(0, n_x),
        )
        b_inds, T_inds, N_y_inds, N_x_inds, CH_inds, phis_inds, t_inds, n_y_inds, n_x_inds = \
            b_inds.reshape(-1), \
            T_inds.reshape(-1), N_y_inds.reshape(-1), N_x_inds.reshape(-1), CH_inds.reshape(-1), \
            phis_inds.reshape(-1), \
            t_inds.reshape(-1), n_y_inds.reshape(-1), n_x_inds.reshape(-1)
        inds = torch.stack([b_inds, T_inds, N_y_inds, N_x_inds, CH_inds, phis_inds, t_inds, n_y_inds, n_x_inds], 1)

        # [batch, T, N_y, N_x, CH, n_philters, t]
        gauss_mus_t_ = gauss_mus_t.detach()
        gauss_sigmas_t_ = gauss_sigmas_t.detach().unsqueeze(6)
        gauss_min_ts = torch.min(torch.tensor(T_max-1, dtype=torch.float32, device=phis.device),
            torch.max(torch.tensor(0, dtype=torch.float32, device=phis.device),
            torch.floor(gauss_mus_t_ - np.sqrt(6) * gauss_sigmas_t_)))
        gauss_max_ts = torch.max(torch.tensor(1, dtype=torch.float32, device=phis.device),
            torch.min(torch.tensor(T_max, dtype=torch.float32, device=phis.device),
            torch.ceil(gauss_mus_t_ + np.sqrt(6) * gauss_sigmas_t_)))
        # [batch, T, N_y, N_x, CH, n_philters, n_y, n_x]
        gauss_mus_x_ = gauss_mus_x.detach()
        gauss_mus_y_ = gauss_mus_y.detach()
        gauss_sigmas_x_ = gauss_sigmas_x.detach().unsqueeze(6).unsqueeze(7)
        gauss_sigmas_y_ = gauss_sigmas_y.detach().unsqueeze(6).unsqueeze(7)
        gauss_rots_ = gauss_rots.detach().unsqueeze(6).unsqueeze(7)
        gauss_min_ys = torch.min(torch.tensor(Y_max-1, dtype=torch.float32, device=phis.device),
            torch.max(torch.tensor(0, dtype=torch.float32, device=phis.device),
            torch.floor(gauss_mus_y_ - \
                np.sqrt(6) * (torch.cos(gauss_rots_)*gauss_sigmas_y_ + \
                    torch.abs(torch.sin(gauss_rots_)*gauss_sigmas_x_)))))
        gauss_max_ys = torch.max(torch.tensor(1, dtype=torch.float32, device=phis.device),
            torch.min(torch.tensor(Y_max, dtype=torch.float32, device=phis.device),
            torch.ceil(gauss_mus_y_ + \
                np.sqrt(6) * (torch.cos(gauss_rots_)*gauss_sigmas_y_ + \
                    torch.abs(torch.sin(gauss_rots_)*gauss_sigmas_x_)))))
        gauss_min_xs = torch.min(torch.tensor(X_max-1, dtype=torch.float32, device=phis.device),
            torch.max(torch.tensor(0, dtype=torch.float32, device=phis.device),
            torch.floor(gauss_mus_x_ - \
                np.sqrt(6) * (torch.cos(gauss_rots_)*gauss_sigmas_x_ + \
                    torch.abs(torch.sin(gauss_rots_)*gauss_sigmas_y_)))))
        gauss_max_xs = torch.max(torch.tensor(1, dtype=torch.float32, device=phis.device),
            torch.min(torch.tensor(X_max, dtype=torch.float32, device=phis.device),
            torch.ceil(gauss_mus_x_ + \
                np.sqrt(6) * (torch.cos(gauss_rots_)*gauss_sigmas_x_ + \
                    torch.abs(torch.sin(gauss_rots_)*gauss_sigmas_y_)))))

        sparse_inds = []
        sparse_ts = []
        sparse_ys = []
        sparse_xs = []
        for ind in inds:
            min_t = gauss_min_ts[ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6]]
            max_t = gauss_max_ts[ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6]]
            min_y = gauss_min_ys[ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[7], ind[8]]
            max_y = gauss_max_ys[ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[7], ind[8]]
            min_x = gauss_min_xs[ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[7], ind[8]]
            max_x = gauss_max_xs[ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[7], ind[8]]

            ts, ys, xs = torch.meshgrid(
                torch.arange(min_t, max_t, dtype=torch.long),
                torch.arange(min_y, max_y, dtype=torch.long),
                torch.arange(min_x, max_x, dtype=torch.long),
            )
            ts, ys, xs = ts.reshape(-1), ys.reshape(-1), xs.reshape(-1)

            if ts.size()[0]:
                sparse_inds.append(torch.cat([
                    ind.unsqueeze(0).repeat(ts.shape[0], 1),
                    ts.unsqueeze(1),
                    ys.unsqueeze(1),
                    xs.unsqueeze(1),
                ], 1))
                sparse_ts.append(0.5 + ts.type(torch.float32).to(phis.device))
                sparse_ys.append(0.5 + ys.type(torch.float32).to(phis.device))
                sparse_xs.append(0.5 + xs.type(torch.float32).to(phis.device))

        # [batch, T, N_y, N_x, CH, n_philters, t, n_y, n_x, T_max, Y_max, X_max]
        sparse_inds = torch.cat(sparse_inds, 0).T
        sparse_ts, sparse_ys, sparse_xs = torch.cat(sparse_ts, 0), torch.cat(sparse_ys, 0), torch.cat(sparse_xs, 0)

        sparse_gauss_mus_t = gauss_mus_t[sparse_inds[0], sparse_inds[1], sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5], sparse_inds[6]]
        sparse_gauss_sigmas_t = gauss_sigmas_t[sparse_inds[0], sparse_inds[1], sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5]]
        sparse_gauss_mus_y = gauss_mus_y[sparse_inds[0], sparse_inds[1], sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5], sparse_inds[7], sparse_inds[8]]
        sparse_gauss_sigmas_y = gauss_sigmas_y[sparse_inds[0], sparse_inds[1], sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5]]
        sparse_gauss_mus_x = gauss_mus_x[sparse_inds[0], sparse_inds[1], sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5], sparse_inds[7], sparse_inds[8]]
        sparse_gauss_sigmas_x = gauss_sigmas_x[sparse_inds[0], sparse_inds[1], sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5]]
        sparse_gauss_rots = gauss_rots[sparse_inds[0], sparse_inds[1], sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5]]
        sparse_Zs = sigma_T[sparse_inds[0], sparse_inds[1], sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5]] * \
            sigma_Y[sparse_inds[0], sparse_inds[1], sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5]] * \
            sigma_X[sparse_inds[0], sparse_inds[1], sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5]]

        sparse_u0 = sparse_gauss_mus_x - sparse_xs
        sparse_u1 = sparse_gauss_mus_y - sparse_ys
        sparse_l1 = 1. / (1e-8 + sparse_gauss_sigmas_x**2)
        sparse_l2 = 1. / (1e-8 + sparse_gauss_sigmas_y**2)
        sparse_c = torch.cos(sparse_gauss_rots)
        sparse_s = torch.sin(sparse_gauss_rots)
        sparse_gauss_ks = ((sparse_gauss_mus_t - sparse_ts) / (1e-8 + sparse_gauss_sigmas_t))**2 + \
            sparse_u0 * (sparse_u0*(sparse_l1*sparse_c**2+sparse_l2*sparse_s**2) + sparse_u1*(sparse_l1-sparse_l2)*sparse_c*sparse_s) + \
            sparse_u1 * (sparse_u0*(sparse_l1-sparse_l2)*sparse_c*sparse_s + sparse_u1*(sparse_l1*sparse_s**2+sparse_l2*sparse_c**2))
        sparse_gauss_vs = torch.exp(-0.5 * sparse_gauss_ks) / (1e-8 + sparse_Zs)
        sparse_gauss_vs = torch.where(sparse_gauss_vs >= np.exp(-3.) / (1e-8 + sparse_Zs), sparse_gauss_vs, torch.zeros_like(sparse_gauss_vs))
        gauss_vs = torch.sparse.FloatTensor(
            sparse_inds.to(phis.device),
            sparse_gauss_vs,
            torch.Size([batch_size, T, N_y, N_x, CH, n_philters, t, n_y, n_x, T_max, Y_max, X_max])
        ).to_dense()

        sparse_as = a[sparse_inds[0], sparse_inds[1], sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5]]
        sparse_phis = phis[sparse_inds[2], sparse_inds[3], sparse_inds[4], sparse_inds[5], sparse_inds[6], sparse_inds[7], sparse_inds[8]]

        sparse_V = torch.sparse.FloatTensor(
            sparse_inds.to(phis.device),
            sparse_gauss_vs.unsqueeze(1) * sparse_as.unsqueeze(1) * sparse_phis,
            torch.Size([batch_size, T, N_y, N_x, CH, n_philters, t, n_y, n_x, T_max, Y_max, X_max, ch])
        )
        sparse_V = torch.sparse.sum(sparse_V, [1, 2, 3, 5, 6, 7, 8])
        V = sparse_V.to_dense().permute(0, 2, 3, 4, 1, 5).reshape(batch_size, T_max, Y_max, X_max, CH*ch)

    else: # not use_sparse
        # do gauss2d calculation, divide results by sigma_X*sigma_Y
        ys, xs = torch.meshgrid(
            0.5 + torch.arange(0, Y_max, dtype=torch.float32, device=phis.device),
            0.5 + torch.arange(0, X_max, dtype=torch.float32, device=phis.device))
        ys = ys.reshape(1, 1, 1, 1, 1, 1, 1, 1, -1)
        xs = xs.reshape(1, 1, 1, 1, 1, 1, 1, 1, -1)

        u0 = gauss_mus_x.unsqueeze(8) - xs
        u1 = gauss_mus_y.unsqueeze(8) - ys
        l1 = 1. / (1e-8 + gauss_sigmas_x**2).unsqueeze(6).unsqueeze(7).unsqueeze(8)
        l2 = 1. / (1e-8 + gauss_sigmas_y**2).unsqueeze(6).unsqueeze(7).unsqueeze(8)
        c = torch.cos(gauss_rots).unsqueeze(6).unsqueeze(7).unsqueeze(8)
        s = torch.sin(gauss_rots).unsqueeze(6).unsqueeze(7).unsqueeze(8)

        gauss_xys = (
            u0 * (u0*(l1*c**2+l2*s**2) + u1*(l1-l2)*c*s) + \
            u1 * (u0*(l1-l2)*c*s + u1*(l1*s**2+l2*c**2))
        ).view(batch_size, T, N_y, N_x, CH, n_philters, n_y, n_x, Y_max, X_max)
        gauss_xys = torch.exp(-0.5 * gauss_xys) / \
            (1e-8 + (sigma_X*sigma_Y).unsqueeze(6).unsqueeze(7).unsqueeze(8).unsqueeze(9))
        # gauss_xys = torch.where(gauss_xys >= np.exp(-3.) / (1e-8 + (sigma_X*sigma_Y).unsqueeze(6).unsqueeze(7).unsqueeze(8).unsqueeze(9)), gauss_xys, torch.zeros_like(gauss_xys))
        # gauss_xys: [batch, T, N_y, N_x, CH, n_philters, n_y, n_x, Y_max, X_max]

        # gauss_ts: [batch, T, N_y, N_x, CH, n_philters, t, T_max]
            # do gauss calculation, divide results by sigma_T
        ts = 0.5 + torch.arange(0, T_max, dtype=torch.float32, device=phis.device).view(
            1, 1, 1, 1, 1, 1, 1, T_max)
        gauss_ts = ((ts - gauss_mus_t.unsqueeze(7)) / (1e-8 + gauss_sigmas_t.unsqueeze(6).unsqueeze(7)))**2
        gauss_ts = torch.exp(-0.5 * gauss_ts) / (1e-8 + sigma_T.unsqueeze(6).unsqueeze(7))
        # gauss_ts = torch.where(gauss_ts >= np.exp(-3.) / (1e-8 + sigma_T.unsqueeze(6).unsqueeze(7)), gauss_ts, torch.zeros_like(gauss_ts))
        # gauss_ts: [batch, T, N_y, N_x, CH, n_philters, t, T_max]

        # gauss_txys: [batch, T, N_y, N_x, CH, n_philters, t, n_y, n_x, T_max, Y_max, X_max]
        gauss_txys = gauss_xys.unsqueeze(6).unsqueeze(9) * \
            gauss_ts.unsqueeze(7).unsqueeze(8).unsqueeze(10).unsqueeze(11)
        gauss_txys = torch.where(gauss_txys >= np.exp(-3.) / (1e-8 + (sigma_X*sigma_Y*sigma_T).unsqueeze(6).unsqueeze(7).unsqueeze(8).unsqueeze(9).unsqueeze(10).unsqueeze(11)), gauss_txys, torch.zeros_like(gauss_txys))

        # phis: [[N_y, N_x], [CH], n_philters, t, n_y, n_x, ch]
        # a: [batch, T, N_y, N_x, [CH], n_philters]
        V = gauss_txys.unsqueeze(9) * \
            a.view(batch_size, T, N_y, N_x, CH, n_philters, 1, 1, 1, 1, 1, 1, 1) * \
            phis.view(1, 1, N_y, N_x, CH, n_philters, t, n_y, n_x, ch, 1, 1, 1)
        V = torch.sum(V, [1, 2, 3, 5, 6, 7, 8]).view(
            batch_size, CH*ch, T_max, Y_max, X_max).permute(
            0, 2, 3, 4, 1)

    ### PLOTTING ###
    if plot_save_dir is not None:
        try:
            os.system("mkdir {}".format(plot_save_dir))
        except:
            pass

        for b in range(batch_size):
            try:
                os.system("mkdir {}/{}".format(plot_save_dir, b))
            except:
                pass

            if not RGB:
                for l in range(CH_max):
                    for i in range(T_max):
                        Z = V[b, i, :, :, l].detach().cpu()

                        fig = plt.figure(figsize=(4, 8))

                        ax = fig.add_subplot(2, 1, 2)
                        ax.imshow(Z.numpy()[::-1], cmap="gray", vmin=0, vmax=1)

                        ax = fig.add_subplot(2, 1, 1, projection='3d')
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.set_zlabel("z")
                        Y, X = torch.meshgrid(torch.arange(Y_max), torch.arange(X_max))
                        if Z.shape[0] == 1:
                            Z = Z.repeat(2, 1)
                            Y, X = torch.meshgrid(torch.arange(2*Y_max), torch.arange(X_max))
                        if Z.shape[1] == 1:
                            Z = Z.repeat(1, 2)
                            Y, X = torch.meshgrid(torch.arange(Y_max), torch.arange(2*X_max))
                        ax.plot_surface(X, Y, Z, color="black", alpha=0.2)
                        ax.set_zlim(0, 1)

                        plt.savefig("{}/{}/t_{:04d}_ch_{:04d}.png".format(plot_save_dir, b, i, l))
                        plt.close()

                    try:
                        os.system("rm {}/{}/ch_{:04d}_vid.mp4".format(plot_save_dir, b, l))
                    except:
                        pass
                    os.system("ffmpeg -r 2 -i {}/{}/t_%04d_ch_{:04d}.png {}/{}/ch_{:04d}_vid.mp4".format(
                        plot_save_dir, b, l, plot_save_dir, b, l))
            else:
                for i in range(T_max):
                    fig = plt.figure(figsize=(12, 8))

                    ax = fig.add_subplot(2, 3, 5)
                    ax.imshow(V[b, i, :, :, :3].detach().cpu().numpy()[::-1], vmin=0, vmax=1)

                    for l in range(3):
                        Z = V[b, i, :, :, l].detach().cpu()

                        ax = fig.add_subplot(2, 3, l, projection='3d')
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.set_zlabel("z")
                        Y, X = torch.meshgrid(torch.arange(Y_max), torch.arange(X_max))
                        if Z.shape[0] == 1:
                            Z = Z.repeat(2, 1)
                            Y, X = torch.meshgrid(torch.arange(2*Y_max), torch.arange(X_max))
                        if Z.shape[1] == 1:
                            Z = Z.repeat(1, 2)
                            Y, X = torch.meshgrid(torch.arange(Y_max), torch.arange(2*X_max))
                        ax.plot_surface(X, Y, Z, color="black", alpha=0.2)
                        ax.set_zlim(0, 1)

                    plt.savefig("{}/{}/t_{:04d}.png".format(plot_save_dir, b, i))
                    plt.close()

                try:
                    os.system("rm {}/{}/vid.mp4".format(plot_save_dir, b))
                except:
                    pass
                os.system("ffmpeg -r 2 -i {}/{}/t_%04d.png {}/{}/vid.mp4".format(
                    plot_save_dir, b, plot_save_dir, b))

    return V

# phis: [[N_y, N_x], [CH], n_philters, t, n_y, n_x, ch]
    # a: [batch, T, N_y, N_x, [CH], n_philters]
    # mu_T, mu_Y, mu_X,
    # sigma_T, sigma_Y, sigma_X,
    # rot,
class HierarchicalGenerativeModel(object):
    def __init__(self,
        phis_list,
        alpha_Ts, alpha_Ys, alpha_Xs,
        stride_Ts, stride_Ys, stride_Xs,
        device,
    ):
        self.n_layers = len(phis_list)
        self.device = device

        for L in range(self.n_layers):
            phis = phis_list[L]

            if len(phis.shape) == 5:
                phis = phis.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            if len(phis.shape) == 6:
                phis = phis.unsqueeze(0).unsqueeze(1)
            if len(phis.shape) == 7:
                phis = phis.unsqueeze(2)

            phis = phis / (1e-8 + torch.sum(torch.abs(phis.detach()), [4, 5, 6, 7]).unsqueeze(
                4).unsqueeze(5).unsqueeze(6).unsqueeze(7))

            phis_list[L] = phis.to(self.device)

        self.phis_list = phis_list
        self.n_philters_list = [phis.shape[3] for phis in phis_list]
        self.alpha_Ts, self.alpha_Ys, self.alpha_Xs = alpha_Ts, alpha_Ys, alpha_Xs
        self.stride_Ts, self.stride_Ys, self.stride_Xs = stride_Ts, stride_Ys, stride_Xs

    def sample(self, n_samples, T, N_y, N_x, CH):
        pass
        #TODO: sample then generate

    def generate_video(self,
        coefs_list,
        phis_list,
        plot_save_dir=None, RGB=False,
        use_sparse=True,
    ):
        top_down_inp = None

        for L in reversed(range(self.n_layers)):
            if type(coefs_list[L]) == list:
                coefs_list[L] = torch.stack(coefs_list[L], 0)

            if top_down_inp is not None:
                coefs_list[L] = coefs_list[L] + top_down_inp

            a, mu_T, mu_Y, mu_X, sigma_T, sigma_Y, sigma_X, rot = coefs_list[L]
            if (L==0) and (plot_save_dir is not None):
                L_plot_save_dir = plot_save_dir + "/L_{:04d}".format(L)
            else:
                L_plot_save_dir = None
            out = generate_video(
                phis_list[L],
                self.alpha_Ts[L], self.alpha_Ys[L], self.alpha_Xs[L],
                self.stride_Ts[L], self.stride_Ys[L], self.stride_Xs[L],
                a, 
                mu_T,  mu_Y, mu_X,
                sigma_T, sigma_Y, sigma_X,
                rot,
                L_plot_save_dir, RGB and (L==0),
                use_sparse,
            )
            if L >= 1:
                batch_size, out_T, out_N_y, out_N_x, out_CH = out.shape
                out = out.view(
                    batch_size, out_T, out_N_y, out_N_x, 
                    out_CH//(8*self.n_philters_list[L-1]), 8, self.n_philters_list[L-1]).permute(
                    5, 0, 1, 2, 3, 4, 6)
                top_down_inp = out
            else:
                return out

    def infer_coefs(self,
        video,
        phis_list,
        use_sparse=True,
        max_itr=50,
        rel_grad_stop_cond=0.01,
        abs_grad_stop_cond=0.01,
        lr=0.001,
        warm_start_vars_list=None,
    ):
        video = video.type(torch.float32)
        batch_size, video_T, video_N_y, video_N_x, video_CH = video.shape

        if not warm_start_vars_list:
            vars_list = []
            for L in range(self.n_layers):
                # create variables
                if L == 0:
                    next_T, next_N_y, next_N_x, next_CH = video_T, video_N_y, video_N_x, video_CH

                _, _, _, n_philters, t, n_y, n_x, ch = phis_list[L].shape
                alpha_T, alpha_Y, alpha_X = self.alpha_Ts[L], self.alpha_Ys[L], self.alpha_Xs[L]
                stride_T, stride_Y, stride_X = self.stride_Ts[L], self.stride_Ys[L], self.stride_Xs[L]

                T = int(np.ceil((next_T/(t*alpha_T) - 1) / stride_T + 1))
                N_y = int(np.ceil((next_N_y/(n_y*alpha_Y) - 1) / stride_Y + 1))
                N_x = int(np.ceil((next_N_x/(n_x*alpha_X) - 1) / stride_X + 1))
                CH = int(np.ceil(next_CH / ch))

                [a, mu_T, mu_Y, mu_X, log_sigma_T, log_sigma_Y, log_sigma_X, raw_rot] = \
                    [torch.zeros([batch_size, T, N_y, N_x, CH, n_philters],
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=True)
                    for _ in range(8)]

                vars_list.extend([a, mu_T, mu_Y, mu_X, log_sigma_T, log_sigma_Y, log_sigma_X, raw_rot])
                next_T, next_N_y, next_N_x, next_CH = T, N_y, N_x, 8*CH*n_philters

                self.coefs_optimizer = torch.optim.Adam(vars_list, lr=lr)
        else:
            vars_list = warm_start_vars_list

        detached_phis_list = [phis.detach() for phis in phis_list]

        import tqdm
        for itr in tqdm.tqdm(range(max_itr)):
            self.coefs_optimizer.zero_grad()
            coefs_list = self.get_coefs_list(vars_list)
            gen_video = self.generate_video(coefs_list, detached_phis_list, use_sparse=use_sparse)
            loss = 1. / batch_size * \
                (torch.sum(torch.abs(gen_video - video)) + \
                    torch.sum(torch.stack([
                        (2**-(1 + L//8)) * torch.sum(torch.abs(var)) for (L, var) in enumerate(vars_list)], 0)))
            print("coefs itr {}:".format(itr), loss)
            loss.backward()
            self.coefs_optimizer.step()

            if all([(torch.abs(var.grad) < abs_grad_stop_cond).all()
                for var in vars_list]):
                break
            if all([(torch.abs(var.grad / (1e-8 + torch.abs(var))) < rel_grad_stop_cond).all()
                for var in vars_list]):
                break

        return vars_list, self.get_coefs_list(vars_list)

    def get_coefs_list(self, vars_list):
        coefs_list = []
        for L in range(self.n_layers):
            this_vars = vars_list[8*L:8*(L+1)]
            coefs_list.append([
                this_vars[0], # a
                this_vars[1], # mu_T
                this_vars[2], # mu_Y
                this_vars[3], # mu_X
                torch.exp(this_vars[4]), # sigma_T
                torch.exp(this_vars[5]), # sigma_Y
                torch.exp(this_vars[6]), # sigma_X
                np.pi/4 * torch.tanh(this_vars[7]) # rot
            ])
        return coefs_list

    def update_phis(self,
        video,
        coefs_list,
        use_sparse=True,
        n_itr=1,
        lr=0.01,
        use_warm_start_optimizer=True,
    ):
        for L in range(self.n_layers):
            self.phis_list[L].requires_grad = True
            for coef in range(8):
                coefs_list[L][coef] = coefs_list[L][coef].detach()

        if not use_warm_start_optimizer:
            self.phis_optimizer = torch.optim.Adam(self.phis_list, lr=lr)

        print("before", self.phis_list[0])
        for itr in range(n_itr):
            self.phis_optimizer.zero_grad()
            gen_video = self.generate_video(coefs_list, self.phis_list, use_sparse=use_sparse)
            loss = torch.mean(torch.abs(gen_video - video))
            print("phis itr {}:".format(itr), loss)
            loss.backward()
            # print(self.phis_list[0].grad)
            self.phis_optimizer.step()

            print("after1", self.phis_list[0])

            # normalize
            for L in range(self.n_layers):
                # phis: [[N_y, N_x], [CH], n_philters, t, n_y, n_x, ch]
                self.phis_list[L] = self.phis_list[L].detach() / \
                    (1e-8 + torch.sum(torch.abs(self.phis_list[L].detach()), [4, 5, 6, 7]).unsqueeze(
                        4).unsqueeze(5).unsqueeze(6).unsqueeze(7))
        print("after2", self.phis_list[0])
