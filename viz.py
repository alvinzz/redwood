import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import os

# phis: [[N_y, N_x], [CH], n_philters, t, n_y, n_x, ch]
def viz_phis(phis, plot_save_dir):
    try:
        os.system("mkdir {}".format(plot_save_dir))
    except:
        pass

    if len(phis.shape) == 5:
        phis = phis.unsqueeze(0).unsqueeze(1).unsqueeze(2)
    if len(phis.shape) == 6:
        phis = phis.unsqueeze(0).unsqueeze(1)
    if len(phis.shape) == 7:
        phis = phis.unsqueeze(2)
    N_y, N_x, CH, n_philters, t, n_y, n_x, ch = phis.shape

    phis = phis.detach().numpy()

    phi_px_size = 20

    figheight = N_y*n_philters*(phi_px_size*n_y+10) + 25 + 50
    figwidth = N_x*ch*(phi_px_size*n_x+10) + 50 + 5

    figs = []
    for l in range(CH):
        for p in range(t):
            fig = plt.figure(figsize=(figwidth/100, figheight/100))

            fig.text(
                x=0.5,
                y=(figheight-20)/figheight,
                s="CH={}, t={}".format(l, p),
                fontsize=15,
                ha="center")
            
            ax = fig.add_axes([50/figwidth, 0, 1-50/figwidth, 50/figheight], frameon=False)
            ax.set_xlim(50, figwidth)
            ax.set_ylim(0, 50)
            ax.text(x=50+(figwidth-55)/2, y=5, s="N_x (ch)", fontsize=10, ha="center")
            ax.add_line(lines.Line2D([50, figwidth-5], [30, 30], color="black"))
            ax.add_line(lines.Line2D([51, 51], [30, 35], color="black"))
            ax.add_line(lines.Line2D([figwidth-5, figwidth-5], [30, 35], color="black"))
            for k in range(N_x):
                ax.add_line(lines.Line2D(
                    [50+(phi_px_size*n_x+10)*ch*k, 50+(phi_px_size*n_x+10)*ch*k],
                    [30, 35], color="black", lw=1))
                ax.text(x=50+(phi_px_size*n_x+10)*ch*(k+0.5), y=18, s=str(k), ha="center", fontsize=10)

                ax.add_line(lines.Line2D(
                    [50+5+(phi_px_size*n_x+10)*ch*k, 50+(phi_px_size*n_x+10)*ch*(k+1)-5],
                    [45, 45], color="black", lw=1))
                ax.add_line(lines.Line2D(
                    [50+5+(phi_px_size*n_x+10)*ch*k, 50+5+(phi_px_size*n_x+10)*ch*k],
                    [45, 50], color="black", lw=1))
                ax.add_line(lines.Line2D(
                    [50+(phi_px_size*n_x+10)*ch*(k+1)-5, 50+(phi_px_size*n_x+10)*ch*(k+1)-5],
                    [45, 50], color="black", lw=1))
                for c in range(ch):
                    if c >= 1:
                        ax.add_line(lines.Line2D(
                            [
                                50+(phi_px_size*n_x+10)*(ch*k+c),
                                50+(phi_px_size*n_x+10)*(ch*k+c)
                            ],
                            [45, 50], color="black", lw=1))
                    ax.text(x=50+(phi_px_size*n_x+10)*(ch*k+c+0.5), y=35,
                        s=str(c), ha="center", fontsize=8)
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

            ax = fig.add_axes([0, 50/figheight, 50/figwidth, 1-75/figheight], frameon=False)
            ax.set_xlim(0, 50)
            ax.set_ylim(50, figheight-25)
            ax.text(x=3, y=50+(figheight-75)/2,
                s="N_y (phi)", fontsize=10, rotation="vertical", va="center")
            ax.add_line(lines.Line2D([30, 30], [50, figheight-25], color="black"))
            ax.add_line(lines.Line2D([30, 35], [51, 51], color="black"))
            ax.add_line(lines.Line2D([30, 35], [figheight-25, figheight-25], color="black"))
            for j in range(N_y):
                ax.add_line(lines.Line2D(
                    [30, 35],
                    [
                        50+(phi_px_size*n_y+10)*n_philters*j,
                        50+(phi_px_size*n_y+10)*n_philters*j
                    ], color="black", lw=1))
                ax.text(x=16, y=50+(phi_px_size*n_y+10)*n_philters*(j+0.5),
                    s=str(j), va="center", fontsize=10, rotation="vertical")

                ax.add_line(lines.Line2D(
                    [45, 45],
                    [
                        50+5+(phi_px_size*n_y+10)*n_philters*j,
                        50+(phi_px_size*n_y+10)*n_philters*(j+1)-5
                    ], color="black", lw=1))
                ax.add_line(lines.Line2D(
                    [45, 50],
                    [
                        50+5+(phi_px_size*n_y+10)*n_philters*j,
                        50+5+(phi_px_size*n_y+10)*n_philters*j
                    ], color="black", lw=1))
                ax.add_line(lines.Line2D(
                    [45, 50],
                    [
                        50+(phi_px_size*n_y+10)*n_philters*(j+1)-5,
                        50+(phi_px_size*n_y+10)*n_philters*(j+1)-5
                    ], color="black", lw=1))
                for phi in range(n_philters):
                    if phi >= 1:
                        ax.add_line(lines.Line2D(
                            [45, 50],
                            [
                                50+(phi_px_size*n_y+10)*(ch*j+phi),
                                50+(phi_px_size*n_y+10)*(ch*j+phi)
                            ], color="black", lw=1))
                    ax.text(x=33, y=50+(phi_px_size*n_y+10)*(ch*j+phi+0.5),
                        s=str(phi), va="center", fontsize=8, rotation="vertical")
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

            ax = fig.add_axes([50/figwidth, 50/figheight, 1-55/figwidth, 1-75/figheight])
            ax.set_xlim(50, figwidth-5)
            ax.set_ylim(50, figheight-25)
            for j in range(0, N_y):
                ax.add_line(lines.Line2D(
                    [50, figwidth-5],
                    [50+j*n_philters*(phi_px_size*n_y+10), 50+j*n_philters*(phi_px_size*n_y+10)],
                    color="black", lw=1))
                for phi in range(1, n_philters):
                    ax.add_line(lines.Line2D(
                    [50, figwidth-5],
                    [50+(j*n_philters+phi)*(phi_px_size*n_y+10), 50+(j*n_philters+phi)*(phi_px_size*n_y+10)],
                    color="black", lw=1, ls="--"))
            for k in range(1, N_x):
                ax.add_line(lines.Line2D(
                    [50+k*ch*(phi_px_size*n_x+10), 50+k*ch*(phi_px_size*n_x+10)],
                    [50, figheight-25],
                    color="black", lw=1))
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

            figs.append(fig)

    for phi in range(n_philters):
        for l in range(CH):
            for j in range(N_y):
                for k in range(N_x):
                    for c in range(ch):
                        for p in range(t):
                            fig = figs[t*l + p]

                            x_start = (50+5+(k*ch+c)*(phi_px_size*n_x+10)) / figwidth
                            y_start = (50+5+(j*n_philters+phi)*(phi_px_size*n_y+10)) / figheight
                            x_range = phi_px_size*n_x / figwidth
                            y_range = phi_px_size*n_y / figheight

                            ax = fig.add_axes([x_start, y_start, x_range, y_range])
                            ax.imshow(phis[j, k, l, phi, p, ::-1, :, c], cmap="gray", vmin=0, vmax=1)
                            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    for l in range(CH):
        for p in range(t):
            fig = figs[t*l + p]
            fig.savefig("{}/CH_{:04d}_t_{:04d}.png".format(plot_save_dir, l, p))

        try:
            os.system("rm {}/CH_{:04d}_vid.mp4".format(plot_save_dir, l))
        except:
            pass
        os.system("ffmpeg -r 2 -i {}/CH_{:04d}_t_%04d.png {}/CH_{:04d}_vid.mp4".format(
            plot_save_dir, l, plot_save_dir, l))

# coefs: [coefs, batch, T, N_y, N_x, [CH], n_philters]
