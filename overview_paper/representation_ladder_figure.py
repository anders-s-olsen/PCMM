import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def build_channel_data():
    t = np.linspace(0, 10, 1800)
    base_w = 2 * np.pi * 0.42
    phi = np.array([0.15, 0.48, 0.72, 1.02, 1.28])
    w = base_w * np.array([1.00, 1.015, 0.985, 1.010, 0.995])

    amps, thetas, xs = [], [], []
    for j in range(5):
        a = 0.75 + 0.26 * np.sin(0.55 * t + 0.65 * j) + 0.11 * np.sin(1.17 * t + 0.35 * j + 0.6)
        a = np.clip(a, 0.18, None)
        theta = w[j] * t + phi[j] + 0.12 * np.sin(0.35 * t + 0.4 * j)
        x = a * np.cos(theta)
        amps.append(a)
        thetas.append(theta)
        xs.append(x)

    amps = np.array(amps)
    thetas = np.array(thetas)
    xs = np.array(xs)

    candidate = np.where((t > 5.0) & (t < 6.3))[0]
    spreads = np.array([np.std(wrap_to_pi(thetas[:, k] - np.mean(thetas[:, k]))) for k in candidate])
    k0 = candidate[np.argmin(spreads)]
    t0 = t[k0]
    # Give the five channels distinct, evenly spaced phases at the highlighted
    # time point while preserving each channel's temporal evolution.
    target_theta0 = np.linspace(0.2, np.pi / 2 + 0.2, thetas.shape[0])
    thetas += (target_theta0 - wrap_to_pi(thetas[:, k0]))[:, None]
    xs = amps * np.cos(thetas)
    theta0 = wrap_to_pi(thetas[:, k0])
    return t, amps, thetas, xs, t0, theta0


def add_card_background(fig, ax, pad_x=0.006, pad_y=0.028):
    bb = ax.get_position()
    rect = Rectangle((bb.x0 - pad_x, bb.y0 - pad_y), bb.width + 2 * pad_x, bb.height + 2 * pad_y,
                     transform=fig.transFigure, facecolor="#fafafa", edgecolor="#e6e6e6", lw=1.0, zorder=-10)
    fig.patches.append(rect)


def add_connector(fig, a, b, y=0.50):
    arrow = FancyArrowPatch((a.x1 + 0.003, y), (b.x0 - 0.003, y), transform=fig.transFigure,
                            arrowstyle="simple", mutation_scale=8,
                            fc="#7f8fa6", ec="#7f8fa6", alpha=0.9)
    fig.patches.append(arrow)


def add_sample_space_footer(fig, ax, text, color, y=0.18):
    """Add a centered sample-space label with its benchmark color swatch."""
    swatch = DrawingArea(4, 16, 0, 0)
    swatch.add_artist(Rectangle((0, 0), 3, 16, facecolor=color, edgecolor="none"))
    label = TextArea(text, textprops={"fontsize": 9.2, "color": "0.28"})
    footer = HPacker(children=[swatch, label], align="center", pad=0, sep=5)

    bb = ax.get_position(original=True)
    artist = AnnotationBbox(
        footer,
        (bb.x0 + bb.width / 2, y),
        xycoords=fig.transFigure,
        box_alignment=(0.5, 0.5),
        frameon=False,
        pad=0,
    )
    fig.add_artist(artist)


def draw_signals_panel(fig, ax, colors, t, amps, thetas, xs, t0):
    ax.axis('off')
    top = ax.inset_axes([0.04, 0.45, 0.92, 0.37])
    bot = ax.inset_axes([0.04, 0.07, 0.92, 0.28], sharex=top)

    offsets = np.arange(len(colors))[::-1] * 2.0
    for j, c in enumerate(colors):
        off = offsets[j]
        top.plot(t, xs[j] + off, color=c, lw=1.9)
        top.plot(t, amps[j] + off, color="0.78", lw=0.9, ls="--")
        top.plot(t, -amps[j] + off, color="0.78", lw=0.9, ls="--")
        top.text(t[0] - 0.12, off, f"Ch {j+1}", ha="right", va="center", fontsize=9.5, color=c)
        bot.plot(t, wrap_to_pi(thetas[j]), color=c, lw=1.6)

    for axis in [top, bot]:
        axis.axvline(t0, color="0.35", lw=1.2, ls=(0, (3, 3)))
        axis.set_xlim(t[0] - 0.55, t[-1])
        axis.set_xticks([])
        axis.set_yticks([])
        for s in axis.spines.values():
            s.set_visible(False)

    top.text(t0, offsets[0] + 1.72, "$t_0$", ha="center", va="bottom", fontsize=11)
    top.set_ylim(-1.5, offsets[0] + 1.95)
    bot.set_ylim(-3.35, 3.35)
    bot.axhline(0, color='0.90', lw=1)
    bot.text(t[0] - 0.35, 0, "$\\theta$", ha="right", va="center", fontsize=10.5, color="0.35")

    start = top.transAxes.transform((0.93, 0.10))
    end = bot.transAxes.transform((0.93, 0.88))
    start_f = fig.transFigure.inverted().transform(start)
    end_f = fig.transFigure.inverted().transform(end)
    arr = FancyArrowPatch(tuple(start_f), tuple(end_f), transform=fig.transFigure,
                          arrowstyle='-|>', mutation_scale=12, lw=1.4, color='0.45')
    fig.patches.append(arr)
    mx = (start_f[0] + end_f[0]) / 2
    my = (start_f[1] + end_f[1]) / 2
    fig.text(mx - 0.008, my, "Discard\namplitude", ha='right', va='center', fontsize=9.5, color='0.35')

def draw_single_channel_phase_vector(ax, t, theta1):
    ax.set_aspect('equal')
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.add_patch(Circle((0, 0), 1.0, edgecolor="0.74", facecolor="none", lw=1.5))
    ax.axhline(0, color="0.92", lw=1)
    ax.axvline(0, color="0.92", lw=1)

    sample_phases = np.linspace(0.2, np.pi / 2 + 0.2, 5)
    blue = "#2E6FBE"
    for k, ang in enumerate(sample_phases):
        alpha = 0.20 + 0.07 * k
        ax.arrow(0, 0, 0.82 * np.cos(ang), 0.82 * np.sin(ang),
                 length_includes_head=True, head_width=0.06, head_length=0.09,
                 lw=2.0, color=blue, alpha=alpha)
        ax.scatter(np.cos(ang), np.sin(ang), s=36, color=blue, alpha=alpha, zorder=4)

    time_arc = FancyArrowPatch(posA=(1.15, 0.15), posB=(0.15, 1.15),
                               connectionstyle="arc3,rad=0.32", arrowstyle='-|>',
                               mutation_scale=14, lw=1.5, color='0.45')
    ax.add_patch(time_arc)
    ax.text(0.92, 0.97, "Time", fontsize=10, color='0.35', ha='center')

    ax.text(0.5, -0.08, "One channel evolving over time", transform=ax.transAxes,
            ha='center', va='top', fontsize=10, color='0.38')


def draw_projective_class(ax, colors, theta0):
    ax.set_aspect('equal')
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(Circle((0, 0), 1.0, edgecolor='0.80', facecolor='none', lw=1.2))

    rot = 0.7 * np.pi
    for j, (c, ang) in enumerate(zip(colors, theta0)):
        for alpha, shift in [(0.96, 0.0), (0.28, rot)]:
            a = ang + shift
            ax.arrow(0, 0, 0.84 * np.cos(a), 0.84 * np.sin(a),
                     length_includes_head=True, head_width=0.06, head_length=0.09,
                     lw=2.0, color=c, alpha=alpha)
            # Decreasing marker sizes keep nearly coincident channel phases visible
            # without moving them away from their exact locations at t0.
            ax.scatter(np.cos(a), np.sin(a), s=100 - 16 * j, color=c, alpha=alpha,
                       edgecolor="white", linewidth=0.6, zorder=5 + j)

    arc = FancyArrowPatch(posA=(-0.15, -0.2), posB=(0.2, 0.05),
                          connectionstyle='arc3,rad=0.70', arrowstyle='-|>',
                          mutation_scale=14, lw=1.5, color='0.45')
    ax.add_patch(arc)
    ax.text(0.25, -0.50, "Same class\nin $\\mathbb{CP}^{4}$", ha='center', va='center',
            fontsize=10.2, color='0.38')

    ax.text(0.5, -0.08, "All channels at one time point;\ncommon phase is irrelevant", transform=ax.transAxes,
            ha='center', va='top', fontsize=9.8, color='0.38')


def draw_cosine_matrix(ax, theta0):
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    mat_ax = ax.inset_axes([0.05, 0.26, 0.90, 0.68])
    A = np.cos(theta0[:, None] - theta0[None, :])
    im = mat_ax.imshow(A, cmap='coolwarm', vmin=-1, vmax=1)
    n = len(theta0)
    mat_ax.set_xticks(range(n))
    mat_ax.set_yticks(range(n))
    mat_ax.set_xticklabels([f"{i+1}" for i in range(n)], fontsize=9)
    mat_ax.set_yticklabels([f"{i+1}" for i in range(n)], fontsize=9)
    for tick, color in zip(mat_ax.get_xticklabels(), plt.get_cmap('tab10').colors[:n]):
        tick.set_color(color)
    for tick, color in zip(mat_ax.get_yticklabels(), plt.get_cmap('tab10').colors[:n]):
        tick.set_color(color)
    mat_ax.tick_params(length=0)
    for i in range(n + 1):
        mat_ax.axhline(i - 0.5, color='white', lw=1.2)
        mat_ax.axvline(i - 0.5, color='white', lw=1.2)
    for i in range(n):
        for j in range(n):
            val = A[i, j]
            txt_color = 'white' if abs(val) > 0.45 else '0.2'
            mat_ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6.8, color=txt_color)
    for s in mat_ax.spines.values():
        s.set_visible(False)

    ax.text(0.5, -0.08, r"Entries $\cos(\theta_{t_0,i}-\theta_{t_0,j})$", transform=ax.transAxes,
            ha='center', va='center', fontsize=9.8, color='0.38', clip_on=False)

    cax = ax.inset_axes([0.25, -0.22, 0.50, 0.05])
    cb = plt.colorbar(im, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0, pad=2)
    return A


def draw_rank2_eigenspace(ax, theta0):
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.55)
    ax.set_ylim(-0.12, 1.52)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    origin = np.array([0.10, 0.10])
    ex = np.array([1.0, 0.0])
    ey = np.array([0.0, 1.0])
    ez = np.array([0.42, 0.80])

    def proj(vec3):
        return origin + vec3[0] * ex + vec3[1] * ez + vec3[2] * ey

    for vec, lab in [([0, 0, 1.10], r"$\mathbb{R}^5$"), ([0, 0.85, 0], "")]:
        a0 = proj(np.array([0, 0, 0]))
        b = proj(np.array(vec))
        ax.add_patch(FancyArrowPatch(a0, b, arrowstyle='-|>', mutation_scale=13, lw=1.25, color='0.45'))
        if lab:
            ax.text(b[0] + 0.02, b[1] + 0.02, lab, fontsize=10, color='0.40')
    # horizontal axis without label
    a0 = proj(np.array([0, 0, 0]))
    b = proj(np.array([1.18, 0, 0]))
    ax.add_patch(FancyArrowPatch(a0, b, arrowstyle='-|>', mutation_scale=13, lw=1.25, color='0.45'))

    plane_pts = np.array([proj(np.array([0.12, 0.0, 0.18])), proj(np.array([0.95, 0.0, 0.10])),
                          proj(np.array([0.86, 0.72, 0.18])), proj(np.array([0.02, 0.72, 0.24]))])
    ax.add_patch(Polygon(plane_pts, closed=True, facecolor="#dbe9ff", edgecolor="#88a6d8", lw=1.4, alpha=0.87))

    base = proj(np.array([0.18, 0.08, 0.12]))
    c_end = proj(np.array([0.76, 0.08, 0.12]))
    s_end = proj(np.array([0.20, 0.56, 0.17]))
    ax.add_patch(FancyArrowPatch(base, c_end, arrowstyle='-|>', mutation_scale=14, lw=2.2, color="#5d8bd3"))
    ax.add_patch(FancyArrowPatch(base, s_end, arrowstyle='-|>', mutation_scale=14, lw=2.2, color="#84a8e2"))
    ax.text(c_end[0] + 0.02, c_end[1] - 0.02, r"$\mathbf{c}=\cos(\boldsymbol{\theta}_{t_0})$", fontsize=9.2, color="#355f9f")
    ax.text(s_end[0] - 0.02, s_end[1] + 0.02, r"$\mathbf{s}=\sin(\boldsymbol{\theta}_{t_0})$", fontsize=9.2, color="#355f9f")

    coords = np.array([[0.20, 0.05, 0.15], [0.43, 0.22, 0.18], [0.61, 0.40, 0.18], [0.30, 0.48, 0.20]])
    for pt in coords:
        q = proj(pt)
        ax.scatter(q[0], q[1], s=34, color="#4f6d9d", zorder=5)

    ax.text(0.5, -0.06, r"$\mathbf{A}_{t_0}^{(c)}=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^{\top}$" + "\n" + r"$\mathbf{A}^{(c)}_{t_0}=\mathbf{c}\mathbf{c}^{\top}+\mathbf{s}\mathbf{s}^{\top}$, rank $\leq 2$", transform=ax.transAxes,
            ha='center', va='top', fontsize=9.2, color='0.38')


def draw_leading_axis(ax, angle=0.85):
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.28, 1.28)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.add_patch(Circle((0, 0), 1.0, edgecolor="0.82", facecolor="none", lw=1.2))
    dx, dy = np.cos(angle), np.sin(angle)
    for sign in (-1, 1):
        ax.add_patch(FancyArrowPatch(
            (0, 0), (sign * dx, sign * dy), arrowstyle="-|>",
            mutation_scale=16, lw=3.0, color="#365e9d",
            shrinkA=0, shrinkB=0,
        ))
    nx, ny = -dy, dx
    ax.text(0.42 * dx + 0.25 * nx, 0.42 * dy + 0.25 * ny, r"$\mathbf{u}_1$",
            fontsize=11, color="#365e9d", ha="center", va="center")

    ax.text(0.5, -0.08, "Sign-indeterminate\nprojective direction", transform=ax.transAxes,
            ha="center", va="top", fontsize=9.8, color="0.38")


def make_figure(output_base: Path):
    t, amps, thetas, xs, t0, theta0 = build_channel_data()
    colors = plt.get_cmap('tab10').colors[:5]

    fig = plt.figure(figsize=(17.4, 6.6), facecolor='white')
    panel_gap = 0.014
    first_width = 0.19
    panel_width = 0.12
    left = 0.025
    positions = [
        [left, 0.18, first_width, 0.66],
        [left + first_width + panel_gap, 0.18, panel_width, 0.66],
        [left + first_width + panel_gap + (panel_width + panel_gap), 0.18, panel_width, 0.66],
        [left + first_width + panel_gap + 2 * (panel_width + panel_gap), 0.18, panel_width, 0.66],
        [left + first_width + panel_gap + 3 * (panel_width + panel_gap), 0.18, panel_width, 0.66],
        [left + first_width + panel_gap + 4 * (panel_width + panel_gap), 0.18, panel_width, 0.66],
    ]
    axs = [fig.add_axes(p) for p in positions]
    for ax in axs:
        add_card_background(fig, ax)

    draw_signals_panel(fig, axs[0], colors, t, amps, thetas, xs, t0)
    draw_single_channel_phase_vector(axs[1], t, thetas[0])
    draw_projective_class(axs[2], colors, theta0)
    draw_cosine_matrix(axs[3], theta0)
    draw_rank2_eigenspace(axs[4], theta0)
    draw_leading_axis(axs[5])

    panel_titles = [
        "Analytic signal",
        "Single-channel\nphase vector " + r"$e^{i\theta_1(t)}$",
        "Projective phase\nclass " + r"$[\mathbf{z}(t_0)]$",
        "Cosine-coherence\nmatrix " + r"$\mathbf{A}_{t_0}^{(c)}$",
        "Eigenspace\n" + r"$\mathrm{span}(\mathbf{U}(t_0))$",
        "Leading eigenvector\n" + r"$[\mathbf{u}_1(t_0)]$",
    ]
    title_y = 0.85
    for i, (ax, title) in enumerate(zip(axs, panel_titles)):
        bb = ax.get_position(original=True)
        fig.text(bb.x0 + (bb.width / 2), title_y, title,
                 ha="center", va="top",
                 fontsize=12.5, fontweight="bold")

    sample_space_footers = [
        (axs[1], r"Sample space: torus $\mathbb{T}^p$", "#E76F51"),
        (axs[2], r"Complex projective $\mathbb{CP}^{p-1}$", "#2A9D8F"),
        (axs[3], r"PSD $\mathcal{S}_+(p)$", "#D65DB1"),
        (axs[4], r"Grassmann $\mathrm{Gr}(2,p)$", "#8E6CBE"),
        (axs[5], r"Real projective $\mathbb{RP}^{p-1}$", "#457B9D"),
    ]
    for ax, text, color in sample_space_footers:
        add_sample_space_footer(fig, ax, text, color)

    first_bb = axs[0].get_position(original=True)
    fig.text(first_bb.x0 + 0.01, 0.785, "$x_j(t)=a_j(t)\\cos(\\theta_j(t))$",
             ha="left", va="top", fontsize=10.0, color="0.38")

    card_pos = [ax.get_position(original=True) for ax in axs]
    for i in range(len(card_pos) - 1):
        add_connector(fig, card_pos[i], card_pos[i + 1], y=0.50)

    fig.text(0.025, 0.955, "Representation ladder for multivariate signals", fontsize=17,
             fontweight='bold', ha='left')
    fig.text(0.025, 0.92,
             "A multichannel oscillatory signal at time $t_0$ is progressively mapped to signal objects with more invariance and less retained information.",
             fontsize=10.8, color='0.35', ha='left')

    png_path = output_base.with_suffix('.png')
    svg_path = output_base.with_suffix('.svg')
    fig.savefig(png_path, dpi=220, bbox_inches='tight')
    # fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    return png_path, svg_path


def main():
    parser = argparse.ArgumentParser(description='Create a visual representation ladder figure (v4).')
    parser.add_argument('--output-base', default='overview_paper/representation_ladder_visual_v4', help='Output filename without extension')
    args = parser.parse_args()
    base = Path(args.output_base)
    if not base.is_absolute():
        base = Path.cwd() / base
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path, svg_path = make_figure(base)
    print(f'Saved {png_path}')
    print(f'Saved {svg_path}')


if __name__ == '__main__':
    main()
