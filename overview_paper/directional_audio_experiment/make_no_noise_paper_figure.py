"""Build the paper figure for the clean speaker-diarization experiment.

The long experiment publishes small diagnostic figures while it runs.  This
script combines the same prepared data and completed K=3 steering readouts in
a publication layout, and adds train/held-out NMI curves. It deliberately
plots only scores that were successfully serialized by the experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from plotting import (
    MODEL_COLORS,
    SOURCE_COLORS,
    _relative_score_rows,
)


CONDITION = "no_driving_noise"
MODEL_ORDER = [
    "complex_watson",
    "complex_bingham",
    "complex_acg",
    "uniform_vmvm",
    "complex_gaussian",
]
MODEL_COLOR = dict(zip(
    ["complex_acg", "uniform_vmvm", "complex_gaussian",
     "complex_bingham", "complex_watson"],
    MODEL_COLORS,
))
DISPLAY_LABELS = {
    "complex_watson": "Complex Watson",
    "complex_bingham": "Complex Bingham",
    "complex_acg": "Complex ACG",
    "uniform_vmvm": "UMVM",
    "complex_gaussian": "Complex Gaussian",
}
FREQUENCY_COLORS = [
    "#4C78A8", "#72B7B2", "#54A24B",
    "#F2CF5B", "#F58518", "#E45756",
]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _dominant_source(split: dict[str, np.ndarray]) -> np.ndarray:
    """Return the experiment's frequency-averaged dominant-source target."""
    energy = np.asarray(split["source_energy"], dtype=float)
    denominator = np.maximum(energy.sum(axis=0, keepdims=True), 1e-30)
    return np.mean(energy / denominator, axis=1).argmax(axis=0)


def _saloon_path() -> MplPath:
    """A compact five-seat saloon outline in the measured top-view frame."""
    vertices = [
        (-.48, -.40), (-.39, -.61), (.02, -.73), (1.40, -.69),
        (1.60, -.50), (1.66, 0.), (1.60, .50), (1.40, .69),
        (.02, .73), (-.39, .61), (-.48, .40), (-.48, -.40),
    ]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * 10 + [MplPath.CLOSEPOLY]
    return MplPath(vertices, codes)


def _draw_saloon(
    ax: plt.Axes,
    geometry: dict,
    *,
    microphone_ids: list[int],
    occupied_seats: list[int],
    candidate_seats: list[int],
    scores: np.ndarray | None = None,
) -> object | None:
    """Draw only the five-seat saloon portion of the measured geometry."""
    seats = np.asarray(geometry["seat_centers_xyz"], dtype=float)[:5]
    microphones = np.asarray(geometry["microphones_xyz"], dtype=float)
    seat_xy = np.column_stack((-seats[:, 0], seats[:, 1]))
    selected_mics = microphones[np.asarray(microphone_ids, dtype=int) - 1]
    mic_xy = np.column_stack((-selected_mics[:, 0], selected_mics[:, 1]))
    body_path = _saloon_path()

    ax.set_aspect("equal")
    ax.set_xlim(-.57, 1.72)
    ax.set_ylim(-.86, .86)
    ax.axis("off")
    ax.add_patch(PathPatch(body_path, facecolor="#F8FAFC", edgecolor="none", zorder=-5))

    contour = None
    normalized_scores = None
    if scores is not None:
        normalized_scores = np.asarray(scores, dtype=float)
        if normalized_scores.shape != (len(candidate_seats),):
            raise ValueError("Spatial component scores must follow candidate_seats.")
        support = seat_xy[np.asarray(candidate_seats, dtype=int) - 1]
        grid_x, grid_y = np.meshgrid(
            np.linspace(-.48, 1.66, 160), np.linspace(-.73, .73, 100)
        )
        distance_sq = (
            (grid_x[..., None] - support[:, 0]) ** 2
            + (grid_y[..., None] - support[:, 1]) ** 2
        )
        weights = 1. / (distance_sq + .035 ** 2)
        interpolated = np.sum(weights * normalized_scores, axis=-1) / np.sum(weights, axis=-1)
        inside = body_path.contains_points(
            np.column_stack((grid_x.ravel(), grid_y.ravel()))
        ).reshape(grid_x.shape)
        interpolated = np.ma.masked_where(~inside, interpolated)
        contour = ax.contourf(
            grid_x, grid_y, interpolated, levels=np.linspace(0., 1., 11),
            cmap="Blues", vmin=0., vmax=1., alpha=.92, antialiased=True,
            zorder=-3,
        )
        ax.contour(
            grid_x, grid_y, interpolated, levels=np.linspace(.2, .8, 4),
            colors="#426B8A", linewidths=.35, alpha=.55, zorder=-2,
        )

    ax.add_patch(
        PathPatch(body_path, facecolor="none", edgecolor="#9EABB6", lw=.9, zorder=1)
    )
    # A short bonnet and boot make the five-seat crop read as a saloon rather
    # than as the front two rows of the original eight-seat cabin.
    ax.plot([-.08, -.08], [-.58, .58], color="#C8D1D9", lw=.9, zorder=1)
    ax.plot([1.43, 1.43], [-.55, .55], color="#C8D1D9", lw=.9, zorder=1)
    for wheel_x in (-.02, 1.20):
        for wheel_y in (-.775, .705):
            ax.add_patch(FancyBboxPatch(
                (wheel_x - .13, wheel_y), .26, .07,
                boxstyle="round,pad=.004,rounding_size=.02",
                facecolor="#74808C", edgecolor="none", zorder=2,
            ))
    ax.text(-.53, 0., "FRONT", rotation=90, ha="center", va="center",
            fontsize=7.4, color="#6B7785")

    score_lookup = (
        {} if normalized_scores is None
        else dict(zip(candidate_seats, normalized_scores))
    )
    occupied_lookup = {seat: i for i, seat in enumerate(occupied_seats)}
    cmap = plt.get_cmap("Blues")
    for seat_index, (cx, cy) in enumerate(seat_xy, start=1):
        ax.add_patch(FancyBboxPatch(
            (cx - .145, cy - .11), .29, .22,
            boxstyle="round,pad=.012,rounding_size=.025",
            facecolor="white", edgecolor="#C5CDD4", lw=.6, zorder=4,
        ))
        if normalized_scores is None:
            if seat_index in occupied_lookup:
                fill = SOURCE_COLORS[occupied_lookup[seat_index]]
                edge = "#B2BCC5"
                text_color = "white"
                line_style = "-"
            elif seat_index in candidate_seats:
                fill, edge, text_color, line_style = "white", "#5F6C78", "#5F6C78", "--"
            else:
                fill, edge, text_color, line_style = "#EDF1F5", "#B2BCC5", "#8792A0", "-"
        elif seat_index in score_lookup:
            value = float(np.clip(score_lookup[seat_index], 0., 1.))
            fill = cmap(value)
            if seat_index in occupied_lookup:
                edge = SOURCE_COLORS[occupied_lookup[seat_index]]
                line_style = "-"
            else:
                edge, line_style = "#5F6C78", "--"
            text_color = "white" if value > .58 else "#263749"
        else:
            fill, edge, text_color, line_style = "#EDF1F5", "#B2BCC5", "#8792A0", "-"
        ax.scatter(
            [cx], [cy], s=150, facecolor=[fill], edgecolor=edge,
            linewidth=1.7 if seat_index in occupied_lookup else 1.0,
            linestyle=line_style, zorder=5,
        )
        ax.text(cx, cy, str(seat_index), ha="center", va="center", fontsize=8.7,
                color=text_color, weight="bold", zorder=6)

    for display_index, point in enumerate(mic_xy, start=1):
        ax.scatter(*point, marker="^", s=30, facecolor="#1F2E3D",
                   edgecolor="white", linewidth=.3, zorder=7)
        ax.annotate(
            f"m{display_index}", point, xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.0, color="#1F2E3D", zorder=8,
        )
    return contour


def _completed_results(
    results_dir: Path,
    train_truth: np.ndarray,
) -> tuple[dict[str, list[tuple[int, float]]],
           dict[str, list[tuple[int, float]]],
           dict[str, dict[str, np.ndarray]]]:
    train_nmi: dict[str, list[tuple[int, float]]] = {}
    test_nmi: dict[str, list[tuple[int, float]]] = {}
    atlases: dict[str, dict[str, np.ndarray]] = {}
    for model in MODEL_ORDER:
        train_nmi[model] = []
        test_nmi[model] = []
        for K in range(1, 6):
            path = results_dir / f"{CONDITION}_{model}_K{K}.npz"
            if not path.exists():
                continue
            values = _load_npz(path)
            predicted = np.asarray(values["final_train_predicted_labels"], dtype=int)
            train_nmi[model].append((K, float(normalized_mutual_info_score(
                train_truth, predicted, average_method="arithmetic"
            ))))
            valid = np.asarray(values["test_valid_frames"], dtype=bool)
            test_nmi[model].append((K, float(normalized_mutual_info_score(
                np.asarray(values["test_truth"], dtype=int)[valid],
                np.asarray(values["test_predicted"], dtype=int)[valid],
                average_method="arithmetic",
            ))))
            if K == 3:
                atlases[model] = {
                    "scores": _relative_score_rows(values["component_seat_scores"]),
                    "order": np.asarray(values["display_order"], dtype=int),
                }
    return train_nmi, test_nmi, atlases


def _plot_source_power(
    ax: plt.Axes,
    split: dict[str, np.ndarray],
    train_scale: np.ndarray,
    positions: list[int],
    shown_seconds: float,
    title: str,
) -> None:
    """Stack the six individual selected-frequency powers for each source."""
    time = np.asarray(split["times"], dtype=float)
    keep = time <= shown_seconds
    time = time[keep]
    energy = np.asarray(split["source_energy"], dtype=float)[:, :, keep]
    frequency_power = np.maximum(
        energy / np.maximum(train_scale[:, None, None], 1e-30), 0.
    )
    # Cap the stacked total—not individual frequencies—so exceptionally loud
    # frames remain inside their row while retaining spectral proportions.
    frequency_power /= np.maximum(1., frequency_power.sum(axis=1, keepdims=True))
    for source in range(3):
        baseline = 2 - source
        cumulative = np.zeros_like(time)
        for frequency, color in enumerate(FREQUENCY_COLORS):
            next_curve = cumulative + 0.76 * frequency_power[source, frequency]
            ax.fill_between(
                time, baseline + cumulative, baseline + next_curve,
                color=color, linewidth=0,
            )
            cumulative = next_curve
        ax.axhline(baseline, color="#DDE3E9", lw=.55, zorder=0)
    ax.set_xlim(0., shown_seconds)
    ax.set_ylim(-.08, 2.83)
    ax.set_xticks(np.linspace(0., shown_seconds, 4))
    ax.set_yticks(
        [2.28, 1.28, .28],
        [f"seat {seat}" for seat in positions],
    )
    ax.set_title(title, fontsize=10.5, loc="left", pad=4)
    ax.tick_params(axis="y", length=0, labelsize=9.0)
    ax.tick_params(axis="x", length=2.5, labelsize=8.8, pad=1.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#B6C0CA")


def _plot_nmi(
    ax: plt.Axes,
    values: dict[str, list[tuple[int, float]]],
    title: str,
    *,
    show_ylabel: bool,
) -> None:
    for model in MODEL_ORDER:
        points = values[model]
        if not points:
            continue
        x, y = zip(*points)
        ax.plot(
            x, y, "o-", color=MODEL_COLOR[model], lw=1.55, ms=3.4,
            label=DISPLAY_LABELS[model], zorder=3,
        )
    ax.axvline(3, color="#8E99A5", ls="--", lw=.9, zorder=0)
    ax.set_xlim(.8, 5.2)
    ax.set_ylim(-.025, 1.025)
    ax.set_xticks(range(1, 6))
    ax.set_yticks([0., .5, 1.])
    ax.set_xlabel("K", fontsize=9.5, labelpad=2)
    ax.set_ylabel(
        "Normalized mutual information" if show_ylabel else "",
        fontsize=9.5, labelpad=5,
    )
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    ax.set_title(title, fontsize=10.5, loc="left", pad=4)
    ax.tick_params(labelsize=8.8, length=2.5, pad=2)
    ax.grid(axis="y", color="#E3E8ED", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(output_dir: Path, output_stem: Path) -> list[Path]:
    metadata = json.loads((output_dir / "prepared.json").read_text())
    train = _load_npz(output_dir / f"train_{CONDITION}.npz")
    test = _load_npz(output_dir / f"test_{CONDITION}.npz")
    progress = json.loads((output_dir / "results" / "progress.json").read_text())
    if progress["preparation_id"] != metadata["preparation_id"]:
        raise ValueError("Result and prepared-data identifiers do not match.")

    positions = list(map(int, metadata["source_positions"]))
    candidate_seats = list(map(int, metadata["display_seat_ids"]))
    microphone_ids = list(map(int, metadata["microphone_ids"]))
    train_truth = _dominant_source(train)
    train_nmi, test_nmi, atlases = _completed_results(
        output_dir / "results", train_truth
    )
    missing_atlas = set(MODEL_ORDER) - set(atlases)
    if missing_atlas:
        raise FileNotFoundError(f"Missing K=3 result(s): {sorted(missing_atlas)}")

    # One train-derived scale per source keeps train and held-out powers
    # comparable without using held-out data to set a display normalization.
    train_scale = np.percentile(
        train["source_energy"].sum(axis=1), 95, axis=1
    )
    shown_seconds = min(
        18., float(train["times"][-1]), float(test["times"][-1])
    )

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelcolor": "#2D3E4E",
        "text.color": "#253749",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }):
        fig = plt.figure(figsize=(15.8, 10.8), facecolor="white")
        fig.text(
            .035, .975, "Speaker diarization in a car",
            fontsize=24, weight="bold", va="top",
        )

        # A: measured geometry.
        fig.text(.035, .900, "A   Measured car setup", fontsize=13.5, weight="bold")
        geometry_ax = fig.add_axes([.035, .705, .205, .175])
        _draw_saloon(
            geometry_ax, metadata["geometry"], microphone_ids=microphone_ids,
            occupied_seats=positions, candidate_seats=candidate_seats,
        )

        # B: individual-frequency isolated-source power.
        fig.text(.270, .900, "B   Source power", fontsize=13.5, weight="bold")
        train_ax = fig.add_axes([.300, .790, .280, .082])
        test_ax = fig.add_axes([.300, .690, .280, .082])
        _plot_source_power(
            train_ax, train, train_scale, positions, shown_seconds,
            "Train · first 18 of 45 s",
        )
        # Keep shared time labels on the lower axis only; with the larger
        # typography, the upper tick labels would crowd the held-out subtitle.
        train_ax.tick_params(axis="x", labelbottom=False)
        _plot_source_power(
            test_ax, test, train_scale, positions, shown_seconds,
            "Held out · first 18 of 30 s",
        )
        test_ax.set_xlabel("Time (s)", fontsize=9.5, labelpad=3)
        frequency_labels = [
            "250 Hz", "898 Hz", "1.55 kHz",
            "2.20 kHz", "2.85 kHz", "3.50 kHz",
        ]
        frequency_handles = [
            Line2D([], [], color=color, lw=5, label=label)
            for color, label in zip(FREQUENCY_COLORS, frequency_labels)
        ]
        fig.legend(
            handles=frequency_handles, loc="upper center",
            bbox_to_anchor=(.440, .647), ncol=3, frameon=False,
            fontsize=8.0, handlelength=1.5, columnspacing=1.0,
            labelspacing=.6,
        )

        # C: train and held-out NMI, without likelihood panels.
        fig.text(.625, .900, "C   Dominant-speaker recovery", fontsize=13.5, weight="bold")
        train_nmi_ax = fig.add_axes([.635, .695, .145, .170])
        test_nmi_ax = fig.add_axes([.815, .695, .145, .170])
        _plot_nmi(train_nmi_ax, train_nmi, "Train", show_ylabel=True)
        _plot_nmi(test_nmi_ax, test_nmi, "Held out", show_ylabel=False)
        model_handles = [
            Line2D([], [], color=MODEL_COLOR[model], marker="o", ms=3.2,
                   lw=1.5, label=DISPLAY_LABELS[model])
            for model in MODEL_ORDER
        ]
        fig.legend(
            handles=model_handles, loc="upper center", bbox_to_anchor=(.798, .674),
            ncol=3, frameon=False, fontsize=8.0, handlelength=1.8,
            columnspacing=1.1, labelspacing=.6,
        )

        # D: K=3 learned components, transposed to match the draft layout.
        fig.text(
            .035, .600, "D   Spatial components at K = 3",
            fontsize=13.5, weight="bold",
        )
        atlas_left, atlas_right = .115, .965
        atlas_width = (atlas_right - atlas_left) / len(MODEL_ORDER)
        atlas_bottoms = [.405, .225, .045]
        for column, model in enumerate(MODEL_ORDER):
            fig.text(
                atlas_left + atlas_width * (column + .5), .572,
                DISPLAY_LABELS[model], ha="center", fontsize=11.2, weight="bold",
                color=MODEL_COLOR[model],
            )
            scores = atlases[model]["scores"]
            order = atlases[model]["order"]
            for row, bottom in enumerate(atlas_bottoms):
                ax = fig.add_axes([
                    atlas_left + column * atlas_width + .006,
                    bottom,
                    atlas_width - .012,
                    .145,
                ])
                component = int(order[row])
                _draw_saloon(
                    ax, metadata["geometry"], scores=scores[component],
                    candidate_seats=candidate_seats, occupied_seats=positions,
                    microphone_ids=microphone_ids,
                )
                if column == 0:
                    ax.text(
                        -.12, .5, f"Component {row + 1}",
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=10.5, color=SOURCE_COLORS[row], weight="bold",
                    )

        colorbar_ax = fig.add_axes([.780, .018, .170, .009])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=colors.Normalize(0., 1.), cmap="Blues"),
            cax=colorbar_ax, orientation="horizontal", ticks=[0., .5, 1.],
        )
        colorbar.ax.tick_params(labelsize=8.2, length=2.5, pad=1.5)
        colorbar.outline.set_linewidth(.4)
        fig.text(
            .772, .022, "Relative spatial score", ha="right", va="center",
            fontsize=9.0,
        )

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        written = []
        for extension in ("png", "pdf", "svg"):
            path = output_stem.with_suffix(f".{extension}")
            fig.savefig(path, dpi=220, facecolor="white")
            written.append(path)
        plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).with_name("output_paired"),
    )
    parser.add_argument(
        "--output-stem", type=Path, default=None,
        help="Default: <output-dir>/results/figures/no_driving_noise_experiment",
    )
    args = parser.parse_args()
    stem = args.output_stem or (
        args.output_dir / "results" / "figures" / "no_driving_noise_experiment"
    )
    for path in make_figure(args.output_dir, stem):
        print(path)


if __name__ == "__main__":
    main()
