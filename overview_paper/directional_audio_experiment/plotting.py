"""Incremental figures for the directional-audio experiment.

The public ``refresh_*`` functions accept in-progress results and replace PNGs
atomically.  This matters because fits can run for hours while a report process
or a user has the previous figure open.  The older, single-page draft/result
figure is retained at the bottom of this module for backwards compatibility.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from data import Config, MODEL_NAMES, load_prepared


SOURCE_COLORS = ["#D17B24", "#21878A", "#8168AA"]
MODEL_COLORS = ["#3C6FA2", "#27816E", "#BA8627", "#8463A7", "#BC6068"]
LABELS = {"complex_watson": "Complex Watson", "complex_bingham": "Complex Bingham",
          "complex_acg": "Complex ACG", "quotient_phase": "Quotient phase",
          "vmvm": "Uniform-marginal VMVM", "uniform_vmvm": "Uniform-marginal VMVM",
          "uniform_marginals_vmvm": "Uniform-marginal VMVM",
          "complex_gaussian": "Complex Gaussian"}
SHORT = {"complex_watson": "Watson", "complex_bingham": "Bingham",
         "complex_acg": "ACG", "quotient_phase": "Phase",
         "vmvm": "VMVM", "uniform_vmvm": "VMVM", "uniform_marginals_vmvm": "VMVM",
         "complex_gaussian": "Gaussian"}

CONDITION_LABELS = {
    "no_driving_noise": "No driving noise",
    "car_noise": "Car noise",
}
CONDITION_FILENAMES = {
    "no_driving_noise": "performance_no_driving_noise.png",
    "car_noise": "performance_car_noise.png",
}
ATLAS_FILENAMES = {
    "no_driving_noise": "components_no_driving_noise_K3.png",
    "car_noise": "components_car_noise_K3.png",
}
ACTIVITY_FILENAME = "speaker_activity.png"


def _canonical_condition(value: Any) -> str:
    """Return one of the two experiment condition keys.

    ``clean`` is deliberately accepted as an input alias, but filenames and
    serialized figure labels always use the unambiguous
    ``no_driving_noise`` spelling.
    """
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "clean": "no_driving_noise",
        "no_noise": "no_driving_noise",
        "noise_free": "no_driving_noise",
        "noiseless": "no_driving_noise",
        "without_noise": "no_driving_noise",
        "no_driving_noise": "no_driving_noise",
        "noise": "car_noise",
        "noisy": "car_noise",
        "driving_noise": "car_noise",
        "with_car_noise": "car_noise",
        "car_noise": "car_noise",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown condition {value!r}; expected 'no_driving_noise' or 'car_noise'."
        )
    return aliases[key]


def _atomic_save_png(fig: plt.Figure, destination: str | Path, *, dpi: int = 180) -> Path:
    """Save *fig* beside *destination* and publish it with one atomic rename."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp.png"
    try:
        fig.savefig(temporary, format="png", dpi=dpi, facecolor="white",
                    bbox_inches="tight", metadata={"Software": "directional_audio_experiment"})
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _records_list(records: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Accept either a flat record sequence or a progress/results payload."""
    if isinstance(records, Mapping):
        if "records" not in records:
            raise ValueError("A result mapping must contain a 'records' list.")
        records = records["records"]
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of result mappings.")
    answer = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Every result record must be a mapping.")
        # An overall progress file normally contains only finished records.  If
        # callers include live/failed placeholders, do not plot them as data.
        status = str(record.get("status", "complete")).lower()
        if status in {"running", "in_progress", "pending", "failed", "error"}:
            continue
        answer.append(record)
    return answer


def _record_condition(record: Mapping[str, Any]) -> str:
    for key in ("condition", "data_condition", "noise_condition"):
        if key in record:
            return _canonical_condition(record[key])
    raise ValueError("Every incremental result record must contain a 'condition' field.")


def _finite_metric(record: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in record:
            value = float(record[key])
            return value if np.isfinite(value) else None
    return None


def _ordered_models(records: Sequence[Mapping[str, Any]],
                    model_order: Sequence[str] | None) -> list[str]:
    present = {str(record["model"]) for record in records if "model" in record}
    if model_order is None:
        preferred = list(dict.fromkeys(MODEL_NAMES))
    else:
        preferred = list(dict.fromkeys(map(str, model_order)))
    return [name for name in preferred if name in present] + sorted(present - set(preferred))


def _model_palette(models: Sequence[str]) -> dict[str, Any]:
    palette = list(MODEL_COLORS)
    if len(models) > len(palette):
        cmap = plt.get_cmap("tab10")
        palette.extend(cmap(i % 10) for i in range(len(models) - len(palette)))
    return dict(zip(models, palette))


def make_performance_figure(
    records: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    condition: str,
    output_file: str | Path,
    *,
    model_order: Sequence[str] | None = None,
    true_speakers: int = 3,
    dpi: int = 180,
) -> Path:
    """Render one condition's partial model-order results as exactly 1 x 2 axes.

    Expected record fields are ``condition``, ``model``, ``K``,
    ``heldout_log_likelihood`` (a *mean* log density; the explicit alias
    ``heldout_mean_log_likelihood`` is also accepted), and ``heldout_nmi``.
    Duplicate ``(model, K, condition)`` records are resolved in favour of the
    last record, which makes resumed runs safe to visualize.

    Absolute densities from the Euclidean, projective and toroidal models are
    not put on a common scale.  Each curve is instead centred at that family's
    K=1 value.  If K=1 has not finished, the family's likelihood points remain
    absent while its available NMI points are still drawn.
    """
    condition = _canonical_condition(condition)
    all_records = _records_list(records)
    relevant = [record for record in all_records
                if _record_condition(record) == condition]
    models = _ordered_models(relevant, model_order)
    palette = _model_palette(models)

    # Keep the last atomically published record after a resumed/refitted run.
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in relevant:
        if "model" not in record or "K" not in record:
            raise ValueError("Each result record needs 'model' and integer 'K'.")
        K = int(record["K"])
        if K < 1 or K != float(record["K"]):
            raise ValueError(f"Invalid model order K={record['K']!r}.")
        by_key[(str(record["model"]), K)] = record

    counts = sorted({K for _, K in by_key})
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelcolor": "#2D3E4E", "text.color": "#253749"}):
        fig, (ax_ll, ax_nmi) = plt.subplots(1, 2, figsize=(11.2, 4.25),
                                            constrained_layout=True, facecolor="white")
        fig.suptitle(f"Directional audio · {CONDITION_LABELS[condition]}",
                     fontsize=14, weight="bold")

        for model in models:
            rows = sorted(((K, row) for (name, K), row in by_key.items() if name == model),
                          key=lambda item: item[0])
            baseline_row = by_key.get((model, 1))
            baseline = None if baseline_row is None else _finite_metric(
                baseline_row, ("heldout_mean_log_likelihood", "heldout_log_likelihood")
            )
            if baseline is not None:
                ll_points = [(K, value - baseline) for K, row in rows
                             if (value := _finite_metric(
                                 row, ("heldout_mean_log_likelihood",
                                       "heldout_log_likelihood"))) is not None]
                if ll_points:
                    xs, ys = zip(*ll_points)
                    ax_ll.plot(xs, ys, marker="o", ms=4.2, lw=1.7,
                               color=palette[model], label=LABELS.get(model, model))
            nmi_points = [(K, value) for K, row in rows
                          if (value := _finite_metric(row, ("heldout_nmi",))) is not None]
            if nmi_points:
                xs, ys = zip(*nmi_points)
                ax_nmi.plot(xs, ys, marker="o", ms=4.2, lw=1.7,
                            color=palette[model], label=LABELS.get(model, model))

        for ax in (ax_ll, ax_nmi):
            ax.grid(axis="y", color="#E3E8ED", linewidth=.75)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=3, labelsize=8.5)
            ax.axvline(true_speakers, color="#8F9AA5", ls="--", lw=1., zorder=0)
            if counts:
                ax.set_xticks(counts)
                ax.set_xlim(min(counts) - .25, max(counts) + .25)
            ax.set_xlabel("Model order K")

        ax_ll.axhline(0., color="#AEB7C0", lw=.8, zorder=0)
        ax_ll.set_ylabel(
            "Held-out Δ mean joint log likelihood\n"
            "relative to K=1 (nats / multfrequency frame)"
        )
        ax_ll.set_title("Predictive log density (within family)", loc="left", fontsize=10.5)
        ax_nmi.set_ylim(-.025, 1.025)
        ax_nmi.set_yticks([0., .25, .5, .75, 1.])
        ax_nmi.set_ylabel("Held-out NMI")
        ax_nmi.set_title("Dominant-speaker recovery", loc="left", fontsize=10.5)

        if not by_key:
            for ax in (ax_ll, ax_nmi):
                ax.text(.5, .5, "Awaiting completed fits", ha="center", va="center",
                        color="#7D8995", transform=ax.transAxes)
        elif models and not any((model, 1) in by_key for model in models):
            ax_ll.text(.5, .5, "Awaiting K=1 baselines", ha="center", va="center",
                       color="#7D8995", transform=ax_ll.transAxes)

        handles, labels = ax_nmi.get_legend_handles_labels()
        if not handles:
            handles, labels = ax_ll.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncol=min(5, len(handles)),
                       frameon=False, fontsize=8.2)
        path = _atomic_save_png(fig, output_file, dpi=dpi)
        plt.close(fig)
    return path


def refresh_performance_figures(
    records: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    output_directory: str | Path,
    *,
    model_order: Sequence[str] | None = None,
    conditions: Sequence[str] = ("no_driving_noise", "car_noise"),
    true_speakers: int = 3,
) -> dict[str, Path]:
    """Atomically refresh the separate no-noise and car-noise PNGs.

    Call this after appending each completed ``(K, model, condition)`` record.
    Both files are safe to read while they are being refreshed, and absent
    model/K points are intentionally allowed.
    """
    output_directory = Path(output_directory)
    output: dict[str, Path] = {}
    for raw_condition in conditions:
        condition = _canonical_condition(raw_condition)
        filename = CONDITION_FILENAMES[condition]
        output[condition] = make_performance_figure(
            records, condition, output_directory / filename,
            model_order=model_order, true_speakers=true_speakers,
        )
    return output


def _default_geometry() -> dict[str, Any]:
    geometry_file = Path(__file__).with_name("geometry_reference.json")
    return json.loads(geometry_file.read_text())


def _validate_geometry(geometry: Mapping[str, Any], candidate_seats: Sequence[int],
                       microphone_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    seats = np.asarray(geometry["seat_centers_xyz"], dtype=float)
    microphones = np.asarray(geometry["microphones_xyz"], dtype=float)
    if seats.ndim != 2 or seats.shape[1] < 2 or microphones.ndim != 2 or microphones.shape[1] < 2:
        raise ValueError("Geometry seat/microphone coordinates must be two-dimensional arrays of XYZ rows.")
    if any(seat < 1 or seat > len(seats) for seat in candidate_seats):
        raise ValueError("candidate_seats contains a seat absent from the measured geometry.")
    if any(mic < 1 or mic > len(microphones) for mic in microphone_ids):
        raise ValueError("microphone_ids contains a microphone absent from the measured geometry.")
    return seats, microphones


def _relative_score_rows(values: np.ndarray) -> np.ndarray:
    """Map each component's four support scores to [0, 1], preserving order."""
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    for i, row in enumerate(values):
        if not np.all(np.isfinite(row)):
            raise ValueError("All four learned component seat scores must be finite.")
        shifted = row - min(0., float(row.min()))
        maximum = float(shifted.max())
        result[i] = shifted / maximum if maximum > 0 else np.zeros_like(row)
    return result


def _component_car_contour(
    ax: plt.Axes,
    geometry: Mapping[str, Any],
    scores: Sequence[float],
    *,
    candidate_seats: Sequence[int],
    occupied_seats: Sequence[int],
    microphone_ids: Sequence[int],
) -> Any:
    """Draw the full car layout with an inverse-distance contour from four measured seats.

    The shaded field is inverse-distance weighting (IDW) from the four
    candidate-seat steering scores. IDW is defined at every point in the
    plane, so nothing is mathematically undefined by evaluating it across the
    whole cabin footprint (same footprint as the car-layout reference figure)
    instead of masking it to the convex hull of the four support points. The
    honest trade-off: far from the four measured seats the field carries no
    extra spatial information and smoothly relaxes toward a distance-weighted
    blend of all four scores, so it should be read as a smooth connector
    between measured points, not as a fitted spatial density. All eight seats
    are drawn for layout context; only the four seats in ``candidate_seats``
    have measured scores.
    """
    seats, microphones = _validate_geometry(geometry, candidate_seats, microphone_ids)
    seat_xy = np.column_stack((-seats[:, 0], seats[:, 1]))
    support = seat_xy[np.asarray(candidate_seats, dtype=int) - 1]
    mic_xyz = microphones[np.asarray(microphone_ids, dtype=int) - 1]
    mic_points = np.column_stack((-mic_xyz[:, 0], mic_xyz[:, 1]))
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (4,):
        raise ValueError(f"Each component needs four seat scores; received {scores.shape}.")

    ax.set_aspect("equal")
    ax.set_xlim(-.28, 2.45)
    ax.set_ylim(-.90, .90)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((-.15, -.72), 2.53, 1.44,
                                boxstyle="round,pad=0.02,rounding_size=0.21",
                                facecolor="#F8FAFC", edgecolor="#A4AEB8", lw=.85, zorder=-5))
    for x in [.2, 1.87]:
        for y in [-.795, .72]:
            ax.add_patch(FancyBboxPatch((x-.14, y), .28, .075,
                                        boxstyle="round,pad=0.005,rounding_size=0.025",
                                        color="#74808C", lw=0, zorder=-5))
    ax.plot([-.04, -.04], [-.50, .50], color="#CAD3DA", lw=1.1, zorder=-3)
    ax.text(-.205, 0, "FRONT", ha="center", va="center", rotation=90,
            fontsize=6.4, color="#6B7785")

    # The fill domain is the whole measured cabin box above (matching the
    # car-layout reference figure), not the convex hull of the four candidate
    # seats: it reaches past the microphones and comfortably contains all
    # eight seats plus a margin.
    grid_x, grid_y = np.meshgrid(np.linspace(-.15, 2.38, 140), np.linspace(-.72, .72, 88))
    distance_sq = ((grid_x[..., None] - support[:, 0]) ** 2
                   + (grid_y[..., None] - support[:, 1]) ** 2)
    scale = max(np.ptp(support[:, 0]), np.ptp(support[:, 1]))
    weights = 1. / (distance_sq + max(scale * .035, 1e-4) ** 2)
    interpolated = np.sum(weights * scores, axis=-1) / np.sum(weights, axis=-1)
    contour = ax.contourf(grid_x, grid_y, interpolated,
                          levels=np.linspace(0., 1., 11), cmap="Blues",
                          vmin=0., vmax=1., alpha=.92, antialiased=True, zorder=-2)
    ax.contour(grid_x, grid_y, interpolated, levels=np.linspace(.2, .8, 4),
               colors="#426B8A", linewidths=.35, alpha=.55, zorder=-1)

    occupied_lookup = {seat: i for i, seat in enumerate(occupied_seats)}
    candidate_lookup = {seat: value for seat, value in zip(candidate_seats, scores)}
    cmap = plt.get_cmap("Blues")
    for seat_index, (cx, cy) in enumerate(seat_xy):
        seat = seat_index + 1
        ax.add_patch(FancyBboxPatch((cx-.145, cy-.11), .29, .22,
                                    boxstyle="round,pad=0.012,rounding_size=0.025",
                                    facecolor="white", edgecolor="#C5CDD4", lw=.6, zorder=4))
        if seat in candidate_lookup:
            value = candidate_lookup[seat]
            is_occupied = seat in occupied_lookup
            edge = SOURCE_COLORS[occupied_lookup[seat] % len(SOURCE_COLORS)] if is_occupied else "#5F6C78"
            ax.scatter([cx], [cy], s=150, facecolor=[cmap(float(np.clip(value, 0., 1.)))],
                       edgecolor=edge, linewidth=1.8 if is_occupied else 1.15,
                       linestyle="-" if is_occupied else "--", zorder=5)
            text_color = "white" if value > .58 else "#263749"
        else:
            ax.scatter([cx], [cy], s=150, color="#EDF1F5", edgecolors="#B2BCC5",
                       linewidths=.4, zorder=5)
            text_color = "#8792A0"
        ax.text(cx, cy, str(seat), ha="center", va="center", fontsize=7.4,
                color=text_color, weight="bold", zorder=6)

    for mic_id, point in zip(microphone_ids, mic_points):
        ax.scatter(*point, marker="^", s=34, facecolor="#1F2E3D", edgecolor="white",
                   linewidth=.3, zorder=7)
        ax.annotate(f"m{mic_id}", point, xytext=(0, 4.3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.6, color="#1F2E3D", zorder=8)
    return contour


def _coerce_atlas_models(
    seat_scores: Mapping[str, Any],
    display_order: Mapping[str, Sequence[int]] | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, bool]]:
    """Convert model -> array/dict payloads into score rows and display orders."""
    output: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
    for model, payload in seat_scores.items():
        local_order = None
        if isinstance(payload, Mapping):
            for score_key in ("seat_scores", "component_seat_scores", "spatial_scores"):
                if score_key in payload:
                    raw_scores = payload[score_key]
                    break
            else:
                raise ValueError(f"Atlas payload for {model!r} has no 'seat_scores'.")
            local_order = payload.get("display_order")
        else:
            raw_scores = payload
        values = np.asarray(raw_scores, dtype=float)
        if values.shape != (3, 4):
            raise ValueError(
                f"K=3 atlas scores for {model!r} must have shape (3, 4), got {values.shape}."
            )
        requested = None if display_order is None else display_order.get(model)
        has_matching = requested is not None or local_order is not None
        order = np.asarray(requested if requested is not None else
                           local_order if local_order is not None else np.arange(3), dtype=int)
        if order.shape != (3,) or sorted(order.tolist()) != [0, 1, 2]:
            raise ValueError(f"display_order for {model!r} must be a permutation of [0, 1, 2].")
        output[str(model)] = (_relative_score_rows(values), order, has_matching)
    return output


def refresh_component_atlas(
    seat_scores: Mapping[str, Any],
    output_file: str | Path,
    *,
    condition: str,
    geometry: Mapping[str, Any] | None = None,
    occupied_seats: Sequence[int] = (1, 2, 3),
    candidate_seats: Sequence[int] = (1, 2, 3, 5),
    microphone_ids: Sequence[int] = (1, 4, 5, 7),
    display_order: Mapping[str, Sequence[int]] | None = None,
    model_order: Sequence[str] | None = None,
    dpi: int = 180,
) -> Path:
    """Atomically render the learned K=3 component scores on four measured seats.

    ``seat_scores`` is normally ``model -> (3, 4) ndarray``.  A model value may
    instead be ``{"seat_scores": array, "display_order": [..]}``.  Score rows
    are the actual fitted components and columns must follow
    ``candidate_seats=(1, 2, 3, 5)``.  ``display_order[j]`` gives the original
    fitted component shown in display column ``j``.  It changes presentation,
    never the fitted scores.

    The background contours are inverse-distance interpolations anchored at
    four measured steering-score support points, drawn across the whole
    measured cabin (the same car-layout footprint used elsewhere) rather than
    only within the four points' convex hull.  They are intentionally
    labelled schematic and must not be interpreted as fitted spatial
    covariances or calibrated location probabilities.
    """
    condition = _canonical_condition(condition)
    if tuple(candidate_seats) != (1, 2, 3, 5):
        raise ValueError("This four-seat atlas fixes measured candidate seats to (1, 2, 3, 5).")
    if len(occupied_seats) != 3 or not set(occupied_seats).issubset(candidate_seats):
        raise ValueError("Exactly three occupied seats must belong to the four candidate seats.")
    if len(microphone_ids) != 4 or len(set(microphone_ids)) != 4:
        raise ValueError("The atlas requires four distinct microphone IDs.")
    geometry = _default_geometry() if geometry is None else geometry
    parsed = _coerce_atlas_models(seat_scores, display_order)
    models = _ordered_models([{"model": model} for model in parsed], model_order)
    if not models:
        raise ValueError("Cannot draw a component atlas without any completed K=3 model.")

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelcolor": "#2D3E4E", "text.color": "#253749"}):
        fig, axes = plt.subplots(len(models), 3, squeeze=False,
                                 figsize=(11.0, 2.05 * len(models) + 1.2),
                                 constrained_layout=True, facecolor="white")
        fig.suptitle(f"Consensus-refit K=3 cross-frequency steering readout · {CONDITION_LABELS[condition]}",
                     fontsize=14, weight="bold")
        final_contour = None
        all_matched = all(item[2] for item in parsed.values())
        for row, model in enumerate(models):
            values, order, _ = parsed[model]
            for column, component in enumerate(order):
                ax = axes[row, column]
                final_contour = _component_car_contour(
                    ax, geometry, values[component], candidate_seats=candidate_seats,
                    occupied_seats=occupied_seats, microphone_ids=microphone_ids,
                )
                if row == 0:
                    target = (f"\n(matched to occupied seat {occupied_seats[column]})"
                              if all_matched else "")
                    ax.set_title(f"Displayed component {column + 1}{target}",
                                 fontsize=8.4, pad=1.5)
                if column == 0:
                    ax.text(-.08, .5, LABELS.get(model, model), transform=ax.transAxes,
                            ha="right", va="center", rotation=90, fontsize=8.3,
                            weight="bold")
                ax.text(.985, .035, f"fit c{component + 1}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=5.5, color="#667684")

        assert final_contour is not None
        colorbar = fig.colorbar(final_contour, ax=axes.ravel().tolist(), location="right",
                                fraction=.025, pad=.015, ticks=[0., .5, 1.])
        colorbar.set_label("Relative steering score (row max = 1)", fontsize=7.5)
        colorbar.ax.tick_params(labelsize=7, length=2)
        legend = [
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                   markeredgecolor=SOURCE_COLORS[i], markeredgewidth=1.7,
                   label=f"occupied seat {seat}")
            for i, seat in enumerate(occupied_seats)
        ]
        legend.extend([
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                   markeredgecolor="#5F6C78", markeredgewidth=1.15,
                   label="empty measured seat 5"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="#EDF1F5",
                   markeredgecolor="#B2BCC5", label="other seat (no microphone coverage)"),
            Line2D([], [], marker="^", linestyle="none", markerfacecolor="#1F2E3D",
                   markeredgecolor="none", label="measured microphone"),
        ])
        fig.legend(handles=legend, loc="outside lower center", ncol=len(legend),
                   frameon=False, fontsize=6.8)
        fig.text(.5, .012,
                 "Inverse-distance interpolation anchored at four measured seat support "
                 "points, shaded across the full cabin for context; not a fitted location "
                 "covariance or location probability, and uninformative far from those seats.",
                 ha="center", fontsize=7.3, color="#667684")
        path = _atomic_save_png(fig, output_file, dpi=dpi)
        plt.close(fig)
    return path


def refresh_component_atlases(
    scores_by_condition: Mapping[str, Mapping[str, Any]],
    output_directory: str | Path,
    **kwargs: Any,
) -> dict[str, Path]:
    """Refresh any available condition atlases using canonical filenames."""
    output_directory = Path(output_directory)
    output: dict[str, Path] = {}
    for raw_condition, seat_scores in scores_by_condition.items():
        condition = _canonical_condition(raw_condition)
        output[condition] = refresh_component_atlas(
            seat_scores, output_directory / ATLAS_FILENAMES[condition],
            condition=condition, **kwargs,
        )
    return output


def refresh_activity_figure(
    conditions: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    output_file: str | Path,
    *,
    source_positions: Sequence[int],
    train_seconds: float,
    test_seconds: float,
    figure_seconds: float | None = None,
    dpi: int = 180,
) -> Path:
    """Atomically render measured per-source activity for the train/test splits.

    Activity is a clipped-sqrt envelope of isolated clean-source power. It is
    identical across the paired no-noise/car-noise conditions by
    construction (the noise is added after activity is computed), so either
    condition's prepared split represents both.
    """
    train = conditions["no_driving_noise"]["train"]
    test = conditions["no_driving_noise"]["test"]
    cap = np.inf if figure_seconds is None else float(figure_seconds)
    shown = min(cap, float(train["times"][-1]), float(test["times"][-1]))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelcolor": "#2D3E4E", "text.color": "#253749"}):
        fig, (ax_train, ax_test) = plt.subplots(
            2, 1, figsize=(7.6, 5.2), constrained_layout=True, facecolor="white",
        )
        # Reserve dedicated top/bottom strips so the suptitle and footer
        # caption never compete with the layout engine's own placements.
        fig.get_layout_engine().set(rect=(0., .07, 1., .93))
        fig.suptitle("Measured per-source activity", fontsize=13.5, weight="bold")
        _activity(ax_train, train, source_positions, shown,
                 title=f"Train · {train_seconds:g} s total", illustrative=False)
        _activity(ax_test, test, source_positions, shown,
                 title=f"Held out · {test_seconds:g} s total", illustrative=False)
        excerpt = "; first excerpt shown" if shown < min(train["times"][-1], test["times"][-1]) else ""
        ax_test.set_xlabel(f"Time (s){excerpt}", fontsize=8.4, labelpad=3)
        fig.text(.008, .018,
                 "Simultaneous colored rows show overlap. Activity is a clipped-sqrt energy "
                 "envelope of the isolated clean source, identical for both noise conditions "
                 "by construction; train and test use different utterances.",
                 fontsize=7.3, color="#657585")
        path = _atomic_save_png(fig, output_file, dpi=dpi)
        plt.close(fig)
    return path


def _car(ax, geometry, *, active_seats=None, microphone_ids=None,
         scores=None, reference_seat=None, reference_color="#222222",
         compact=False, placeholder=False):
    """Front is LEFT; points use measured coordinates, body is schematic."""
    seats = np.asarray(geometry["seat_centers_xyz"])
    mics = np.asarray(geometry["microphones_xyz"])
    ax.set_aspect("equal")
    ax.set_xlim(-.28, 2.45)
    ax.set_ylim(-.90, .90)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((-.15, -.72), 2.53, 1.44,
                                boxstyle="round,pad=0.02,rounding_size=0.21",
                                facecolor="#F8FAFC", edgecolor="#A4AEB8", lw=.85))
    for x in [.2, 1.87]:
        for y in [-.795, .72]:
            ax.add_patch(FancyBboxPatch((x-.14, y), .28, .075,
                                       boxstyle="round,pad=0.005,rounding_size=0.025",
                                       color="#74808C", lw=0))
    ax.plot([-.04, -.04], [-.50, .50], color="#CAD3DA", lw=1.1)
    ax.text(-.205, 0, "FRONT", ha="center", va="center", rotation=90,
            fontsize=6.1 if compact else 7, color="#6B7785")
    palette = plt.get_cmap("Blues")
    for seat_index, xyz in enumerate(seats):
        seat = seat_index + 1
        cx, cy = -xyz[0], xyz[1]
        ax.add_patch(FancyBboxPatch((cx-.145, cy-.11), .29, .22,
                                    boxstyle="round,pad=0.012,rounding_size=0.025",
                                    facecolor="white", edgecolor="#C5CDD4", lw=.6))
        value = None if scores is None else scores[seat_index]
        if value is not None and np.isfinite(value):
            face = palette(float(value))
            text_color = "white" if value > .65 else "#263749"
        elif active_seats and seat in active_seats:
            face = SOURCE_COLORS[active_seats.index(seat)]
            text_color = "white"
        else:
            face, text_color = "#EDF1F5", "#637381"
        ax.scatter([cx], [cy], s=155 if compact else 205, color=[face],
                   edgecolors="#B2BCC5", linewidths=.4, zorder=4)
        ax.text(cx, cy, str(seat), ha="center", va="center", color=text_color,
                fontsize=7.3 if compact else 8.2, weight="medium", zorder=5)
        if reference_seat == seat:
            ax.scatter([cx], [cy], s=230 if compact else 275,
                       facecolors="none", edgecolors=reference_color, linewidths=1.8, zorder=6)
    if microphone_ids:
        for mic_id in microphone_ids:
            cx, cy = -mics[mic_id-1, 0], mics[mic_id-1, 1]
            ax.scatter([cx], [cy], marker="^", s=55 if not compact else 23,
                       color="#1F2E3D", zorder=7)
            ax.text(cx, cy + (.095 if cy >= 0 else -.095), f"m{mic_id}",
                    ha="center", va="bottom" if cy >= 0 else "top",
                    fontsize=7.1, color="#1F2E3D")


def _activity(ax, split, positions, shown_seconds, *, title, illustrative):
    times, activity = split["times"], split["activity"]
    mask = times <= shown_seconds
    times, activity = times[mask], activity[:, mask]
    for s in range(3):
        base = 2 - s
        ax.axhline(base, color="#DDE3E9", lw=.65, zorder=0)
        ax.fill_between(times, base, base + .76 * activity[s],
                         color=SOURCE_COLORS[s], alpha=.80, lw=0)
        ax.plot(times, base + .76 * activity[s], color=SOURCE_COLORS[s], lw=.55)
    ax.set_yticks([2.3, 1.3, .3], [f"S{s+1} · seat {seat}" for s, seat in enumerate(positions)])
    ax.tick_params(axis="y", length=0, labelsize=8)
    ax.tick_params(axis="x", labelsize=8, length=2, pad=2)
    ax.set_ylim(-.12, 2.96)
    ax.set_xlim(0, shown_seconds)
    ax.set_xticks(np.linspace(0, shown_seconds, 4))
    ax.set_title(title + (" · illustrative" if illustrative else ""), fontsize=9.2, loc="left", pad=5)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#B6C0CA")


def _illustrative_split(seconds: float, offset: int) -> dict:
    t = np.linspace(0, seconds, 700)
    intervals = [
        [(.5, 3.2), (5., 7.8), (11., 15.)],
        [(1.7, 5.), (8., 10.5), (13., 17.)],
        [(3.9, 6.6), (9., 13.), (15.5, 17.8)],
    ]
    activity = np.zeros((3, len(t)))
    for s, episodes in enumerate(intervals):
        for a, b in episodes:
            a = max(0, a + .55 * offset * (-1 if s == 1 else 1))
            b = min(seconds, b + .3 * offset)
            keep = (t >= a) & (t <= b)
            local = t[keep] - a
            taper = np.minimum(1., np.minimum(local, b-a-local) / .2)
            activity[s, keep] = np.maximum(taper, 0) * (.67 + .14*np.sin(14*local+s) + .09*np.sin(29*local))
    return {"times": t, "activity": activity}


def _figure(metadata, train, test, *, results=None, result_arrays=None,
            output_stem: Path, shown_seconds=18., activity_is_illustrative=False):
    draft = results is None
    cfg = Config(**metadata["config"])
    models = cfg.models
    model_colors = {name: MODEL_COLORS[MODEL_NAMES.index(name)] for name in models}
    positions = metadata["source_positions"]
    geometry = metadata["geometry"]
    counts = sorted(set(cfg.component_counts))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.labelcolor": "#2D3E4E", "text.color": "#253749"}):
        fig = plt.figure(figsize=(15.4, 9.7), facecolor="white")
        heading = "Directional audio mixture experiment"
        fig.text(.035, .967, heading, fontsize=21, weight="bold", va="top")
        subtitle = "Three fixed source positions · several frequencies · frozen train-to-test component alignment"
        fig.text(.035, .933, subtitle, fontsize=10.7, color="#687787", va="top")
        if draft:
            fig.text(.966, .960, "DESIGN DRAFT\nNO FITTED RESULTS", ha="right", va="top",
                     fontsize=9.5, color="#A36D26", weight="bold", linespacing=1.35,
                     bbox=dict(facecolor="#FFF5E5", edgecolor="none", boxstyle="round,pad=.5"))

        fig.text(.035, .886, "A   Measured car setup", fontsize=11, weight="bold")
        car = fig.add_axes([.035, .680, .225, .194])
        _car(car, geometry, active_seats=positions, microphone_ids=metadata["microphone_ids"])
        fig.text(.043, .659, "Colored seats: three independent speakers\nTriangles: selected microphone channels\nBody outline schematic; point coordinates measured",
                 fontsize=8.4, color="#687787", linespacing=1.45, va="top")

        fig.text(.306, .886, "B   Source activity, including overlap", fontsize=11, weight="bold")
        ax_train = fig.add_axes([.352, .767, .267, .090])
        ax_test = fig.add_axes([.352, .635, .267, .090])
        _activity(ax_train, train, positions, shown_seconds,
                  title=f"Train · {cfg.train_seconds:g} s total", illustrative=activity_is_illustrative)
        _activity(ax_test, test, positions, shown_seconds,
                  title=f"Held out · {cfg.test_seconds:g} s total", illustrative=activity_is_illustrative)
        ax_test.set_xlabel("Time (s); first excerpt shown", fontsize=8.2, labelpad=3)
        fig.text(.307, .582, "Simultaneous colored rows show overlap. Train and test use different utterances.",
                 fontsize=8.2, color="#687787", va="top")

        fig.text(.684, .886, "C   Choose K; measure source recovery", fontsize=11, weight="bold")
        ax_nmi = fig.add_axes([.718, .764, .245, .104])
        ax_nmi.set_xlim(min(counts)-.18, max(counts)+.18)
        ax_nmi.set_ylim(-.03, 1.04)
        ax_nmi.set_xticks(counts)
        ax_nmi.set_yticks([0, .5, 1.])
        ax_nmi.set_ylabel("Held-out NMI", fontsize=8.7)
        ax_nmi.tick_params(labelsize=8, length=2)
        ax_nmi.grid(axis="y", color="#E7EBEF", linewidth=.7)
        ax_nmi.axvline(3, color="#8E99A5", ls="--", lw=1, zorder=0)
        ax_nmi.spines[["top", "right"]].set_visible(False)
        if draft:
            ax_nmi.text(.5, .54, "Insert NMI curves here\n★ marks K selected by held-out likelihood",
                        ha="center", va="center", fontsize=8.5, color="#8995A1", transform=ax_nmi.transAxes)
        else:
            for family in models:
                rows = sorted([r for r in results["records"] if r["model"] == family], key=lambda r: r["K"])
                xs, ys = [r["K"] for r in rows], [r["heldout_nmi"] for r in rows]
                ax_nmi.plot(xs, ys, "o-", ms=3.3, lw=1.5, color=model_colors[family], zorder=3)
                selected = results["selected_K"][family]
                ax_nmi.plot([selected], [ys[xs.index(selected)]], marker="*", ms=10,
                            color=model_colors[family], mec="white", mew=.4, zorder=4)

        # LL values are centred at each family's own maximum; never compare
        # densities on different sample spaces by their absolute magnitudes.
        ax_ll = fig.add_axes([.718, .625, .245, .090])
        n_models = len(models)
        matrix = np.full((n_models, len(counts)), np.nan)
        shading = np.zeros_like(matrix)
        if not draft:
            for i, family in enumerate(models):
                by_k = {r["K"]: r["heldout_log_likelihood"] for r in results["records"] if r["model"] == family}
                values = np.array([by_k[k] for k in counts])
                matrix[i] = values - values.max()
                span = values.max() - values.min()
                shading[i] = (values - values.min()) / span if span > 0 else .5
        ax_ll.imshow(shading, cmap="Blues", vmin=-.35, vmax=1.45, aspect="auto")
        ax_ll.set_yticks(range(n_models), [SHORT[m] for m in models], fontsize=7.5)
        ax_ll.set_xticks(range(len(counts)), counts, fontsize=8)
        ax_ll.tick_params(length=0, pad=3)
        ax_ll.set_xlabel("Fitted components K  ·  true positions = 3", fontsize=8.1, labelpad=3)
        ax_ll.set_title("Δ log likelihood from row maximum (nats / observation)", fontsize=7.8, loc="left", pad=6)
        for i in range(n_models):
            for j, k in enumerate(counts):
                label = "—" if draft else f"{matrix[i,j]:.2g}"
                ax_ll.text(j, i, label, ha="center", va="center", fontsize=7.2,
                           color="#61788E" if draft else "#263B4F")
                if not draft and k == results["selected_K"][models[i]]:
                    ax_ll.add_patch(Rectangle((j-.47, i-.47), .94, .94, fill=False,
                                              edgecolor="#253749", linewidth=1.25))
        for spine in ax_ll.spines.values():
            spine.set_visible(False)
        fig.text(.684, .582, "Select K within each family; curve colors match the columns below.",
                 fontsize=8.2, color="#687787", va="top")

        fig.text(.035, .541, "D   At K = 3: learned spatial fingerprints for every distribution", fontsize=11, weight="bold")
        fig.text(.035, .519, "Seat shading = posterior-weighted steering similarity. Colored ring = corresponding true source seat.",
                 fontsize=8.6, color="#687787")
        left, right = .152, .966
        width = (right-left) / n_models
        column_y = .485
        for j, family in enumerate(models):
            x = left + width * (j+.5)
            fig.text(x, column_y, LABELS[family], ha="center", fontsize=9.4,
                     color=model_colors[family], weight="bold")
        bottoms = [.348, .218, .088]
        for s, bottom in enumerate(bottoms):
            fig.text(.040, bottom+.084, f"Component ↔ S{s+1}", fontsize=9.4,
                     color=SOURCE_COLORS[s], weight="bold", va="center")
            fig.text(.040, bottom+.058, f"True seat {positions[s]}", fontsize=8.6, color="#6B7B89", va="center")
            if s == 2:
                fig.text(.040, bottom+.016, "One global display\nmatching per model", fontsize=7.6,
                         color="#82909D", linespacing=1.35)
            for j, family in enumerate(models):
                ax = fig.add_axes([left + width*j + .012, bottom, width-.025, .128])
                scores = None
                if not draft:
                    values = result_arrays[family]
                    scores = values["spatial_similarity"][values["display_order"][s]]
                _car(ax, geometry, scores=scores, reference_seat=positions[s],
                     reference_color=SOURCE_COLORS[s], compact=True, placeholder=draft)
        cax = fig.add_axes([.760, .070, .180, .009])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap="Blues"),
                          cax=cax, orientation="horizontal", ticks=[0, .5, 1])
        cb.ax.tick_params(labelsize=7, length=2, pad=1)
        cb.outline.set_linewidth(.4)
        fig.text(.749, .074, "Similarity", ha="right", va="center", fontsize=8)
        if draft:
            footer = ("DRAFT: activity traces are illustrative; result panels are placeholders. " if activity_is_illustrative else
                      "DRAFT: activity is from the constructed recordings; model results are unfilled. ")
        else:
            footer = "Scores use the full held-out recording; the activity panel shows only an excerpt. "
        fig.text(.035, .037, footer + "NMI scores dominant-source identity, not simultaneous-speaker recovery.",
                 fontsize=8.1, color="#657585")
        fig.text(.035, .019,
                 "Spatial similarities are not location probabilities. Training labels only reorder the K=3 display. "
                 "LL shading is scaled within each family. Geometry: In-Car McVAMPIRE.",
                 fontsize=8.0, color="#657585")
        output_stem = Path(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for extension in ["png", "pdf", "svg"]:
            filename = output_stem.with_suffix(f".{extension}")
            fig.savefig(filename, dpi=180, facecolor="white")
            print(f"Figure: {filename}")
        plt.close(fig)


def make_draft(output_stem: Path, prepared_directory: Path | None = None) -> None:
    from dataclasses import asdict
    if prepared_directory is not None:
        metadata, train, test, _ = load_prepared(prepared_directory)
        shown = min(metadata["config"]["figure_seconds"], float(train["times"][-1]), float(test["times"][-1]))
        _figure(metadata, train, test, output_stem=output_stem, shown_seconds=shown)
        return
    geometry_file = Path(__file__).with_name("geometry_reference.json")
    geometry = json.loads(geometry_file.read_text())
    cfg = Config()
    metadata = {"config": asdict(cfg), "source_positions": cfg.source_positions,
                "microphone_ids": cfg.microphone_ids, "geometry": geometry}
    _figure(metadata, _illustrative_split(18., 0), _illustrative_split(18., 1),
            output_stem=output_stem, shown_seconds=18., activity_is_illustrative=True)


def make_results_figure(directory: Path, *, figure_seconds: float = 18.) -> None:
    metadata, train, test, _ = load_prepared(directory)
    results_dir = directory / "results"
    results = json.loads((results_dir / "results.json").read_text())
    if results["preparation_id"] != metadata["preparation_id"]:
        raise ValueError("These fitted results belong to an earlier preparation. Rerun fitting before plotting.")
    # Plot the family/K configuration that actually produced these results.
    metadata["config"].update({k: results["config"][k] for k in ["models", "component_counts"]})
    arrays = {}
    for family in results["config"]["models"]:
        with np.load(results_dir / f"{family}_K3.npz", allow_pickle=False) as archive:
            arrays[family] = {key: archive[key] for key in ["spatial_similarity", "display_order"]}
    shown = min(figure_seconds, float(train["times"][-1]), float(test["times"][-1]))
    _figure(metadata, train, test, results=results, result_arrays=arrays,
            output_stem=directory / "figures" / "experiment_results", shown_seconds=shown)
