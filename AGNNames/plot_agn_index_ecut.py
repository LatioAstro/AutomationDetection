#!/usr/bin/env python3
"""Scatter plot AGN spectral index versus high-energy cutoff.

The script reads a text table with spectral index and Ecut columns. It is
intentionally permissive about column names so it can handle files with headers
such as:

    Source  Gamma  Ecut_keV  Status
    NGC4151 1.75   120       detection
    Mrk509  1.90   300       upperlimit

If no usable header is found, the first two numeric columns are interpreted as
spectral index and Ecut. Upper limits can be marked with values such as
"upperlimit", "upper_limit", "ul", "<", or "limit".
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DETECTED_ONLY = False  # If True, only include detected sources in the output CSV


@dataclass(frozen=True)
class AGNPoint:
    spectral_index: float
    ecut_keV: float
    is_upper_limit: bool
    spectral_index_err_low: float | None = None
    spectral_index_err_high: float | None = None
    ecut_err_low: float | None = None
    ecut_err_high: float | None = None
    label: str = ""


@dataclass(frozen=True)
class Measurement:
    value: float
    is_limit: bool
    err_low: float | None = None
    err_high: float | None = None


@dataclass(frozen=True)
class BATSource:
    names: tuple[str, ...]
    flux_bat: float
    photon_index: float


INDEX_COLUMNS = {
    "index",
    "gamma",
    "photonindex",
    "photon_index",
    "spectralindex",
    "spectral_index",
    "plindex",
    "pl_index",
}
ECUT_COLUMNS = {
    "ecut",
    "e_cut",
    "ecutkev",
    "ecut_kev",
    "ecutkeV",
    "cutoff",
    "cutoffkev",
    "cutoff_kev",
    "cutoffenergy",
    "cutoff_energy",
}
LIMIT_COLUMNS = {
    "limit",
    "limits",
    "status",
    "flag",
    "detection",
    "upperlimit",
    "upper_limit",
    "isupperlimit",
    "is_upper_limit",
    "ul",
    "uplim",
}
LABEL_COLUMNS = {"source", "name", "object", "agn", "targetname", "target_name"}
UPPER_LIMIT_VALUES = {"<", "ul", "uplim", "upper", "upperlimit", "upper_limit", "limit"}
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
ELECTRON_REST_ENERGY_KEV = 511.0
SPECIAL_SENSITIVITY_LABELS = {"circinusgalaxy", "ic4329a", "ngc1068", "ngc4151"}


def normalized_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("\\mathrm", "")
        .replace("\\rm", "")
        .replace("\\", "")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
        .replace("^", "")
        .replace("_", "")
        .replace("/", "")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )


def normalized_source_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def split_line(line: str) -> list[str]:
    if "\t" in line:
        return [field.strip() for field in line.split("\t") if field.strip()]
    if "," in line:
        return [field.strip() for field in next(csv.reader([line]))]
    return line.split()


def is_float(value: str) -> bool:
    try:
        parse_measurement(value)
    except ValueError:
        return False
    return True


def clean_numeric(value: str) -> str:
    return value.strip().replace("<", "").replace(">", "")


def parse_measurement(value: str) -> Measurement:
    """Return the central value, limit flag, and parsed uncertainties.

    Handles plain numbers, values with symmetric errors, and common LaTeX-ish
    table entries such as ${170}_{-30}^{+40}$ or $\\gt 230$.
    """

    is_limit = any(token in value for token in (">", "\\gt", "\\gtrsim", "<", "\\lt"))
    numbers = NUMBER_PATTERN.findall(value.replace("+or-", " "))
    if not numbers:
        raise ValueError(f"No numeric value found in {value!r}")

    central_value = float(numbers[0])
    err_low: float | None = None
    err_high: float | None = None

    if "+or-" in value and len(numbers) >= 2:
        err_low = abs(float(numbers[1]))
        err_high = abs(float(numbers[1]))
    elif len(numbers) >= 3:
        err_low = abs(float(numbers[1]))
        err_high = abs(float(numbers[2]))
    elif len(numbers) >= 2 and "_{-" in value:
        err_low = abs(float(numbers[1]))

    return Measurement(central_value, is_limit, err_low, err_high)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def is_upper_limit_value(value: str) -> bool:
    normalized = normalized_name(value)
    return normalized in UPPER_LIMIT_VALUES or normalized.startswith("<")


def j_of_tau(tau: float) -> float:
    if tau <= 1.0:
        return 2.0
    if tau < 2.0:
        return tau + 1.0
    return 3.0


def tau_left_hand_side(tau: float) -> float:
    return tau * (1.0 + tau / 3.0) / j_of_tau(tau)


def tau_right_hand_side(spectral_index: float, ecut_keV: float) -> float | None:
    """Right-hand side of the tau relation, including the 1/Ecut dependence."""

    denominator = (spectral_index + 0.5) ** 2 - (9.0 / 4.0)
    if denominator <= 0.0 or ecut_keV <= 0.0:
        return None

    return (ELECTRON_REST_ENERGY_KEV / ecut_keV) / denominator


def ecut_contour_for_tau(spectral_index: float, tau: float) -> float | None:
    """Invert tau(Gamma, Ecut) to draw the constant-tau contour."""

    denominator = (spectral_index + 0.5) ** 2 - (9.0 / 4.0)
    tau_factor = tau_left_hand_side(tau)
    if denominator <= 0.0 or tau_factor <= 0.0:
        return None

    return ELECTRON_REST_ENERGY_KEV / (denominator * tau_factor)


def header_indices(header: list[str]) -> tuple[int | None, int | None, int | None, int | None]:
    normalized = [normalized_name(column) for column in header]

    index_idx = next(
        (i for i, name in enumerate(normalized) if name in INDEX_COLUMNS or "gamma" in name),
        None,
    )
    ecut_idx = next(
        (
            i
            for i, name in enumerate(normalized)
            if name in ECUT_COLUMNS or "ecut" in name or "cut" in name
        ),
        None,
    )
    limit_idx = next((i for i, name in enumerate(normalized) if name in LIMIT_COLUMNS), None)
    label_idx = next((i for i, name in enumerate(normalized) if name in LABEL_COLUMNS), None)

    return index_idx, ecut_idx, limit_idx, label_idx


def row_from_header(
    fields: list[str],
    index_idx: int,
    ecut_idx: int,
    limit_idx: int | None,
    label_idx: int | None,
) -> AGNPoint | None:
    if len(fields) <= max(index_idx, ecut_idx):
        return None

    try:
        spectral_index = parse_measurement(fields[index_idx])
        ecut = parse_measurement(fields[ecut_idx])
    except ValueError:
        return None

    is_upper_limit = (
        is_upper_limit_value(fields[limit_idx])
        if limit_idx is not None and limit_idx < len(fields)
        else spectral_index.is_limit or ecut.is_limit
    )
    label = fields[label_idx] if label_idx is not None and label_idx < len(fields) else ""
    return AGNPoint(
        spectral_index.value,
        ecut.value,
        is_upper_limit,
        spectral_index.err_low,
        spectral_index.err_high,
        ecut.err_low,
        ecut.err_high,
        label,
    )


def row_without_header(fields: list[str]) -> AGNPoint | None:
    numeric_indices = [i for i, field in enumerate(fields) if is_float(field)]
    if len(numeric_indices) < 2:
        return None

    index_idx, ecut_idx = numeric_indices[:2]
    is_upper_limit = any(is_upper_limit_value(field) for field in fields)
    try:
        spectral_index = parse_measurement(fields[index_idx])
        ecut = parse_measurement(fields[ecut_idx])
        return AGNPoint(
            spectral_index.value,
            ecut.value,
            is_upper_limit or spectral_index.is_limit or ecut.is_limit,
            spectral_index.err_low,
            spectral_index.err_high,
            ecut.err_low,
            ecut.err_high,
        )
    except ValueError:
        return None


def read_agn_points(input_path: Path) -> list[AGNPoint]:
    rows: list[AGNPoint] = []
    header: list[str] | None = None
    index_idx: int | None = None
    ecut_idx: int | None = None
    limit_idx: int | None = None
    label_idx: int | None = None

    with input_path.open() as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if not line or line.startswith(("#", "%", "---")):
                continue

            fields = split_line(line)
            if not fields:
                continue

            if header is None and not all(is_float(field) for field in fields[:2]):
                candidate = header_indices(fields)
                if candidate[0] is not None and candidate[1] is not None:
                    header = fields
                    index_idx, ecut_idx, limit_idx, label_idx = candidate
                    continue

            if header is not None and index_idx is not None and ecut_idx is not None:
                point = row_from_header(fields, index_idx, ecut_idx, limit_idx, label_idx)
            else:
                point = row_without_header(fields)

            if point is not None:
                rows.append(point)

    if not rows:
        raise ValueError(
            f"No spectral-index/Ecut rows could be read from {input_path}. "
            "Expected a table with columns such as Gamma and Ecut_keV, plus an "
            "optional status/limit column for upper limits."
        )
    return rows


def read_bat_catalog(catalog_path: Path) -> dict[str, BATSource]:
    sources: dict[str, BATSource] = {}
    with catalog_path.open(newline="") as catalog_file:
        reader = csv.DictReader(catalog_file)
        for row in reader:
            names = tuple(
                name.strip()
                for name in (
                    row.get("COUNTERPART_NAME", ""),
                    row.get("OTHER_NAME", ""),
                    row.get("Swift_BAT_name", ""),
                )
                if name.strip()
            )
            if not names:
                continue

            try:
                flux_bat = float(row["Flux_BAT"]) * 1.0e-12
                photon_index = float(row["index_BAT"])
            except (KeyError, ValueError):
                continue

            source = BATSource(names, flux_bat, photon_index)
            for name in names:
                sources[normalized_source_key(name)] = source
    return sources


def matched_bat_source(point: AGNPoint, bat_sources: dict[str, BATSource]) -> BATSource | None:
    point_key = normalized_source_key(point.label)
    if not point_key:
        return None

    if point_key in bat_sources:
        return bat_sources[point_key]

    for catalog_key, source in bat_sources.items():
        if point_key in catalog_key or catalog_key in point_key:
            return source
    return None


def special_label_key(source: BATSource) -> str | None:
    for name in source.names:
        key = normalized_source_key(name)
        if key in SPECIAL_SENSITIVITY_LABELS:
            return key
    return None


def cutoff_power_law_energy_flux_shape(
    energy_keV: float,
    photon_index: float,
    ecut_keV: float,
) -> float:
    return energy_keV ** (1.0 - photon_index) * math_exp_safe(-energy_keV / ecut_keV)


def math_exp_safe(value: float) -> float:
    if value < -745.0:
        return 0.0
    return math.exp(value)


def trapezoid_integral(y_values: list[float], x_values: list[float]) -> float:
    return sum(
        0.5 * (y_values[i] + y_values[i - 1]) * (x_values[i] - x_values[i - 1])
        for i in range(1, len(x_values))
    )


def logspace(start: float, stop: float, count: int) -> list[float]:
    log_start = math.log(start)
    log_stop = math.log(stop)
    return [
        math.exp(log_start + index * (log_stop - log_start) / (count - 1))
        for index in range(count)
    ]


def normalized_cutoff_power_law_spectrum(
    energy_grid_keV: list[float],
    photon_index: float,
    ecut_keV: float,
    bat_flux: float,
) -> list[float]:
    norm_grid = logspace(14.0, 195.0, 300)
    norm_shape = [
        cutoff_power_law_energy_flux_shape(energy, photon_index, ecut_keV)
        for energy in norm_grid
    ]
    integral = trapezoid_integral(norm_shape, norm_grid)
    if integral <= 0.0:
        return [0.0 for _energy in energy_grid_keV]

    normalization = bat_flux / integral
    return [
        energy * normalization * cutoff_power_law_energy_flux_shape(
            energy,
            photon_index,
            ecut_keV,
        )
        for energy in energy_grid_keV
    ]


def no_names_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_no_names{output_path.suffix}")


def annotate_spectrum(
    ax,
    label: str,
    energy_grid_keV: list[float],
    spectrum: list[float],
    band_keV: list[float],
    annotation_index: int,
):
    band_indices = [
        index
        for index, energy in enumerate(energy_grid_keV)
        if band_keV[0] <= energy <= band_keV[1]
    ]
    if not band_indices:
        return None

    peak_index = max(band_indices, key=lambda index: spectrum[index])
    x_stagger_values = [1.1, 1.8, 2.8]
    y_stagger_values = [1.10, 1.45, 1.95]
    stagger_index = annotation_index % len(x_stagger_values)
    label_x = min(
        band_keV[1],
        max(
            band_keV[0] * 1.05,
            energy_grid_keV[peak_index] * x_stagger_values[stagger_index],
        ),
    )
    label_y = spectrum[peak_index] * y_stagger_values[stagger_index]
    return ax.annotate(
        label,
        xy=(energy_grid_keV[peak_index], spectrum[peak_index]),
        xytext=(label_x, label_y),
        textcoords="data",
        fontsize=7,
        color="0.25",
        ha="left",
        arrowprops={
            "arrowstyle": "-",
            "color": "0.45",
            "linewidth": 0.5,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def plot_sensitivity_reference_lines(
    ax,
    cosi_energy_band_keV: list[float],
    cosi_2yr: float,
    cosi_5yr: float,
    include_labels: bool = True,
) -> None:
    bat_90_label = "Swift-BAT 105 mo, 90% sky" if include_labels else "_nolegend_"
    bat_50_label = "Swift-BAT 105 mo, 50% sky" if include_labels else "_nolegend_"
    cosi_2yr_label = "COSI 2 yr all-sky req." if include_labels else "_nolegend_"
    cosi_5yr_label = "COSI 5 yr all-sky" if include_labels else "_nolegend_"

    ax.hlines(
        8.40e-12,
        14.0,
        195.0,
        color="tab:purple",
        linewidth=4,
        label=bat_90_label,
    )
    ax.hlines(
        7.24e-12,
        14.0,
        195.0,
        color="tab:purple",
        alpha=0.45,
        linewidth=7,
        label=bat_50_label,
    )
    ax.hlines(
        cosi_2yr,
        cosi_energy_band_keV[0],
        cosi_energy_band_keV[1],
        color="tab:orange",
        linewidth=3,
        label=cosi_2yr_label,
    )
    ax.hlines(
        cosi_5yr,
        cosi_energy_band_keV[0],
        cosi_energy_band_keV[1],
        color="tab:orange",
        linestyle="--",
        linewidth=3,
        label=cosi_5yr_label,
    )


def format_sensitivity_axis(
    ax,
    shared_xlim: tuple[float, float],
    show_xlabel: bool,
) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*shared_xlim)
    ax.set_ylim(1.0e-12, 1.0e-9)
    ax.set_ylabel(r"Flux sens. [erg s$^{-1}$ cm$^{-2}$]")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if show_xlabel:
        ax.set_xlabel(r"Energy or $E_{\rm cut}$ [keV]")
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    ax.grid(alpha=0.2, which="both")


def detected_sources_dataframe(
    points: list[AGNPoint],
    bat_sources: dict[str, BATSource] | None = None,
    special_only: bool = False,
):

    if special_only:
        rows = []
        if bat_sources is None:
            return pd.DataFrame(rows, columns=["Name", "E_cut", "Spectral Index"])

        for special_key in sorted(SPECIAL_SENSITIVITY_LABELS):
            bat_source = bat_sources.get(special_key)
            if bat_source is None:
                continue

            matched_point = None
            for point in points:
                matched_source = matched_bat_source(point, bat_sources)
                if matched_source is not None and special_label_key(matched_source) == special_key:
                    matched_point = point
                    break

            if matched_point is not None:
                rows.append(
                    {
                        "Name": matched_point.label,
                        "E_cut": matched_point.ecut_keV,
                        "Spectral Index": matched_point.spectral_index,
                    }
                )
            else:
                rows.append(
                    {
                        "Name": bat_source.names[0],
                        "E_cut": float("nan"),
                        "Spectral Index": bat_source.photon_index,
                    }
                )

        data_frame = pd.DataFrame(rows, columns=["Name", "E_cut", "Spectral Index"])
        if not data_frame.empty:
            data_frame = data_frame.sort_values("Name").reset_index(drop=True)
        return data_frame

    rows = []
    seen_keys = set()
    for point in points:
        key = normalized_source_key(point.label)
        if key in seen_keys:
            continue

        rows.append(
            {
                "Name": point.label,
                "E_cut": point.ecut_keV,
                "Spectral Index": point.spectral_index,
            }
        )
        seen_keys.add(key)
    data_frame = pd.DataFrame(rows, columns=["Name", "E_cut", "Spectral Index"])
    if not data_frame.empty:
        data_frame = data_frame.sort_values("Name").reset_index(drop=True)
    return data_frame


def plot_points(
    points: list[AGNPoint],
    output_path: Path,
    show: bool,
    bat_catalog_path: Path,
) -> tuple[Path, Path]:
    plt = importlib.import_module("matplotlib.pyplot")
    detections = [point for point in points if not point.is_upper_limit]
    upper_limits = [point for point in points if point.is_upper_limit]
    shared_xlim = (10.0, 5000.0)
    bat_sources = read_bat_catalog(bat_catalog_path)
    cosi_energy_band_keV = [200.0, 5000.0]
    cosi_2yr_low = 2.0e-11
    cosi_2yr_high = 5.0e-11
    cosi_2yr = (cosi_2yr_low * cosi_2yr_high) ** 0.5
    cosi_5yr_scale = (2.0 / 5.0) ** 0.5
    cosi_5yr = cosi_2yr * cosi_5yr_scale
    cosi_band_detection_names = []

    fig, (ax, ax_sens, ax_special) = plt.subplots(
        3,
        1,
        figsize=(6.4, 9),
        gridspec_kw={"height_ratios": [2.0, 1.5, 1.5], "hspace": 0},
    )
    ax.axvspan(
        200.0,
        5000.0,
        color="tab:orange",
        alpha=0.18,
        label="COSI band (0.2-5 MeV)",
        zorder=0,
    )

    if detections:
        ax.errorbar(
            [point.ecut_keV for point in detections],
            [point.spectral_index for point in detections],
            xerr=[
                [point.ecut_err_low or 0.0 for point in detections],
                [point.ecut_err_high or 0.0 for point in detections],
            ],
            yerr=[
                [point.spectral_index_err_low or 0.0 for point in detections],
                [point.spectral_index_err_high or 0.0 for point in detections],
            ],
            fmt="o",
            markersize=5,
            color="tab:blue",
            markeredgecolor="black",
            markeredgewidth=0.4,
            ecolor="tab:blue",
            elinewidth=1.4,
            capsize=2,
            label="Swift/BAT Detections",
        )

    y_min, _y_max = ax.get_ylim()
    for point in upper_limits:
        ax.plot(
            [shared_xlim[0], point.ecut_keV],
            [point.spectral_index, point.spectral_index],
            color="0.65",
            linewidth=0.6,
            zorder=0,
        )
        ax.plot(
            [point.ecut_keV, point.ecut_keV],
            [y_min, point.spectral_index],
            color="0.65",
            linewidth=0.6,
            zorder=0,
        )

    if upper_limits:
        ax.plot([], [], color="0.65", linewidth=0.6, label="Upper limits")

    spectral_indices = [point.spectral_index for point in points]
    gamma_min = min(spectral_indices) - 0.1
    gamma_max = max(spectral_indices) + 0.1
    gamma_grid = [
        gamma_min + index * (gamma_max - gamma_min) / 299.0
        for index in range(300)
    ]
    tau_styles = {
        1.0: {"color": "black", "linestyle": "-", "alpha": 0.8},
        2.0: {"color": "black", "linestyle": "--", "alpha": 0.75},
        3.0: {"color": "black", "linestyle": ":", "alpha": 0.75},
    }
    for tau, style in tau_styles.items():
        curve_points = [
            (ecut, spectral_index)
            for spectral_index in gamma_grid
            if (ecut := ecut_contour_for_tau(spectral_index, tau)) is not None
        ]
        if not curve_points:
            continue

        # ax.plot(
        #     [point[0] for point in curve_points],
        #     [point[1] for point in curve_points],
        #     linewidth=1.3,
        #     label=rf"$\tau={tau:g}$",
        #     **style,
        # )

    ax.set_xlabel("")
    ax.set_ylabel("Spectral index")
    ax.set_title("105-month Swift-BAT AGN: High-Energy Cutoff vs. Spectral Index")
    ax.set_xscale("log")
    ax.set_xlim(*shared_xlim)
    ax.set_ylim(0.75, 2.25)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)

    energy_grid_keV = logspace(shared_xlim[0], shared_xlim[1], 300)
    matched_spectra = 0
    annotated_spectra = 0
    name_annotations = []
    for point in detections:
        bat_source = matched_bat_source(point, bat_sources)
        if bat_source is None:
            continue

        spectrum = normalized_cutoff_power_law_spectrum(
            energy_grid_keV,
            point.spectral_index,
            point.ecut_keV,
            bat_source.flux_bat,
        )
        ax_sens.plot(
            energy_grid_keV,
            spectrum,
            color="0.45",
            alpha=0.35,
            linewidth=0.6,
            zorder=1,
        )
        matched_spectra += 1

        cosi_band_values = [
            spectrum[index]
            for index, energy in enumerate(energy_grid_keV)
            if cosi_energy_band_keV[0] <= energy <= cosi_energy_band_keV[1]
        ]
        if not cosi_band_values or max(cosi_band_values) < cosi_2yr:
            continue

        label = bat_source.names[0] if bat_source.names else point.label
        annotation = annotate_spectrum(
           ax_sens,
           label,
           energy_grid_keV,
           spectrum,
           cosi_energy_band_keV,
           annotated_spectra,
        )
        if annotation is not None:
            name_annotations.append(annotation)
            annotated_spectra += 1
    if matched_spectra:
        ax_sens.plot(
            [],
            [],
            color="0.45",
            alpha=0.45,
            linewidth=0.8,
            label="Detected BAT PL+Ecut spectra",
        )
    
    
    # Stores a list of all sources that are above COSI detection threshold in the 0.2-5 MeV band.
    for ann in name_annotations:
        normalized_label = normalized_source_key(ann.get_text())
        cosi_band_detection_names.append(normalized_label)
    
    plot_sensitivity_reference_lines(ax_sens, cosi_energy_band_keV, cosi_2yr, cosi_5yr)
    format_sensitivity_axis(ax_sens, shared_xlim, show_xlabel=False)
    ax_sens.legend(loc="upper left", fontsize=9)

    special_linestyles = {
        "circinusgalaxy": "-",
        "ic4329a": "--",
        "ngc1068": "-.",
        "ngc4151": ":",
    }
    for special_key in sorted(SPECIAL_SENSITIVITY_LABELS):
        bat_source = bat_sources.get(special_key)
        if bat_source is None:
            continue

        spectrum = normalized_cutoff_power_law_spectrum(
            energy_grid_keV,
            bat_source.photon_index,
            1.0e12,
            bat_source.flux_bat,
        )
        label = bat_source.names[0]
        ax_special.plot(
            energy_grid_keV,
            spectrum,
            color="0.35",
            linestyle=special_linestyles.get(special_key, "-"),
            linewidth=1.6,
            label=label,
        )

    plot_sensitivity_reference_lines(
        ax_special,
        cosi_energy_band_keV,
        cosi_2yr,
        cosi_5yr,
        include_labels=False,
    )
    format_sensitivity_axis(ax_special, shared_xlim, show_xlabel=True)
    ax_special.legend(loc="upper left", ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    output_without_names = no_names_output_path(output_path)
    # for annotation in name_annotations:
    #     annotation.remove()
    fig.savefig(output_without_names, dpi=200)
    if show:
        plt.show()
    return output_without_names, cosi_band_detection_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AGN spectral index versus high-energy cutoff."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "AGN_Index_ecut_2020.txt",
        help="Input text table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "AGN_Index_ecut_scatter.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--bat-catalog",
        type=Path,
        default=SCRIPT_DIR / "Swift_BAT_105mo_catalog.csv",
        help="Swift-BAT 105-month catalog CSV used to normalize matched spectra.",
    )
    parser.add_argument(
        "--show",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Display the plot interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = read_agn_points(args.input)
    output_without_names, cosi_band_detection_names = plot_points(
    points, args.output, args.show, args.bat_catalog
    )
    # Makes a dataframe of all sources, then filters to only COSI detected sources if DETECTED_ONLY is True.
    all_points_df = pd.DataFrame([asdict(point) for point in points])
    if DETECTED_ONLY:
        all_points_df = all_points_df[all_points_df['label'].apply(normalized_source_key).isin(cosi_band_detection_names)]
        all_points_csv_path = args.output.with_name(f"{args.output.stem}_detected_points.csv")
    else:
        all_points_csv_path = args.output.with_name(f"{args.output.stem}_all_points.csv")
    all_points_df.to_csv(all_points_csv_path, index=False)
    print(f"Read {len(points)} AGN rows from {args.input}")
    print(f"Wrote {all_points_csv_path}")
    print(f"Wrote {args.output}")
    print(f"Wrote {output_without_names}")


if __name__ == "__main__":
    main()
