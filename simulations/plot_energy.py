"""Collect annual sensible heating demand and draw the conference figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
CONVERSION = ROOT.parents[1] / "1_citygml/citygml-epjson"
SCENARIOS = {
    "2023": "Present scenario",
    "ssp126_2055": "2050 low emissions scenario",
    "ssp370_2055": "2050 high emissions scenario",
}
WRAPPED_LABELS = {
    "2023": "Present scenario",
    "ssp126_2055": "2050 low emissions\nscenario",
    "ssp370_2055": "2050 high emissions\nscenario",
}
HEATING_VARIABLE = "Zone Air System Sensible Heating Energy"
JOULES_PER_KWH = 3_600_000.0
HOURS_PER_YEAR = 8760
REPORT_SCHEMA = 1
ID_PATTERN = re.compile(
    r"^(ID_[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12})(?:_|$)", re.I
)
INK, UNAVAILABLE = "#252729", "#DFDFDF"
# Present-scenario blue follows the conference methodology figure.
SCENARIO_COLOURS = ("#1554FF", "#B193D5", "#59318C")
SCENARIO_LINESTYLES = ("-", "--", "-.")
COMPARISON_FIGSIZE = (6.0, 2.0)
DISTRIBUTION_MAX = 400.0
MAP_CMAP = "RdYlGn_r"
FIGURE_NAMES = ("annual_heating_intensity_maps", "heating_demand_comparison")


def building_id(name: str) -> str:
    match = ID_PATTERN.match(name.strip())
    if match is None:
        raise ValueError(f"Provide a building identifier in {name!r}.")
    return match[1].upper()


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def input_record(path: Path) -> dict:
    path = path.resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    before = path.stat()
    checksum = digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Use a stable input file: {path}.")
    return {"path": str(path), "exists": True, "bytes": after.st_size,
            "sha256": checksum}


def read_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["gml_id", "floor_area", "number_of_floors"])
    frame["building_id"] = frame.gml_id.map(building_id)
    if frame.building_id.duplicated().any():
        raise ValueError("Provide one floor-area record per building identifier.")
    area = pd.to_numeric(frame.floor_area, errors="coerce")
    if (~np.isfinite(area) | (area <= 0)).any():
        raise ValueError("Provide a finite positive total floor area for every building.")
    return frame.assign(total_floor_area_m2=area).set_index("building_id")


def discover_blocks(simulations: Path) -> list[str]:
    pattern = re.compile(r"(?:2023|ssp126_2055|ssp370_2055)_(block_\d+)$")
    blocks = {match[1] for p in simulations.iterdir() if p.is_dir()
              and (match := pattern.fullmatch(p.name))}
    if not blocks:
        raise ValueError(f"Provide scenario output directories in {simulations}.")
    return sorted(blocks, key=lambda block: int(block.rsplit("_", 1)[1]))


def read_geometry(models: Path, blocks: list[str]) -> tuple[dict, dict, list[dict]]:
    """Read model footprints and retain their zone and surface ownership."""
    buildings, block_geometry, sources = {}, {}, []
    for block in blocks:
        path = models / f"{block}.epJSON"
        model = json.loads(path.read_text())
        sources.append(input_record(path))
        if any(b.get("north_axis", 0) != 0 for b in model.get("Building", {}).values()):
            raise ValueError(f"Provide north-aligned model coordinates in {path}.")
        for zone in model["Zone"].values():
            if any(zone.get(key, 0) != 0 for key in (
                "direction_of_relative_north", "x_origin", "y_origin", "z_origin"
            )):
                raise ValueError(f"Provide zero zone origins and rotations in {path}.")
        zones = {name.upper(): building_id(name) for name in model["Zone"]}
        parts = {identifier: [] for identifier in zones.values()}
        floors = {}
        for name, surface in model["BuildingSurface:Detailed"].items():
            if (surface["surface_type"].lower() != "floor"
                    or surface["outside_boundary_condition"].lower() != "ground"):
                continue
            vertices = surface["vertices"]
            polygon = Polygon([(v["vertex_x_coordinate"], v["vertex_y_coordinate"])
                               for v in vertices])
            if not polygon.is_valid or polygon.area <= 0:
                raise ValueError(f"Provide a valid ground-floor polygon: {name}.")
            identifier = zones[surface["zone_name"].upper()]
            parts[identifier].append(polygon)
            xyz = np.array([(v["vertex_x_coordinate"], v["vertex_y_coordinate"],
                             v["vertex_z_coordinate"]) for v in vertices], dtype=float)
            xyz -= xyz.mean(axis=0)
            # Simulation areas include slope; the map uses the XY projection.
            floors[name.upper()] = float(np.linalg.norm(
                np.cross(xyz, np.roll(xyz, -1, axis=0)).sum(axis=0)) / 2)
        for identifier, polygons in parts.items():
            if identifier in buildings:
                raise ValueError(f"Assign {identifier} to one simulation block.")
            if not polygons:
                raise ValueError(f"Provide a footprint for {identifier}.")
            footprint = unary_union(polygons)
            if not footprint.is_valid or footprint.geom_type not in ("Polygon", "MultiPolygon"):
                raise ValueError(f"Provide a valid footprint for {identifier}.")
            buildings[identifier] = {"block": block, "footprint": footprint}
        block_geometry[block] = {"zones": zones, "floors": floors}
    return buildings, block_geometry, sources


def annual_calendar_valid(rows: list[tuple]) -> bool:
    """Check all hourly intervals in one complete non-leap weather year."""
    if len(rows) != HOURS_PER_YEAR:
        return False
    expected = {(day.month, day.day, hour, 0, 60)
                for offset in range(365)
                for day in [date(2017, 1, 1) + timedelta(days=offset)]
                for hour in range(1, 25)}
    # Columns: TimeIndex, Year, Month, Day, Hour, Minute, Interval.
    return (len({r[0] for r in rows}) == HOURS_PER_YEAR
            and len({r[1] for r in rows}) == 1
            and {tuple(r[2:]) for r in rows} == expected)


def read_block(run: Path, geometry: dict) -> tuple[list[dict], dict]:
    """Validate completion and aggregate hourly sensible heating per zone."""
    expected_zones = geometry["zones"]
    zone_rows = [{"zone": name, "building_id": identifier, "complete": False,
                  "heating_kwh": np.nan, "valid_hours": 0, "reason": "missing output"}
                 for name, identifier in expected_zones.items()]
    status = {"run": run.name, "status": "incomplete", "zones": len(zone_rows),
              "complete_zones": 0, "calendar_year": None, "reason": "missing output"}
    sql, end = run / "eplusout.sql", run / "eplusout.end"
    if not sql.exists() or not end.exists():
        return zone_rows, status
    if not re.search(r"Completed Successfully.*\b0 Severe Errors", end.read_text()):
        status["reason"] = "unsuccessful completion record"
        for row in zone_rows:
            row["reason"] = status["reason"]
        return zone_rows, status
    with closing(sqlite3.connect(sql.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        sql_zones = conn.execute("SELECT ZoneName, Multiplier, ListMultiplier FROM Zones").fetchall()
        if {r[0].upper() for r in sql_zones} != set(expected_zones):
            raise ValueError(f"Match model zones to simulation zones in {run}.")
        if len(sql_zones) != len(expected_zones):
            raise ValueError(f"Provide unique simulation zone names in {run}.")
        if any(r[1:] != (1.0, 1.0) for r in sql_zones):
            raise ValueError(f"Provide unit zone multipliers in {run}.")
        floors = dict(conn.execute("SELECT SurfaceName, GrossArea FROM Surfaces WHERE ClassName='Floor'"))
        for name, area in geometry["floors"].items():
            if name not in floors or not math.isclose(area, floors[name], rel_tol=1e-8, abs_tol=1e-6):
                raise ValueError(f"Match model ground-floor surface {name} to {run}.")
        environments = conn.execute(
            "SELECT EnvironmentPeriodIndex FROM EnvironmentPeriods WHERE EnvironmentType=3"
        ).fetchall()
        if len(environments) != 1:
            raise ValueError(f"Provide one weather simulation environment in {run}.")
        environment = environments[0][0]
        times = conn.execute(
            'SELECT TimeIndex, Year, Month, Day, Hour, Minute, "Interval" FROM Time '
            'WHERE EnvironmentPeriodIndex=? AND COALESCE(WarmupFlag,0)=0',
            (environment,),
        ).fetchall()
        valid_calendar = annual_calendar_valid(times)
        status["calendar_year"] = times[0][1] if times else None
        dictionary = conn.execute(
            "SELECT KeyValue, Units, Type FROM ReportDataDictionary "
            "WHERE Name=? AND ReportingFrequency='Hourly' AND IsMeter=0",
            (HEATING_VARIABLE,),
        ).fetchall()
        names = [r[0].upper() for r in dictionary]
        if len(names) != len(set(names)) or not set(names) <= set(expected_zones):
            raise ValueError(f"Provide unique heating outputs for the model zones in {run}.")
        if any(r[1:] != ("J", "Sum") for r in dictionary):
            raise ValueError(f"Provide summed hourly heating energy in joules in {run}.")
        energy = conn.execute(
            'SELECT d.KeyValue, COUNT(*), COUNT(r.Value), COUNT(DISTINCT r.TimeIndex), '
            'SUM(r.Value), MIN(r.Value), MAX(r.Value) '
            'FROM ReportData r JOIN ReportDataDictionary d '
            'ON d.ReportDataDictionaryIndex=r.ReportDataDictionaryIndex '
            'JOIN Time t ON t.TimeIndex=r.TimeIndex '
            'WHERE d.Name=? AND d.ReportingFrequency=\'Hourly\' AND d.IsMeter=0 '
            'AND t.EnvironmentPeriodIndex=? AND COALESCE(t.WarmupFlag,0)=0 '
            'GROUP BY d.ReportDataDictionaryIndex', (HEATING_VARIABLE, environment),
        ).fetchall()
    energy = {r[0].upper(): r[1:] for r in energy}
    for row in zone_rows:
        values = energy.get(row["zone"])
        if values is None:
            row["reason"] = "missing heating variable"
            continue
        count, finite_count, distinct_count, joules, minimum, maximum = values
        row["valid_hours"] = finite_count
        complete = (valid_calendar and count == finite_count == distinct_count == HOURS_PER_YEAR
                    and all(v is not None and np.isfinite(v) for v in (joules, minimum, maximum))
                    and minimum >= 0)
        row.update(complete=bool(complete),
                   heating_kwh=joules / JOULES_PER_KWH if complete else np.nan,
                   reason="" if complete else "incomplete or invalid annual heating records")
    status["complete_zones"] = sum(r["complete"] for r in zone_rows)
    status["status"] = "complete" if status["complete_zones"] == len(zone_rows) else "incomplete"
    status["reason"] = "" if status["status"] == "complete" else "; ".join(
        sorted({r["reason"] for r in zone_rows if r["reason"]}))
    return zone_rows, status


def collect(simulations: Path, block_geometry: dict, metrics: pd.DataFrame) -> tuple:
    rows, status_rows, sources = [], [], []
    jobs = [(scenario, block, geometry) for scenario in SCENARIOS
            for block, geometry in block_geometry.items()]

    def read_run(job):
        scenario, block, geometry = job
        run = simulations / f"{scenario}_{block}"
        zone_rows, status = read_block(run, geometry)
        inputs = [input_record(run / name) for name in ("eplusout.sql", "eplusout.end")]
        return scenario, block, zone_rows, status, inputs

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (scenario, block, zone_rows, status, inputs) in enumerate(pool.map(read_run, jobs)):
            label = SCENARIOS[scenario]
            status_rows.append({"scenario": scenario, "scenario_label": label, "block": block, **status})
            sources.extend(inputs)
            frame = pd.DataFrame(zone_rows)
            for identifier, group in frame.groupby("building_id", sort=True):
                if identifier not in metrics.index:
                    raise ValueError(f"Provide total floor area for {identifier}.")
                area = float(metrics.loc[identifier, "total_floor_area_m2"])
                complete = bool(group.complete.all())
                demand = float(group.heating_kwh.sum()) if complete else np.nan
                rows.append({
                    "scenario": scenario, "scenario_label": label, "block": block,
                    "building_id": identifier, "total_floor_area_m2": area,
                    "zones": len(group), "minimum_valid_hours": int(group.valid_hours.min()),
                    "complete": complete, "heating_kwh": demand,
                    "heating_intensity_kwh_m2": demand / area,
                    "reason": "; ".join(sorted(set(group.reason) - {""})),
                })
            completed = index % len(block_geometry) + 1
            if completed % 10 == 0 or completed == len(block_geometry):
                print(f"Collected {scenario}: {completed}/{len(block_geometry)} blocks.", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(status_rows), sources


def summarise(buildings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if buildings.duplicated(["scenario", "building_id"]).any():
        raise ValueError("Provide one result per building and scenario.")
    sets = [set(buildings.loc[(buildings.scenario == s) & buildings.complete, "building_id"])
            for s in SCENARIOS]
    common = set.intersection(*sets)
    if not common:
        raise ValueError("Provide complete buildings shared by all three scenarios.")
    buildings = buildings.assign(in_common_comparison=buildings.building_id.isin(common))
    summary = []
    for scenario, label in SCENARIOS.items():
        selected = buildings.loc[(buildings.scenario == scenario) & buildings.in_common_comparison]
        total = float(selected.heating_kwh.sum())
        summary.append({"scenario": scenario, "scenario_label": label,
                        "buildings": len(selected),
                        "excluded_buildings": int((buildings.scenario == scenario).sum() - len(selected)),
                        "total_floor_area_m2": float(selected.total_floor_area_m2.sum()),
                        "heating_kwh": total, "heating_gwh": total / 1e6})
    summary = pd.DataFrame(summary)
    baseline = summary.heating_kwh.iloc[0]
    summary["change_from_present_pct"] = (
        100 * (summary.heating_kwh - baseline) / baseline if baseline > 0 else np.nan)
    return buildings, summary


def crosscheck_csv(simulations: Path, buildings: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Check one complete block per scenario against the independent CSV export."""
    checks, sources = [], []
    for scenario in SCENARIOS:
        selected = buildings.loc[buildings.scenario == scenario]
        complete = selected.groupby("block").complete.all()
        blocks = sorted(complete.index[complete], key=lambda b: int(b.rsplit("_", 1)[1]))
        if not blocks:
            raise ValueError(f"Provide a complete block for the CSV check of {scenario}.")
        block = blocks[0]
        path = simulations / f"{scenario}_{block}" / "eplusout.csv"
        suffix = f":{HEATING_VARIABLE} [J](Hourly)"
        hourly = pd.read_csv(path, usecols=lambda c: c.strip().endswith(suffix))
        if len(hourly) != HOURS_PER_YEAR or not len(hourly.columns):
            raise ValueError(f"Provide annual hourly CSV heating records in {path}.")
        if not np.isfinite(hourly.to_numpy()).all() or (hourly.to_numpy() < 0).any():
            raise ValueError(f"Provide finite non-negative CSV heating records in {path}.")
        expected = selected.loc[selected.block == block].set_index("building_id").heating_kwh
        actual = hourly.sum() / JOULES_PER_KWH
        actual.index = actual.index.map(lambda name: building_id(name.split(":", 1)[0]))
        actual = actual.groupby(level=0).sum()
        if set(actual.index) != set(expected.index):
            raise ValueError(f"Match CSV building identifiers in {path}.")
        actual = actual.reindex(expected.index)
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-3):
            raise ValueError(f"Reconcile SQLite and CSV heating demand in {path}.")
        checks.append({"scenario": scenario, "block": block, "buildings": len(actual),
                       "sql_heating_kwh": float(expected.sum()),
                       "csv_heating_kwh": float(actual.sum()),
                       "maximum_building_difference_kwh": float((actual - expected).abs().max()),
                       "passed": True})
        sources.append(input_record(path))
    return pd.DataFrame(checks), sources


def polygon_patch(polygon: Polygon) -> PathPatch:
    polygon = orient(polygon, sign=1.0)
    vertices, codes = [], []
    for ring in [polygon.exterior, *polygon.interiors]:
        coordinates = list(ring.coords)
        vertices.extend(coordinates)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(coordinates) - 2)
                     + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(vertices, codes))


def colour_limits(buildings: pd.DataFrame) -> tuple[float, float]:
    values = buildings.loc[buildings.complete, "heating_intensity_kwh_m2"].dropna()
    if values.empty:
        raise ValueError("Provide complete heating demand intensities for the maps.")
    return 0.0, max(25.0, math.ceil(float(values.quantile(.98)) / 25) * 25.0)


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#A8ADB2",
        "axes.linewidth": .5, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.size": 2.5, "ytick.major.size": 0,
        "hatch.linewidth": .35,
    })


def draw_building_maps(buildings: pd.DataFrame, geometry: dict, *, value_column: str,
                       limits: tuple[float, float], colourbar_label: str,
                       norm=None, cmap=None, colourbar_options: dict | None = None) -> tuple:
    """Draw three building maps with shared geometry, labels, and colour limits."""
    norm = Normalize(*limits, clip=True) if norm is None else norm
    cmap = plt.get_cmap(MAP_CMAP) if cmap is None else cmap
    identifiers, patches = [], []
    for identifier, record in sorted(geometry.items()):
        footprint = record["footprint"]
        polygons = [footprint] if footprint.geom_type == "Polygon" else footprint.geoms
        for polygon in polygons:
            identifiers.append(identifier)
            patches.append(polygon_patch(polygon))
    bounds = np.array([v["footprint"].bounds for v in geometry.values()])
    left, bottom = bounds[:, :2].min(axis=0)
    right, top = bounds[:, 2:].max(axis=0)
    span = max(right - left, top - bottom)
    centre = ((left + right) / 2, (bottom + top) / 2)
    extent = (centre[0] - .55 * span, centre[0] + .55 * span,
              centre[1] - .55 * span, centre[1] + .55 * span)
    fig = plt.figure(figsize=(12.2 / 2.54, 6.4 / 2.54))
    clipped, unavailable = {}, {}
    for i, (scenario, label) in enumerate(SCENARIOS.items()):
        ax = fig.add_axes([.01 + i * .33, .28, .32, .57])
        frame = buildings.loc[buildings.scenario == scenario].set_index("building_id")
        values = frame[value_column].where(frame.complete).reindex(identifiers).to_numpy()
        available = np.isfinite(values)
        ax.add_collection(PatchCollection(
            [patch for patch, valid in zip(patches, available) if valid],
            facecolors=cmap(norm(values[available])), edgecolors=INK, linewidths=.12))
        if (~available).any():
            ax.add_collection(PatchCollection(
                [patch for patch, valid in zip(patches, available) if not valid],
                facecolors=UNAVAILABLE, edgecolors=INK, linewidths=.12, hatch="///"))
        ax.set(xlim=extent[:2], ylim=extent[2:], aspect="equal")
        ax.set_axis_off()
        fig.text(.17 + i * .33, .97, f"({chr(97+i)}) {WRAPPED_LABELS[scenario]}",
                 ha="center", va="top", fontsize=8)
        clipped[scenario] = int((frame.loc[frame.complete, value_column] > limits[1]).sum())
        unavailable[scenario] = int((~frame.complete).sum())
        if i == 0:
            length = 200.0
            x, y = extent[0] + .07 * span, extent[2] + .06 * span
            ax.plot([x, x + length], [y, y], color=INK, lw=1)
            ax.text(x + length / 2, y + .025 * span, "200 m", ha="center", fontsize=6)
            ax.annotate("N", xy=(.92, .85), xytext=(.92, .98), xycoords="axes fraction",
                        ha="center", va="center", fontsize=7,
                        arrowprops={"arrowstyle": "<-", "lw": .65, "color": INK})
    bar_ax = fig.add_axes([.15, .15, .70, .04])
    options = {"orientation": "horizontal", "extend": "max" if any(clipped.values()) else "neither"}
    options.update(colourbar_options or {})
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar_ax, **options)
    bar.set_label(colourbar_label, labelpad=3)
    if "ticks" not in options:
        bar.locator = MaxNLocator(nbins=6)
        bar.update_ticks()
    fig.text(.02, .235, f"{len(geometry):,} buildings", fontsize=7, va="center")
    fig.legend(handles=[Patch(facecolor=UNAVAILABLE, edgecolor=INK, hatch="///", label="Unavailable")],
               loc="center right", bbox_to_anchor=(.99, .235), fontsize=7, frameon=False)
    return fig, {"colour_limits": list(limits),
                 "clipped_buildings": clipped, "unavailable_buildings": unavailable,
                 "map_extent_m": list(extent), "width_cm": 12.2, "height_cm": 6.4}


def draw_maps(buildings: pd.DataFrame, geometry: dict) -> tuple:
    fig, settings = draw_building_maps(
        buildings, geometry, value_column="heating_intensity_kwh_m2",
        limits=colour_limits(buildings),
        colourbar_label="Annual heating demand intensity (kWh/m²)")
    settings["colour_limits_kwh_m2"] = settings.pop("colour_limits")
    settings["upper_limit_rule"] = "pooled p98 rounded up to 25 kWh/m2"
    return fig, settings


def draw_totals(ax: plt.Axes, summary: pd.DataFrame) -> None:
    frame = summary.set_index("scenario").loc[list(SCENARIOS)]
    values = frame.heating_gwh.to_numpy()
    bars = ax.bar(np.arange(3), values, width=.62, color=SCENARIO_COLOURS,
                  edgecolor=INK, linewidth=.45)
    for bar, hatch in zip(bars, ("", "///", "...")):
        bar.set_hatch(hatch)
    largest = max(float(values.max()), 1.0)
    for i, (value, change) in enumerate(zip(values, frame.change_from_present_pct)):
        annotation = f"{value:.1f}"
        if i and np.isfinite(change):
            annotation += f"\n{change:+.1f}%"
        ax.annotate(annotation, (i, value), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(3), ["Present", "Low", "High"])
    ax.set(xlim=(-.6, 2.6), ylim=(0, largest * 1.38), ylabel="Heating demand (GWh)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E9EBEC", linewidth=.5)
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.5)


def heating_distribution(buildings: pd.DataFrame, bin_width: float = 25.0) -> tuple:
    """Use shared bins and one smoothing bandwidth for the common building stock."""
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("Provide a finite positive bin width.")
    selected = buildings.loc[buildings.complete & buildings.in_common_comparison]
    groups = [selected.loc[selected.scenario == s].set_index("building_id") for s in SCENARIOS]
    if not len(groups[0]) or any(set(g.index) != set(groups[0].index) or g.index.has_duplicates for g in groups):
        raise ValueError("Use the same complete buildings in all three distributions.")
    samples = [g.heating_intensity_kwh_m2.to_numpy() for g in groups]
    pooled = np.concatenate(samples)
    if not np.isfinite(pooled).all() or (pooled < 0).any():
        raise ValueError("Provide finite non-negative heating demand intensities.")
    n = len(samples[0])
    upper = max(bin_width, math.ceil(float(pooled.max()) / bin_width) * bin_width)
    edges = np.arange(round(upper / bin_width) + 1) * bin_width
    spread = min(float(pooled.std(ddof=1)), float(np.subtract(*np.percentile(pooled, [75, 25]))) / 1.34)
    bandwidth = .9 * spread * n ** (-.2) if spread > 0 else bin_width / 2
    grid = np.linspace(0, upper + 5 * bandwidth, math.ceil((upper + 5 * bandwidth) / 2) + 1)
    bins, curves, tails, medians = [], [], {}, {}
    for scenario, values in zip(SCENARIOS, samples):
        counts, _ = np.histogram(values, edges)
        density = counts / (n * bin_width)
        bins.extend({"scenario": scenario, "scenario_label": SCENARIOS[scenario],
                     "bin_lower_kwh_m2": lo, "bin_upper_kwh_m2": hi,
                     "building_count": int(count), "building_share_pct": 100 * count / n,
                     "density": value}
                    for lo, hi, count, value in zip(edges[:-1], edges[1:], counts, density))
        # Reflect the Gaussian kernel at zero to retain the non-negative domain.
        offsets = (grid[:, None] - values[None, :]) / bandwidth
        reflected = (grid[:, None] + values[None, :]) / bandwidth
        smooth = (np.exp(-.5 * offsets ** 2) + np.exp(-.5 * reflected ** 2)).mean(axis=1)
        smooth /= bandwidth * math.sqrt(2 * math.pi)
        curves.extend({"scenario": scenario, "heating_intensity_kwh_m2": x, "density": y,
                       "smoothed_building_count": y * n * bin_width}
                      for x, y in zip(grid, smooth))
        tails[scenario] = int((values > DISTRIBUTION_MAX).sum())
        medians[scenario] = float(np.median(values))
    settings = {"bin_width_kwh_m2": bin_width, "bandwidth_kwh_m2": bandwidth,
                "bandwidth_rule": "0.9 * min(pooled SD, pooled IQR / 1.34) * buildings_per_scenario ** (-0.2)",
                "zero_spread_bandwidth": "half the bin width",
                "kernel": "Gaussian, reflected at zero", "buildings_per_scenario": n,
                "display_range_kwh_m2": [0, DISTRIBUTION_MAX], "buildings_above_display_range": tails,
                "median_heating_eui_kwh_m2": medians,
                "vertical_axis": "Number of buildings",
                "curve_scale": "density * buildings_per_scenario * bin_width_kwh_m2",
                "scenario_colours": dict(zip(SCENARIOS, SCENARIO_COLOURS)),
                "scenario_linestyles": dict(zip(SCENARIOS, SCENARIO_LINESTYLES)),
                "panel_width_ratios": [3, 1],
                "width_cm": COMPARISON_FIGSIZE[0] * 2.54,
                "height_cm": COMPARISON_FIGSIZE[1] * 2.54, "aspect_ratio": "3:1"}
    return pd.DataFrame(bins), pd.DataFrame(curves), settings


def draw_distribution(ax: plt.Axes, bins: pd.DataFrame, curves: pd.DataFrame, settings: dict) -> None:
    visible_peak = 0.0
    lower, upper = settings["display_range_kwh_m2"]
    for scenario, colour, style in zip(SCENARIOS, SCENARIO_COLOURS, SCENARIO_LINESTYLES):
        group = bins.loc[bins.scenario == scenario]
        edges = np.r_[group.bin_lower_kwh_m2.to_numpy(), group.bin_upper_kwh_m2.iloc[-1]]
        ax.stairs(group.building_count, edges, color=colour, linestyle=style,
                  alpha=.35, linewidth=.6, baseline=None)
        curve = curves.loc[curves.scenario == scenario]
        median = settings["median_heating_eui_kwh_m2"][scenario]
        ax.plot(curve.heating_intensity_kwh_m2, curve.smoothed_building_count,
                color=colour, linestyle=style, linewidth=1.25,
                label=f"{SCENARIOS[scenario]}\nMedian: {median:.1f} kWh/m²")
        ax.axvline(median, color=colour, linestyle=style, linewidth=.7, alpha=.75)
        visible_bins = group.loc[(group.bin_lower_kwh_m2 < upper) & (group.bin_upper_kwh_m2 > lower)]
        visible_curve = curve.loc[curve.heating_intensity_kwh_m2.between(lower, upper)]
        visible_peak = max(visible_peak, float(visible_bins.building_count.max()),
                           float(visible_curve.smoothed_building_count.max()))
    ax.set(xlim=(lower, upper), ylim=(0, max(visible_peak, 1) * 1.15),
           xlabel="Annual heating EUI (kWh/m²)", ylabel="Number of buildings")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E9EBEC", linewidth=.5)
    ax.set_axisbelow(True)
    ax.text(.98, .95, f"{settings['buildings_per_scenario']:,} buildings per scenario\n"
            f"{settings['bin_width_kwh_m2']:g} kWh/m² bins", transform=ax.transAxes,
            ha="right", va="top", fontsize=6)


def draw_comparison(summary: pd.DataFrame, bins: pd.DataFrame,
                    curves: pd.DataFrame, settings: dict) -> plt.Figure:
    """Combine building counts and aggregated heating demand in a 3:1 canvas."""
    fig, (distribution_ax, totals_ax) = plt.subplots(
        1, 2, figsize=COMPARISON_FIGSIZE, gridspec_kw={
            "width_ratios": [3, 1], "left": .09, "right": .985,
            "bottom": .25, "top": .75, "wspace": .34})
    draw_distribution(distribution_ax, bins, curves, settings)
    draw_totals(totals_ax, summary)
    distribution_ax.set_title("(a) Heating EUI distribution", loc="left", fontsize=7.5)
    totals_ax.set_title("(b) Aggregated demand", loc="left", fontsize=7.5)
    handles, labels = distribution_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1.0),
               ncol=3, fontsize=6.5, frameon=False, handlelength=2.5, columnspacing=1.5)
    return fig


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> list[Path]:
    output = path.with_suffix(".png")
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    return [output]


def write_captions(reports: Path, summary: pd.DataFrame, settings: dict, distribution: dict) -> None:
    counts = ", ".join(str(settings["clipped_buildings"][s]) for s in SCENARIOS)
    upper = settings["colour_limits_kwh_m2"][1]
    n = int(summary.buildings.iloc[0])
    text = f"""# Draft captions and alt text

## Figure 1

Caption: Annual heating demand intensity for the Present scenario, 2050 low emissions scenario, and 2050 high emissions scenario.
Sensible heating demand was divided by total modelled floor area across all storeys.
The shared colour scale extends from 0 to {upper:g} kWh/m².
Values above this limit occur in {counts} buildings, respectively.
Each map retains every available building footprint.
Hatched grey footprints indicate unavailable results.

Alt text: Three maps show building footprints coloured by annual heating demand intensity.
The maps share one extent and one horizontal colour bar.
Green indicates lower heating demand intensity, while red indicates higher heating demand intensity.

## Figure 2

Caption: Heating EUI distributions and aggregated annual heating demand across the same {n:,} buildings in the three scenarios.
The distribution occupies 75% of the combined panel width, and vertical bars occupy 25%.
Histogram outlines use shared {distribution['bin_width_kwh_m2']:g} kWh/m² bins.
Curves show Gaussian kernel density estimates with a shared bandwidth of {distribution['bandwidth_kwh_m2']:.2f} kWh/m².
The kernels are reflected at zero.
Each density curve is multiplied by the building count and bin width to express smoothed building counts per bin.
Vertical lines indicate scenario medians calculated from all included buildings.
The distribution displays 0–400 kWh/m².
The distribution reports retain all bins and the complete curve range.
The bars include all {n:,} buildings with complete results in every scenario.
Bar annotations show annual heating demand in GWh and percentage changes relative to the Present scenario.

Alt text: The left panel shows building counts by heating EUI, smooth curves, and vertical median lines.
The right panel shows three vertical bars of aggregated annual heating demand.
Blue identifies the Present scenario, light purple identifies low emissions, and dark purple identifies high emissions.
Future scenarios retain distinct dashed line styles.
The combined figure has a 3:1 aspect ratio.

## Scenario provenance

The figure labels follow the author's confirmed scenario names.
The source identifiers remain `2023`, `ssp126_2055`, and `ssp370_2055` in the reports.
Both future source identifiers contain 2055; both figure labels use 2050.
This label mapping leaves the simulation results unchanged.
"""
    (reports / "captions.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=Path, default=ROOT / "simulations")
    parser.add_argument("--models", type=Path, default=CONVERSION / "3_block_epjson")
    parser.add_argument("--building-metrics", type=Path,
                        default=CONVERSION / "1_individual_epjson/building_metrics.csv")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "simulations/reports/energy")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--bin-width", type=float, default=25,
                        help="Shared heating demand intensity bin width in kWh/m².")
    parser.add_argument("--reuse-reports", action="store_true",
                        help="Verify saved inputs and reports before plotting them again.")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("Provide a positive PNG resolution.")
    if not np.isfinite(args.bin_width) or args.bin_width <= 0:
        parser.error("Provide a finite positive bin width.")
    blocks = discover_blocks(args.simulations)
    metrics = read_metrics(args.building_metrics)
    geometry, block_geometry, sources = read_geometry(args.models, blocks)
    missing = set(geometry) - set(metrics.index)
    if missing:
        raise ValueError(f"Provide total floor areas for {len(missing)} modelled buildings.")
    sources.append(input_record(args.building_metrics))
    reports = args.reports_dir
    if args.reuse_reports:
        provenance = json.loads((reports / "provenance.json").read_text())
        if provenance["schema_version"] != REPORT_SCHEMA or provenance["scenarios"] != SCENARIOS:
            raise ValueError("Collect reports for the current schema and scenario labels.")
        current = {"simulations": str(args.simulations.resolve()),
                   "models": str(args.models.resolve()),
                   "building_metrics": str(args.building_metrics.resolve())}
        if provenance["locations"] != current or provenance["blocks"] != blocks:
            raise ValueError("Collect reports for the selected input locations and blocks.")
        for record in provenance["inputs"]:
            if input_record(Path(record["path"])) != record:
                raise ValueError(f"Collect reports for the changed input: {record['path']}.")
        for name, checksum in provenance["reports_sha256"].items():
            if digest(reports / name) != checksum:
                raise ValueError(f"Collect the changed report again: {name}.")
        buildings = pd.read_csv(reports / "building_energy.csv", dtype={"scenario": str})
        buildings, summary = summarise(buildings)
        print("Verified saved reports and source hashes.", flush=True)
    else:
        buildings, status, energy_sources = collect(args.simulations, block_geometry, metrics)
        buildings, summary = summarise(buildings)
        checks, csv_sources = crosscheck_csv(args.simulations, buildings)
        reports.mkdir(parents=True, exist_ok=True)
        for name, frame in (("building_energy", buildings), ("scenario_energy", summary),
                            ("run_status", status), ("csv_crosscheck", checks)):
            frame.to_csv(reports / f"{name}.csv", index=False)
        provenance = {
            "schema_version": REPORT_SCHEMA, "scenarios": SCENARIOS,
            "scenario_label_source": "Author-confirmed labels; future source folders retain 2055.",
            "blocks": blocks,
            "locations": {"simulations": str(args.simulations.resolve()),
                          "models": str(args.models.resolve()),
                          "building_metrics": str(args.building_metrics.resolve())},
            "inputs": sources + energy_sources + csv_sources,
            "collection_script_sha256": digest(Path(__file__)),
            "metric": {"variable": HEATING_VARIABLE, "input_units": "J",
                       "joules_per_kwh": JOULES_PER_KWH, "annual_hours": HOURS_PER_YEAR,
                       "floor_area": "Total modelled floor area across all storeys: building_metrics.csv floor_area.",
                       "comparison": "Buildings with complete results in all three scenarios.",
                       "completion": "Successful eplusout.end and complete annual hourly heating records.",
                       "calendar": "Internal calendar years are independent of scenario display labels."},
            "reports_sha256": {p.name: digest(p) for p in reports.glob("*.csv")},
        }
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, map_settings = draw_maps(buildings, geometry)
    outputs = save_figure(fig, args.output_dir / FIGURE_NAMES[0], args.dpi)
    bins, curves, distribution = heating_distribution(buildings, args.bin_width)
    outputs += save_figure(draw_comparison(summary, bins, curves, distribution),
                           args.output_dir / FIGURE_NAMES[1], args.dpi)
    for name, frame in (("heating_eui_bins.csv", bins), ("heating_eui_curves.csv", curves)):
        frame.to_csv(reports / name, index=False)
        provenance["reports_sha256"][name] = digest(reports / name)
    write_captions(reports, summary, map_settings, distribution)
    provenance["figures"] = {"maps": map_settings, "comparison": distribution,
                              "format": "PNG", "dpi": args.dpi,
                              "script_sha256": digest(Path(__file__)),
                              "outputs": [input_record(p) for p in outputs]}
    provenance["software"] = {"numpy": np.__version__, "pandas": pd.__version__,
                               "matplotlib": matplotlib.__version__, "sqlite": sqlite3.sqlite_version}
    (reports / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(summary[["scenario_label", "buildings", "heating_gwh", "change_from_present_pct"]].to_string(index=False))
    print(f"Wrote figures to {args.output_dir.resolve()}.", flush=True)


if __name__ == "__main__":
    main()
