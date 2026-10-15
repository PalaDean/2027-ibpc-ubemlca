"""Map annual hours with operative temperature above 30 °C per building."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import math
from pathlib import Path
import re
import sqlite3

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import plot_energy as energy


OPERATIVE_VARIABLE = "Zone Operative Temperature"
THRESHOLD_C = 30.0
FIGURE_NAME = "annual_overheating_hours_maps"
REPORT_SCHEMA = 1
HOUR_BANDS = ("0–50", ">50–100", ">100–200", ">200–400", ">400")
HOUR_COLOURS = ("#A8C5A0", "#F2DFA2", "#DC8B80", "#BD5B55", "#873C46")
# Counts are whole hours: 50 stays green, 100 stays yellow, and 101 starts red.
HOUR_BOUNDARIES = (0, 51, 101, 201, 401, energy.HOURS_PER_YEAR + 1)


def apply_style() -> None:
    """Use the original serif typography with normal-weight headings."""
    energy.apply_style()
    energy.plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.weight": "normal", "axes.labelweight": "normal",
    })


def colour_scale() -> tuple:
    cmap = ListedColormap(HOUR_COLOURS, name="annual_overheating_hours")
    return BoundaryNorm(HOUR_BOUNDARIES, cmap.N, clip=True), cmap


def read_block(run: Path, geometry: dict) -> tuple[list[dict], dict]:
    """Count annual exceedance hours after validating single-zone building results."""
    zones = geometry["zones"]
    if len(set(zones.values())) != len(zones):
        raise ValueError("Provide one thermal zone per building for this overheating metric.")
    rows = [{"zone": name, "building_id": identifier, "complete": False,
             "overheating_hours": np.nan, "valid_hours": 0, "reason": "missing output"}
            for name, identifier in zones.items()]
    status = {"run": run.name, "status": "incomplete", "zones": len(rows),
              "complete_zones": 0, "calendar_year": None, "reason": "missing output"}
    sql, end = run / "eplusout.sql", run / "eplusout.end"
    if not sql.exists() or not end.exists():
        return rows, status
    if not re.search(r"Completed Successfully.*\b0 Severe Errors", end.read_text()):
        status["reason"] = "unsuccessful completion record"
        for row in rows:
            row["reason"] = status["reason"]
        return rows, status
    with closing(sqlite3.connect(sql.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        sql_zones = conn.execute("SELECT ZoneName, Multiplier, ListMultiplier FROM Zones").fetchall()
        if len(sql_zones) != len(zones) or {r[0].upper() for r in sql_zones} != set(zones):
            raise ValueError(f"Match unique model zones to simulation zones in {run}.")
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
            'WHERE EnvironmentPeriodIndex=? AND COALESCE(WarmupFlag,0)=0', (environment,),
        ).fetchall()
        valid_calendar = energy.annual_calendar_valid(times)
        status["calendar_year"] = times[0][1] if times else None
        dictionary = conn.execute(
            "SELECT KeyValue, Units, Type FROM ReportDataDictionary "
            "WHERE Name=? AND ReportingFrequency='Hourly' AND IsMeter=0", (OPERATIVE_VARIABLE,),
        ).fetchall()
        names = [r[0].upper() for r in dictionary]
        if len(names) != len(set(names)) or not set(names) <= set(zones):
            raise ValueError(f"Provide unique operative temperature outputs for the model zones in {run}.")
        if any(r[1:] != ("C", "Avg") for r in dictionary):
            raise ValueError(f"Provide hourly mean operative temperatures in degrees Celsius in {run}.")
        temperatures = conn.execute(
            'SELECT d.KeyValue, COUNT(*), COUNT(r.Value), COUNT(DISTINCT r.TimeIndex), '
            'SUM(CASE WHEN r.Value > ? THEN 1 ELSE 0 END), MIN(r.Value), MAX(r.Value) '
            'FROM ReportData r JOIN ReportDataDictionary d '
            'ON d.ReportDataDictionaryIndex=r.ReportDataDictionaryIndex '
            'JOIN Time t ON t.TimeIndex=r.TimeIndex '
            'WHERE d.Name=? AND d.ReportingFrequency=\'Hourly\' AND d.IsMeter=0 '
            'AND t.EnvironmentPeriodIndex=? AND COALESCE(t.WarmupFlag,0)=0 '
            'GROUP BY d.ReportDataDictionaryIndex', (THRESHOLD_C, OPERATIVE_VARIABLE, environment),
        ).fetchall()
    temperatures = {r[0].upper(): r[1:] for r in temperatures}
    for row in rows:
        values = temperatures.get(row["zone"])
        if values is None:
            row["reason"] = "missing operative temperature variable"
            continue
        count, finite_count, distinct_count, hours, minimum, maximum = values
        complete = (valid_calendar and count == finite_count == distinct_count == energy.HOURS_PER_YEAR
                    and all(v is not None and np.isfinite(v) for v in (minimum, maximum)))
        row.update(complete=bool(complete), valid_hours=finite_count,
                   overheating_hours=int(hours) if complete else np.nan,
                   reason="" if complete else "incomplete or invalid annual operative temperature records")
    status["complete_zones"] = sum(r["complete"] for r in rows)
    status["status"] = "complete" if status["complete_zones"] == len(rows) else "incomplete"
    status["reason"] = "; ".join(sorted({r["reason"] for r in rows if r["reason"]}))
    return rows, status


def collect(simulations: Path, block_geometry: dict) -> tuple:
    rows, statuses, sources = [], [], []
    jobs = [(scenario, block, geometry) for scenario in energy.SCENARIOS
            for block, geometry in block_geometry.items()]

    def read_run(job):
        scenario, block, geometry = job
        run = simulations / f"{scenario}_{block}"
        result, status = read_block(run, geometry)
        inputs = [energy.input_record(run / name) for name in ("eplusout.sql", "eplusout.end")]
        return scenario, block, result, status, inputs

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (scenario, block, result, status, inputs) in enumerate(pool.map(read_run, jobs)):
            common = {"scenario": scenario, "scenario_label": energy.SCENARIOS[scenario], "block": block}
            rows.extend({**common, **row} for row in result)
            statuses.append({**common, **status})
            sources.extend(inputs)
            completed = index % len(block_geometry) + 1
            if completed % 10 == 0 or completed == len(block_geometry):
                print(f"Collected {scenario}: {completed}/{len(block_geometry)} blocks.", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(statuses), sources


def summarise(buildings: pd.DataFrame) -> pd.DataFrame:
    if buildings.duplicated(["scenario", "building_id"]).any():
        raise ValueError("Provide one result per building and scenario.")
    summary = []
    for scenario, label in energy.SCENARIOS.items():
        selected = buildings.loc[buildings.scenario == scenario]
        values = selected.loc[selected.complete, "overheating_hours"]
        if values.empty:
            raise ValueError(f"Provide complete annual operative temperature records for {scenario}.")
        summary.append({"scenario": scenario, "scenario_label": label,
                        "buildings": len(values), "unavailable_buildings": int((~selected.complete).sum()),
                        "buildings_above_30_c": int((values > 0).sum()),
                        "minimum_hours": float(values.min()), "median_hours": float(values.median()),
                        "mean_hours": float(values.mean()), "maximum_hours": float(values.max())})
    return pd.DataFrame(summary)


def crosscheck_csv(simulations: Path, buildings: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Compare annual counts with one complete hourly CSV export per scenario."""
    checks, sources = [], []
    for scenario in energy.SCENARIOS:
        selected = buildings.loc[buildings.scenario == scenario]
        complete = selected.groupby("block").complete.all()
        blocks = sorted(complete.index[complete], key=lambda b: int(b.rsplit("_", 1)[1]))
        if not blocks:
            raise ValueError(f"Provide a complete block for the CSV check of {scenario}.")
        block = blocks[0]
        path = simulations / f"{scenario}_{block}" / "eplusout.csv"
        suffix = f":{OPERATIVE_VARIABLE} [C](Hourly)"
        hourly = pd.read_csv(path, usecols=lambda c: c.strip().endswith(suffix))
        if len(hourly) != energy.HOURS_PER_YEAR or not len(hourly.columns):
            raise ValueError(f"Provide annual hourly CSV operative temperatures in {path}.")
        if not np.isfinite(hourly.to_numpy()).all():
            raise ValueError(f"Provide finite CSV operative temperatures in {path}.")
        expected = selected.loc[selected.block == block].set_index("building_id").overheating_hours
        actual = (hourly > THRESHOLD_C).sum()
        actual.index = actual.index.map(lambda name: energy.building_id(name.split(":", 1)[0]))
        if actual.index.duplicated().any() or set(actual.index) != set(expected.index):
            raise ValueError(f"Match unique CSV building identifiers in {path}.")
        actual = actual.reindex(expected.index)
        if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
            raise ValueError(f"Reconcile SQLite and CSV overheating hours in {path}.")
        checks.append({"scenario": scenario, "block": block, "buildings": len(actual),
                       "maximum_building_difference_hours": float((actual - expected).abs().max()),
                       "passed": True})
        sources.append(energy.input_record(path))
    return pd.DataFrame(checks), sources


def draw_maps(buildings: pd.DataFrame, geometry: dict) -> tuple:
    values = buildings.loc[buildings.complete, "overheating_hours"].dropna()
    if values.empty:
        raise ValueError("Provide complete annual overheating hours for the maps.")
    if ((values < 0) | (values > energy.HOURS_PER_YEAR) | (values != np.floor(values))).any():
        raise ValueError("Provide whole-hour annual overheating counts between 0 and 8760.")
    norm, cmap = colour_scale()
    ticks = (np.array(HOUR_BOUNDARIES[:-1]) + np.array(HOUR_BOUNDARIES[1:])) / 2
    fig, settings = energy.draw_building_maps(
        buildings, geometry, value_column="overheating_hours", limits=(0.0, energy.HOURS_PER_YEAR),
        colourbar_label="Annual overheating hours (operative temperature > 30 °C)",
        norm=norm, cmap=cmap,
        colourbar_options={"ticks": ticks, "spacing": "uniform", "drawedges": True})
    fig.axes[-1].set_xticks(ticks, labels=HOUR_BANDS)
    fig.axes[-1].tick_params(axis="x", which="both", length=0, pad=5)
    fig.axes[-1].minorticks_off()
    for ax in fig.axes[:3]:
        for collection in ax.collections:
            collection.set_edgecolor("#70757A")
            collection.set_linewidth(.11)
        for line in ax.lines:
            line.set_linewidth(.8)
    for x in (.335, .665):
        fig.add_artist(Line2D([x, x], [.29, .975], transform=fig.transFigure,
                              color="#CED1D5", linewidth=.45))
    settings["style"] = {"font": "DejaVu Serif",
                         "scenario_heading_weight": "normal", "panel_dividers": "light grey",
                         "building_outline_colour": "#70757A"}
    settings["valid_range_hours"] = settings.pop("colour_limits")
    settings["hour_bands"] = list(HOUR_BANDS)
    settings["colours"] = list(HOUR_COLOURS)
    settings["integer_count_boundaries"] = list(HOUR_BOUNDARIES)
    settings["classification_basis"] = "Author-selected colour thresholds; annual overheating exposure bands."
    settings["interpretation"] = "Study-defined exposure bands; health-risk classes require separate validation."
    settings["band_buildings"] = {
        scenario: {label: int((norm(frame.overheating_hours.to_numpy()) == i).sum())
                   for i, label in enumerate(HOUR_BANDS)}
        for scenario in energy.SCENARIOS
        for frame in [buildings.loc[(buildings.scenario == scenario) & buildings.complete]]}
    return fig, settings


def write_captions(reports: Path, settings: dict) -> None:
    (reports / "captions.md").write_text(f"""# Draft caption and alt text

Caption: Annual overheating hours per building for the Present scenario, 2050 low emissions scenario, and 2050 high emissions scenario.
An overheating hour was counted when hourly operative temperature exceeded 30 °C.
All 8,760 hours of the annual weather simulation were assessed, including unoccupied hours.
Each building was modelled with one thermal zone.
The maps share one extent and five annual overheating exposure bands.
Green indicates 0–50 hours, and yellow indicates more than 50 to 100 hours.
Red indicates more than 100 to 200 hours.
Darker red indicates more than 200 to 400 hours, and the darkest red indicates more than 400 hours.
The 50-hour and 100-hour thresholds were selected for this figure.
These study-defined exposure bands describe annual exceedance duration; health-risk classes require separate validation.
Hatched grey footprints indicate unavailable results.

Alt text: Three maps in one row show building footprints coloured by annual overheating hours.
The maps share one horizontal colour bar with the labels {', '.join(HOUR_BANDS)} hours.
Yellow starts above 50 hours, and red starts above 100 hours.
Darker red indicates longer annual exceedance duration.
A north arrow and a 200 m scale appear in the first map.

The scenario labels match the heating demand figures.
The future source identifiers retain 2055, while the figure labels use 2050.
The evidence review and source links are in `simulations/overheating_thresholds.md`.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=Path, default=energy.ROOT / "simulations")
    parser.add_argument("--models", type=Path, default=energy.CONVERSION / "3_block_epjson")
    parser.add_argument("--reports-dir", type=Path, default=energy.ROOT / "simulations/reports/overheating")
    parser.add_argument("--output-dir", type=Path, default=energy.ROOT / "figures")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("Provide a positive preview resolution.")
    blocks = energy.discover_blocks(args.simulations)
    geometry, block_geometry, sources = energy.read_geometry(args.models, blocks)
    buildings, status, temperature_sources = collect(args.simulations, block_geometry)
    summary = summarise(buildings)
    checks, csv_sources = crosscheck_csv(args.simulations, buildings)
    reports = args.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    report_paths = []
    for name, frame in (("building_overheating", buildings), ("scenario_overheating", summary),
                        ("run_status", status), ("csv_crosscheck", checks)):
        path = reports / f"{name}.csv"
        frame.to_csv(path, index=False)
        report_paths.append(path)
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, settings = draw_maps(buildings, geometry)
    outputs = energy.save_figure(fig, args.output_dir / FIGURE_NAME, args.dpi)
    write_captions(reports, settings)
    provenance = {
        "schema_version": REPORT_SCHEMA, "scenarios": energy.SCENARIOS,
        "scenario_label_source": "Heating demand figure labels; future source folders retain 2055.",
        "blocks": blocks,
        "locations": {"simulations": str(args.simulations.resolve()), "models": str(args.models.resolve())},
        "inputs": sources + temperature_sources + csv_sources,
        "scripts_sha256": {Path(p).name: energy.digest(Path(p)) for p in (__file__, energy.__file__)},
        "metric": {"variable": OPERATIVE_VARIABLE, "input_units": "C", "threshold_c": THRESHOLD_C,
                   "comparison_operator": ">", "output_units": "hours per year",
                   "definition": "Count hourly mean operative temperatures strictly above 30 degrees Celsius.",
                   "annual_hours": energy.HOURS_PER_YEAR, "assessment": "All annual hours, including unoccupied hours.",
                   "excluded_periods": "Warm-up and design-day records.",
                   "building_aggregation": "One thermal zone per building, verified against model geometry.",
                   "completion": "Successful eplusout.end and complete finite annual operative temperature records.",
                   "calendar": "Internal calendar years are independent of scenario display labels."},
        "reports_sha256": {p.name: energy.digest(p) for p in report_paths},
        "figures": {"maps": settings, "preview_dpi": args.dpi,
                    "outputs": [energy.input_record(p) for p in outputs]},
        "software": {"numpy": np.__version__, "pandas": pd.__version__,
                     "matplotlib": energy.matplotlib.__version__, "sqlite": sqlite3.sqlite_version},
    }
    (reports / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote figures to {args.output_dir.resolve()}.", flush=True)


if __name__ == "__main__":
    main()
