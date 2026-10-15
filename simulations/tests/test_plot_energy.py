"""Check annual heating demand, normalisation, and scenario comparisons."""

import importlib.util
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon


spec = importlib.util.spec_from_file_location("plot_energy", Path(__file__).parents[1] / "plot_energy.py")
energy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(energy)

A = "ID_00000000-0000-0000-0000-000000000001"
B = "ID_00000000-0000-0000-0000-000000000002"


@pytest.fixture
def annual_run(tmp_path):
    run = tmp_path / "2023_block_1"
    run.mkdir()
    (run / "eplusout.end").write_text("EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors;")
    with sqlite3.connect(run / "eplusout.sql") as conn:
        conn.executescript("""
            CREATE TABLE Zones (ZoneName TEXT, Multiplier REAL, ListMultiplier REAL);
            CREATE TABLE Surfaces (SurfaceName TEXT, GrossArea REAL, ClassName TEXT);
            CREATE TABLE EnvironmentPeriods (EnvironmentPeriodIndex INTEGER, EnvironmentType INTEGER);
            CREATE TABLE Time (TimeIndex INTEGER, Year INTEGER, Month INTEGER, Day INTEGER,
                Hour INTEGER, Minute INTEGER, Interval INTEGER, EnvironmentPeriodIndex INTEGER, WarmupFlag INTEGER);
            CREATE TABLE ReportDataDictionary (ReportDataDictionaryIndex INTEGER, KeyValue TEXT,
                Name TEXT, ReportingFrequency TEXT, IsMeter INTEGER, Units TEXT, Type TEXT);
            CREATE TABLE ReportData (TimeIndex INTEGER, ReportDataDictionaryIndex INTEGER, Value REAL);
        """)
        conn.execute("INSERT INTO Zones VALUES (?, 1, 1)", (A,))
        conn.execute("INSERT INTO Surfaces VALUES ('GROUND_A', 100, 'Floor')")
        conn.executemany("INSERT INTO EnvironmentPeriods VALUES (?, ?)", [(1, 3), (2, 1)])
        times = [(i, 2017, day.month, day.day, (i - 1) % 24 + 1, 0, 60, 1, None)
                 for i in range(1, 8761)
                 for day in [energy.date(2017, 1, 1) + energy.timedelta(days=(i - 1) // 24)]]
        conn.executemany("INSERT INTO Time VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", times)
        # Warm-up and design-day energy must contribute zero annual demand.
        conn.executemany("INSERT INTO Time VALUES (?, 2017, 1, 1, 1, 0, 60, ?, ?)",
                         [(8761, 1, 1), (8762, 2, 0)])
        conn.execute("INSERT INTO ReportDataDictionary VALUES (1, ?, ?, 'Hourly', 0, 'J', 'Sum')",
                     (A, energy.HEATING_VARIABLE))
        conn.execute("INSERT INTO ReportDataDictionary VALUES (2, ?, ?, 'Hourly', 0, 'J', 'Sum')",
                     (A + ' IDEAL LOADS', 'Zone Ideal Loads Supply Air Total Heating Energy'))
        conn.executemany("INSERT INTO ReportData VALUES (?, 1, ?)",
                         [(i, 3_600_000.) for i in range(1, 8761)]
                         + [(8761, 1e15), (8762, 1e15)])
        conn.execute("INSERT INTO ReportData VALUES (1, 2, 1e15)")
    geometry = {"zones": {A: A}, "floors": {"GROUND_A": 100.}}
    return run, geometry


def mutate(run, statement, parameters=()):
    with sqlite3.connect(run / "eplusout.sql") as conn:
        conn.execute(statement, parameters)


def test_hourly_energy_conversion_excludes_warmup_design_days_and_ideal_loads(annual_run):
    rows, status = energy.read_block(*annual_run)
    assert status["status"] == "complete"
    assert status["calendar_year"] == 2017
    assert rows[0]["valid_hours"] == 8760
    assert rows[0]["heating_kwh"] == 8760


@pytest.mark.parametrize("statement", [
    "DELETE FROM ReportData WHERE TimeIndex=7000 AND ReportDataDictionaryIndex=1",
    "INSERT INTO ReportData VALUES (7000, 1, 3600000)",
    "UPDATE ReportData SET Value=NULL WHERE TimeIndex=7000 AND ReportDataDictionaryIndex=1",
    "UPDATE ReportData SET Value=-1 WHERE TimeIndex=7000 AND ReportDataDictionaryIndex=1",
    "UPDATE Time SET Month=13 WHERE TimeIndex=7000",
    "DELETE FROM ReportDataDictionary WHERE ReportDataDictionaryIndex=1",
])
def test_invalid_annual_records_are_unavailable(annual_run, statement):
    run, geometry = annual_run
    mutate(run, statement)
    rows, status = energy.read_block(run, geometry)
    assert status["status"] == "incomplete"
    assert not rows[0]["complete"]
    assert math.isnan(rows[0]["heating_kwh"])


def test_failed_run_retains_building_as_unavailable(annual_run):
    run, geometry = annual_run
    (run / "eplusout.end").write_text("EnergyPlus Terminated--Fatal Error Detected.")
    rows, status = energy.read_block(run, geometry)
    assert rows[0]["building_id"] == A
    assert not rows[0]["complete"]
    assert status["reason"] == "unsuccessful completion record"


def test_geometry_mismatch_stops_collection(annual_run):
    run, geometry = annual_run
    mutate(run, "UPDATE Surfaces SET GrossArea=99")
    with pytest.raises(ValueError, match="ground-floor surface"):
        energy.read_block(run, geometry)


def test_duplicate_heating_variable_stops_collection(annual_run):
    run, geometry = annual_run
    mutate(run, "INSERT INTO ReportDataDictionary VALUES (3, ?, ?, 'Hourly', 0, 'J', 'Sum')",
           (A, energy.HEATING_VARIABLE))
    with pytest.raises(ValueError, match="unique heating"):
        energy.read_block(run, geometry)


def test_total_floor_area_normalises_demand(annual_run):
    run, geometry = annual_run
    metrics = pd.DataFrame({"building_id": [A], "total_floor_area_m2": [400.]}).set_index("building_id")
    results, status, _ = energy.collect(run.parent, {"block_1": geometry}, metrics)
    present = results.loc[results.scenario == "2023"].iloc[0]
    assert present.heating_intensity_kwh_m2 == 8760 / 400
    assert present.heating_intensity_kwh_m2 != 8760 / geometry["floors"]["GROUND_A"]
    assert len(results) == 3
    assert len(status) == 3


@pytest.mark.parametrize("records", [
    [(A, 100, 1), (A.lower(), 100, 1)],
    [(A, 0, 1)],
    [(A, float("nan"), 1)],
])
def test_invalid_floor_area_records_stop_collection(tmp_path, records):
    path = tmp_path / "metrics.csv"
    pd.DataFrame(records, columns=["gml_id", "floor_area", "number_of_floors"]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        energy.read_metrics(path)


def test_missing_floor_area_stops_collection(annual_run):
    run, geometry = annual_run
    metrics = pd.DataFrame({"building_id": [B], "total_floor_area_m2": [400.]}).set_index("building_id")
    with pytest.raises(ValueError, match="total floor area"):
        energy.collect(run.parent, {"block_1": geometry}, metrics)


def comparison_rows():
    rows = []
    for scenario, values in zip(energy.SCENARIOS, [(100., 900.), (50., 900.), (40., 800.)]):
        for identifier, demand in zip((A, B), values):
            rows.append({"scenario": scenario, "building_id": identifier, "complete": True,
                         "heating_kwh": demand, "total_floor_area_m2": 100.})
    return pd.DataFrame(rows)


def test_percentage_change_uses_aggregated_demand():
    _, summary = energy.summarise(comparison_rows())
    assert list(summary.heating_kwh) == [1000, 950, 840]
    assert list(summary.change_from_present_pct) == [0, -5, -16]
    assert list(summary.scenario_label) == list(energy.SCENARIOS.values())
    assert summary.heating_gwh.iloc[0] == .001


def test_comparison_uses_same_complete_buildings():
    rows = comparison_rows()
    rows.loc[(rows.scenario == "ssp370_2055") & (rows.building_id == B), "complete"] = False
    buildings, summary = energy.summarise(rows)
    assert list(summary.heating_kwh) == [100, 50, 40]
    assert list(summary.buildings) == [1, 1, 1]
    assert list(summary.excluded_buildings) == [1, 1, 1]
    assert not buildings.loc[buildings.building_id == B, "in_common_comparison"].any()


def test_duplicate_building_result_stops_comparison():
    rows = comparison_rows()
    with pytest.raises(ValueError, match="one result"):
        energy.summarise(pd.concat([rows, rows.iloc[:1]]))


def test_zero_baseline_has_undefined_percentage_change():
    rows = comparison_rows()
    rows.loc[rows.scenario == "2023", "heating_kwh"] = 0
    _, summary = energy.summarise(rows)
    assert summary.change_from_present_pct.isna().all()


def test_polygon_paths_preserve_courtyard():
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(2, 2), (8, 2), (8, 8), (2, 8)]
    path = energy.polygon_patch(Polygon(shell, holes=[hole])).get_path()
    rings = path.to_polygons()
    signed_areas = [sum(x * y2 - x2 * y for (x, y), (x2, y2)
                        in zip(ring[:-1], ring[1:])) / 2 for ring in rings]
    assert signed_areas == [100, -36]


def test_colour_limit_uses_pooled_scenarios():
    frame = pd.DataFrame({"complete": [True] * 100, "heating_intensity_kwh_m2": np.arange(100.)})
    assert energy.colour_limits(frame) == (0, 100)


def test_sloping_ground_surface_uses_projected_footprint(tmp_path):
    model = {"Zone": {A: {}}, "BuildingSurface:Detailed": {
        "ground": {"surface_type": "Floor", "outside_boundary_condition": "Ground",
                   "zone_name": A, "vertices": [
                       {"vertex_x_coordinate": x, "vertex_y_coordinate": y, "vertex_z_coordinate": z}
                       for x, y, z in [(0, 0, 0), (10, 0, 0), (10, 10, 10), (0, 10, 10)]
                   ]}}}
    (tmp_path / "block_1.epJSON").write_text(json.dumps(model))
    buildings, blocks, _ = energy.read_geometry(tmp_path, ["block_1"])
    assert buildings[A]["footprint"].area == 100
    assert blocks["block_1"]["floors"]["GROUND"] == pytest.approx(100 * math.sqrt(2))


def test_csv_crosscheck_accepts_buildings_without_address(tmp_path):
    rows = []
    for scenario in energy.SCENARIOS:
        run = tmp_path / f"{scenario}_block_1"
        run.mkdir()
        column = f"{A}:{energy.HEATING_VARIABLE} [J](Hourly)"
        pd.DataFrame({column: np.full(8760, 3_600_000.)}).to_csv(run / "eplusout.csv", index=False)
        rows.append({"scenario": scenario, "block": "block_1", "complete": True,
                     "building_id": A, "heating_kwh": 8760.})
    checks, sources = energy.crosscheck_csv(tmp_path, pd.DataFrame(rows))
    assert checks.passed.all()
    assert (checks.maximum_building_difference_kwh == 0).all()
    assert len(sources) == 3


def distribution_rows():
    return pd.DataFrame([
        {"scenario": scenario, "building_id": str(i), "complete": True,
         "in_common_comparison": True, "heating_intensity_kwh_m2": value * factor}
        for scenario, factor in zip(energy.SCENARIOS, (1., .9, .8))
        for i, value in enumerate((0., 25., 75., 650.))
    ])


def test_distribution_bins_retain_extreme_values_and_normalisation():
    bins, curves, settings = energy.heating_distribution(distribution_rows())
    for scenario in energy.SCENARIOS:
        group = bins.loc[bins.scenario == scenario]
        assert group.building_count.sum() == 4
        assert group.building_share_pct.sum() == pytest.approx(100)
        assert (group.density * (group.bin_upper_kwh_m2 - group.bin_lower_kwh_m2)).sum() == pytest.approx(1)
        curve = curves.loc[curves.scenario == scenario]
        assert np.isfinite(curve.density).all()
        assert (curve.density >= 0).all()
        assert np.trapezoid(curve.density, curve.heating_intensity_kwh_m2) == pytest.approx(1, abs=1e-4)
        # Smoothed counts use the same building count and bin width as the histogram.
        count_integral = np.trapezoid(curve.smoothed_building_count, curve.heating_intensity_kwh_m2) / 25
        assert count_integral == pytest.approx(4, abs=4e-4)
    assert settings["display_range_kwh_m2"] == [0, 400]
    assert settings["buildings_above_display_range"] == {"2023": 1, "ssp126_2055": 1, "ssp370_2055": 1}
    # Values above the displayed range still contribute to the medians.
    assert settings["median_heating_eui_kwh_m2"] == {"2023": 50, "ssp126_2055": 45, "ssp370_2055": 40}
    present = bins.loc[bins.scenario == "2023"]
    assert present.iloc[0].building_count == 1
    assert present.iloc[1].building_count == 1
    assert present.iloc[-1].building_count == 1


def test_distribution_excludes_buildings_outside_common_comparison():
    rows = distribution_rows()
    extra = pd.DataFrame([{"scenario": "2023", "building_id": "extra", "complete": True,
                           "in_common_comparison": False, "heating_intensity_kwh_m2": 10000}])
    bins, _, settings = energy.heating_distribution(pd.concat([rows, extra]))
    assert settings["buildings_per_scenario"] == 4
    assert bins.bin_upper_kwh_m2.max() == 650


@pytest.mark.parametrize("bin_width", [0, -25, np.nan, np.inf])
def test_distribution_requires_positive_bin_width(bin_width):
    with pytest.raises(ValueError, match="positive bin width"):
        energy.heating_distribution(distribution_rows(), bin_width)


def test_distribution_constant_sample_has_finite_density():
    rows = distribution_rows()
    rows["heating_intensity_kwh_m2"] = 0
    bins, curves, settings = energy.heating_distribution(rows)
    assert settings["bandwidth_kwh_m2"] == 12.5
    assert np.isfinite(curves.density).all()
    assert (bins.building_count == 4).all()


def test_comparison_export_is_png_with_three_to_one_ratio(tmp_path):
    from PIL import Image

    rows = distribution_rows()
    rows["total_floor_area_m2"] = 100.
    rows["heating_kwh"] = rows.heating_intensity_kwh_m2 * rows.total_floor_area_m2
    _, summary = energy.summarise(rows)
    bins, curves, settings = energy.heating_distribution(rows)
    energy.apply_style()
    fig = energy.draw_comparison(summary, bins, curves, settings)
    distribution_ax, totals_ax = fig.axes
    assert distribution_ax.get_position().width / totals_ax.get_position().width == pytest.approx(3)
    assert distribution_ax.get_xlim() == (0, 400)
    assert distribution_ax.get_ylabel() == "Number of buildings"
    assert [bar.get_height() for bar in totals_ax.patches] == pytest.approx([.075, .0675, .06])
    outputs = energy.save_figure(fig, tmp_path / "comparison", 150)
    assert [p.name for p in outputs] == ["comparison.png"]
    with Image.open(outputs[0]) as result:
        assert result.width == 3 * result.height
    assert sorted(p.name for p in tmp_path.iterdir()) == ["comparison.png"]
