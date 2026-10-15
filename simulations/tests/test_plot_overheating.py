"""Check annual operative temperature exceedance counts and source validation."""

from contextlib import closing
import importlib.util
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

from test_plot_energy import A, annual_run, mutate  # noqa: F401


spec = importlib.util.spec_from_file_location("plot_energy", Path(__file__).parents[1] / "plot_energy.py")
energy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(energy)
sys.modules["plot_energy"] = energy
spec = importlib.util.spec_from_file_location("plot_overheating", Path(__file__).parents[1] / "plot_overheating.py")
overheating = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overheating)


@pytest.fixture
def operative_run(annual_run):
    run, geometry = annual_run
    with closing(sqlite3.connect(run / "eplusout.sql")) as conn:
        conn.execute("UPDATE ReportDataDictionary SET Name=?, Units='C', Type='Avg' "
                     "WHERE ReportDataDictionaryIndex=1", (overheating.OPERATIVE_VARIABLE,))
        conn.execute("UPDATE ReportData SET Value=30 WHERE ReportDataDictionaryIndex=1")
        # January and December exceedances establish assessment across the full year.
        conn.execute("UPDATE ReportData SET Value=30.000001 WHERE TimeIndex IN (1, 2, 8760) "
                     "AND ReportDataDictionaryIndex=1")
        conn.execute("UPDATE ReportData SET Value=60 WHERE TimeIndex IN (8761, 8762) "
                     "AND ReportDataDictionaryIndex=1")
        conn.execute("UPDATE ReportData SET Value=29.999999 WHERE TimeIndex=3 AND ReportDataDictionaryIndex=1")
        conn.commit()
    return run, geometry


def test_strict_annual_threshold_excludes_warmup_design_days_and_other_variables(operative_run):
    rows, status = overheating.read_block(*operative_run)
    assert status["status"] == "complete"
    assert rows[0]["valid_hours"] == 8760
    assert rows[0]["overheating_hours"] == 3


@pytest.mark.parametrize("value, expected", [(30.0, 0), (30.000001, 8760), (-5.0, 0)])
def test_complete_zero_and_full_year_counts(operative_run, value, expected):
    run, geometry = operative_run
    mutate(run, "UPDATE ReportData SET Value=? WHERE ReportDataDictionaryIndex=1", (value,))
    rows, _ = overheating.read_block(run, geometry)
    assert rows[0]["complete"]
    assert rows[0]["overheating_hours"] == expected


@pytest.mark.parametrize("statement", [
    "DELETE FROM ReportData WHERE TimeIndex=7000 AND ReportDataDictionaryIndex=1",
    "INSERT INTO ReportData VALUES (7000, 1, 31)",
    "UPDATE ReportData SET Value=NULL WHERE TimeIndex=7000 AND ReportDataDictionaryIndex=1",
    "UPDATE ReportData SET Value=1e999 WHERE TimeIndex=7000 AND ReportDataDictionaryIndex=1",
    "UPDATE ReportData SET Value=-1e999 WHERE TimeIndex=7000 AND ReportDataDictionaryIndex=1",
    "UPDATE Time SET Month=13 WHERE TimeIndex=7000",
    "UPDATE Time SET Interval=30 WHERE TimeIndex=7000",
    "DELETE FROM ReportDataDictionary WHERE ReportDataDictionaryIndex=1",
])
def test_invalid_annual_records_are_unavailable(operative_run, statement):
    run, geometry = operative_run
    mutate(run, statement)
    rows, status = overheating.read_block(run, geometry)
    assert status["status"] == "incomplete"
    assert not rows[0]["complete"]
    assert math.isnan(rows[0]["overheating_hours"])


def test_failed_run_retains_unavailable_footprint(operative_run):
    run, geometry = operative_run
    (run / "eplusout.end").write_text("EnergyPlus Terminated--Fatal Error Detected.")
    rows, _ = overheating.read_block(run, geometry)
    assert rows[0]["building_id"] == A
    assert not rows[0]["complete"]


def test_multizone_building_requires_explicit_aggregation(operative_run):
    run, geometry = operative_run
    geometry["zones"][A + "_UPPER"] = A
    with pytest.raises(ValueError, match="one thermal zone"):
        overheating.read_block(run, geometry)


@pytest.mark.parametrize("statement, match", [
    ("UPDATE Surfaces SET GrossArea=99", "ground-floor surface"),
    ("UPDATE ReportDataDictionary SET Units='K' WHERE ReportDataDictionaryIndex=1", "degrees Celsius"),
    ("UPDATE ReportDataDictionary SET Type='Sum' WHERE ReportDataDictionaryIndex=1", "hourly mean"),
    ("INSERT INTO ReportDataDictionary SELECT 3, KeyValue, Name, ReportingFrequency, IsMeter, Units, Type "
     "FROM ReportDataDictionary WHERE ReportDataDictionaryIndex=1", "unique operative"),
])
def test_mismatched_sources_stop_collection(operative_run, statement, match):
    run, geometry = operative_run
    mutate(run, statement)
    with pytest.raises(ValueError, match=match):
        overheating.read_block(run, geometry)


def test_csv_crosscheck_counts_strict_exceedances(tmp_path):
    rows = []
    for scenario in energy.SCENARIOS:
        run = tmp_path / f"{scenario}_block_1"
        run.mkdir()
        column = f"{A}:{overheating.OPERATIVE_VARIABLE} [C](Hourly)"
        values = np.full(8760, 30.)
        values[[0, 1, 8759]] = 30.000001
        pd.DataFrame({column: values}).to_csv(run / "eplusout.csv", index=False)
        rows.append({"scenario": scenario, "block": "block_1", "complete": True,
                     "building_id": A, "overheating_hours": 3})
    buildings = pd.DataFrame(rows)
    checks, sources = overheating.crosscheck_csv(tmp_path, buildings)
    assert checks.passed.all()
    assert (checks.maximum_building_difference_hours == 0).all()
    assert len(sources) == 3
    buildings.loc[0, "overheating_hours"] = 4
    with pytest.raises(ValueError, match="Reconcile"):
        overheating.crosscheck_csv(tmp_path, buildings)
