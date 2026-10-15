# Annual heating demand and overheating figures

This command produces two PNG figures from the existing EnergyPlus simulations.
The first figure contains one row of three building maps.
The second figure combines heating EUI distributions and aggregated annual heating demand within a 3:1 canvas.
The distribution occupies 75% of the combined panel width.
Three vertical bars occupy the remaining 25%.

## Run the command

Use the `citygml-epjson` conda environment.
Run this command from the conference repository:

```bash
python simulations/plot_energy.py
```

The default geometry and floor-area sources belong to the adjacent CityGML conversion project.
Provide explicit paths when these inputs move:

```bash
python simulations/plot_energy.py \
  --simulations simulations \
  --models /path/to/3_block_epjson \
  --building-metrics /path/to/1_individual_epjson/building_metrics.csv \
  --reports-dir simulations/reports/energy \
  --output-dir figures
```

Render existing reports again with this command:

```bash
python simulations/plot_energy.py --reuse-reports
```

This option verifies source hashes and report hashes before plotting.
Collect the reports again after changing an input.

## Scenario labels

| Source identifier | Figure label |
| --- | --- |
| `2023` | Present scenario |
| `ssp126_2055` | 2050 low emissions scenario |
| `ssp370_2055` | 2050 high emissions scenario |

The author confirmed these figure labels.
Both future source identifiers retain 2055, while both figure labels use 2050.
Reports retain both the source identifier and the figure label.
The internal simulation calendar reports 2017 independently of these labels.

## Metric definitions

Annual heating demand is the sum of `Zone Air System Sensible Heating Energy` over the annual weather simulation.
Warm-up records and design-day records are excluded.
Energy in joules is divided by 3,600,000 to obtain kWh.
This variable represents sensible thermal heating demand.

Annual heating demand intensity is annual heating demand divided by total modelled floor area across all storeys.
The denominator comes from `building_metrics.csv`, using its `floor_area` column.
Building identifiers link the denominator to the simulation results.
The single-zone models report footprint area in the EnergyPlus zone records.
Their total modelled floor area is larger because it includes upper storeys.

Aggregated annual heating demand is summed across buildings with complete results in all three scenarios.
The bars report this sum in GWh.
Percentage changes are calculated from these sums relative to the Present scenario.

The maps share one extent and one colour scale.
The upper colour limit is the pooled 98th percentile, rounded upwards to 25 kWh/m².
The lower colour limit is zero.
The colour-bar extension identifies clipped values.
Provenance and draft captions record the clipped building counts.
Hatched grey footprints indicate unavailable results.

The heating EUI distribution uses the same annual heating demand intensity and the same buildings as the scenario totals.
Histogram outlines use shared 25 kWh/m² bins.
Use `--bin-width` to change this width.
Smooth curves show Gaussian kernel density estimates with one shared bandwidth.
The bandwidth uses the smaller of the pooled standard deviation and the pooled interquartile range divided by 1.34.
This value is multiplied by 0.9 and the building count per scenario raised to −0.2.
Constant samples use half the bin width.
The kernels are reflected at zero to retain the non-negative domain.

The distribution displays 0–400 kWh/m², with the vertical axis showing the number of buildings.
Histogram outlines show actual building counts per bin.
Smooth curves show density multiplied by the building count and bin width.
This scaling expresses smoothed building counts per bin.
Vertical lines indicate scenario medians calculated from all included buildings.
The shared legend lists each scenario and its median heating EUI.
The distribution reports retain the complete range, including the extreme values.
Buildings above 400 kWh/m² remain included in the medians, curve estimation, and aggregated annual heating demand.

Blue follows the methodology figure and identifies the Present scenario.
Light purple identifies the 2050 low emissions scenario.
Dark purple identifies the 2050 high emissions scenario.
The Present scenario uses solid lines, while future scenarios retain dashed and dash-dot lines.
The vertical bars use the same scenario colours and distinct hatching.
Bar annotations show annual heating demand in GWh and percentage changes relative to the Present scenario.

## Outputs

The `figures/` folder receives these image files:

- `annual_heating_intensity_maps.png`.
- `heating_demand_comparison.png`.

The exporter writes PNG files.
The maps are 12.2 cm wide.
The maps occupy a 6.4 cm high canvas.
The combined figure occupies a 6 by 2 inch canvas.
At the default 600 dpi, the combined figure is 3,600 by 1,200 pixels.
Use `--dpi` to change the PNG resolution.

The `simulations/reports/energy/` folder receives these reports:

- `building_energy.csv`: annual demand, floor area, intensity, completeness, and comparison membership.
- `scenario_energy.csv`: aggregated demand, included building counts, and percentage changes.
- `run_status.csv`: completion and annual-data checks for each block.
- `csv_crosscheck.csv`: comparisons against one hourly CSV export per scenario.
- `heating_eui_bins.csv`: bin boundaries, building counts, percentages, and probability densities.
- `heating_eui_curves.csv`: probability densities and smoothed building counts across the complete range.
- `provenance.json`: source hashes, report hashes, metric definitions, scenario labels, and plotting settings.
- `captions.md`: draft captions and alt text for both figures.

The command uses the blocks found in the scenario output folders.
The current outputs cover blocks 1–50 and 1,422 buildings per scenario.
The additional geometry model for block 100 is outside this simulation coverage.
Manuscript integration is a separate task.

## Validation

Successful completion is checked through `eplusout.end`.
The SQL completion flags remain false in the current outputs despite successful completion records.
Each thermal zone must also contain 8,760 unique, finite, non-negative hourly heating values.
The annual calendar must contain every hourly interval of a non-leap year.
Incomplete results retain their footprint and an unavailable value.
The comparison excludes those buildings from all three totals.

Model zone identifiers and ground-floor surface areas are checked against the simulation database.
Duplicate identifiers, invalid floor areas, and geometry mismatches stop collection.
One complete block per scenario is checked against its CSV export.
The comparison allows a relative tolerance of 0.000001 and an absolute tolerance of 0.001 kWh.

Run the calculation tests with this command:

```bash
python -m pytest simulations/tests -q
```

The tests cover unit conversion, excluded simulation periods, floor-area normalisation, common building coverage, and percentage changes.
They also cover missing records, duplicate records, invalid geometry matches, and courtyard preservation.

## Annual overheating map

Run this command in the same environment:

```bash
python simulations/plot_overheating.py
```

The figure contains one row of three building maps.
Scenario labels, footprints, map extent, and dimensions match the heating demand maps.
The original serif typography is combined with a muted green-to-red palette and thin grey panel dividers.
The output is `figures/annual_overheating_hours_maps.png`.

Each footprint shows annual hours with `Zone Operative Temperature` strictly above 30 °C.
Hours at exactly 30 °C contribute zero to the count.
All 8,760 annual hours are assessed, including unoccupied hours.
Warm-up records and design-day records are excluded.
Each building has one thermal zone, which is verified before collection.

All three maps share one colour scale.
Green indicates 0–50 annual overheating hours.
Yellow indicates more than 50 to 100 hours.
Red indicates more than 100 to 200 hours.
Darker red indicates more than 200 to 400 hours.
The darkest red indicates more than 400 hours.
Each band has equal width in the colour bar.
These study-defined exposure bands describe annual exceedance duration; health-risk classes require separate validation.
The [threshold evidence review](overheating_thresholds.md) records the sources and their applicability.
Hatched grey footprints indicate unavailable results.

The command accepts `--simulations`, `--models`, `--reports-dir`, `--output-dir`, and `--dpi`.
The default report folder is `simulations/reports/overheating/`.
It contains building results, scenario summaries, completion checks, CSV comparisons, source hashes, a draft caption, and alt text.
Every building result requires 8,760 unique, finite hourly operative temperatures and successful simulation completion.
One complete block per scenario is checked against its CSV export.
The SQL and CSV exceedance counts must agree exactly.

The calculation tests also check strict threshold exceedance, complete annual coverage, invalid temperatures, and the requirement for one thermal zone per building.
