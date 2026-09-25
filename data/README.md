# Data

Both files are public, non-sensitive open-government datasets, downloaded unchanged as CSV.
The profiler reads them as-is; no cleaning was done before analysis.

## Dataset A — Chicago Energy Benchmarking (development dataset)

* **Organization:** City of Chicago, Chicago Data Portal
* **Title:** Chicago Energy Benchmarking
* **Source:** https://data.cityofchicago.org/d/xq83-jr8c
* **Direct CSV:** https://data.cityofchicago.org/api/views/xq83-jr8c/rows.csv?accessType=DOWNLOAD
* **Date accessed:** 2026-09-23 (portal shows rows last updated 2025-02-05)
* **File:** `chicago_energy_benchmarking.csv` — 28,329 rows × 30 columns, ~6.6 MB
* **Publisher description (summary):** annual whole-building energy use reported under Chicago's
  Building Energy Use Benchmarking Ordinance by buildings larger than 50,000 sq ft.

## Dataset B — Electric Vehicle Population Data (new dataset)

* **Organization:** Washington State Department of Licensing (data.wa.gov)
* **Title:** Electric Vehicle Population Data
* **Source:** https://data.wa.gov/d/f6w7-q2d2
* **Direct CSV:** https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD
* **Date accessed:** 2026-09-23 (portal shows rows last updated 2026-09-15)
* **File:** `wa_electric_vehicle_population.csv` — 299,705 rows × 16 columns, ~72 MB
* **Publisher description (summary):** battery-electric and plug-in hybrid vehicles currently registered
  through the Washington State Department of Licensing.

> The WA file is ~72 MB, so it may be left out of the submission ZIP to keep it small. To reproduce,
> download it from the direct-CSV link above and save it in this folder as
> `wa_electric_vehicle_population.csv`. Both datasets are updated by their publishers, so a later
> download may have slightly different row counts than the reports in `output/`.

## Why these two

They differ in structure, which tests generalization: A is numeric-heavy with heavy missingness and a
year column; B is categorical-heavy, nearly complete, with only one continuous numeric measure (so
correlation analysis is correctly skipped).
