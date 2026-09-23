# Data and tutorial

`tutorial/` contains one processed real route from the **2021 Amazon Last Mile
Routing Research Challenge Dataset**. Raw downloads and other generated datasets
remain excluded from Git.

## Included example

- Route: `RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77` (2018-07-27).
- Size: 118 customers, one depot, and a 119 × 119 directed travel-time matrix.
- `tutorial/instance.json`: stops, volume demand in cm³, service time in seconds,
  UTC service-start time windows, departure time and vehicle capacity.
- `tutorial/matrix.json`: stop IDs in the same order and travel times in seconds.

Source: [Amazon's dataset registry](https://registry.opendata.aws/amazon-last-mile-challenges/),
training `model_build_inputs/{route_data,package_data,travel_times}.json`.
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
The included data is an adaptation distributed under
[Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The upstream dataset anonymises identifiers and obfuscates coordinates; these
are not customer addresses. Registry and licence checked on 2026-09-21.

CargoFlow converts the public records with `scripts/convert_amazon_route.py`:
package volumes and service times are summed per stop, package time windows
are intersected, timestamps are standardised to UTC, and matrix rows/columns
follow the node order. Missing windows become `null`; self-travel is zero.
No customer is removed from this route, and historical visit order is not
included in the solver input. The example is taken from an existing processed
route; it is a tutorial, not a new holdout or benchmark result.

## Run from the repository root

Use Python 3.11+ and the pinned dependencies:

```bash
python -m pip install -e ".[dev]"
python scripts/solve_route.py \
  --instance data/tutorial/instance.json \
  --matrix data/tutorial/matrix.json \
  --output-dir outputs/tutorial/baseline --seed 42 --iterations 5000
```

The command solves and independently checks the route, writing `solution.json`
and `schedule.csv`. It must report `feasible: true`; exact timings vary by machine.
To replay the saved solution independently:

```bash
python scripts/solve_route.py \
  --instance data/tutorial/instance.json \
  --matrix data/tutorial/matrix.json \
  --check-solution outputs/tutorial/baseline/solution.json \
  --output-dir outputs/tutorial/audit
```

Optionally run adaptive remove-and-reinsert search in all four policy modes,
using that feasible baseline as the fixed reference. This step needs a C++17 compiler (`c++` or `CXX`); first-run
compilation is outside the per-mode search budget. Use a new output directory
on each policy run.

```bash
python scripts/run_policy_search.py \
  --instance data/tutorial/instance.json \
  --matrix data/tutorial/matrix.json \
  --seed-solution outputs/tutorial/baseline/solution.json \
  --output-dir outputs/tutorial/policies --seconds 1 --seed 42
```

The output contains four policy JSON files and `comparison.csv`. This tutorial
uses the pinned PyVRP baseline to generate the reference. For an ILS+LKH
reference followed by adaptive policies, follow the **Run ILS+LKH and adaptive
policies** section in the repository README. Both workflows use the same
included route data.

## Download and regenerate from raw data

Install the AWS CLI, then download the public dataset (about 3.34 GB):

```bash
./scripts/download_amazon_lmrrc2021.sh
python scripts/convert_amazon_route.py \
  --raw-dir data/raw/amazon_lmrrc2021 \
  --route-id RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77 \
  --output-dir data/processed/amazon_lmrrc2021/route_00143bdd
```

The download script accepts an alternative destination as its first argument.
Source: `s3://amazon-last-mile-challenges/almrrc2021/`. Some source files contain
bare `NaN` values; the converter explicitly handles missing values and rejects
invalid required fields. For the full 50-route benchmark, use the fixed route
list and commands in the repository README.
