#!/usr/bin/env python3
"""Select a fixed, station- and size-stratified Amazon route benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Hashable


DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SEED = 20260907
SIZE_BINS = ("small", "medium", "large", "very_large")


@dataclass(frozen=True)
class Candidate:
    route_id: str
    station_code: str
    customer_count: int
    size_bin: str


def _route_metadata(route_id: str, route: Any) -> tuple[str, int] | None:
    if not isinstance(route, dict):
        return None
    station_code = route.get("station_code")
    capacity = route.get("executor_capacity_cm3")
    if not isinstance(station_code, str) or not station_code:
        return None
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, (int, float))
        or not math.isfinite(capacity)
        or capacity <= 0
    ):
        return None
    if not isinstance(route.get("date_YYYY_MM_DD"), str) or not isinstance(
        route.get("departure_time_utc"), str
    ):
        return None
    stops = route.get("stops")
    if not isinstance(stops, dict) or not stops:
        return None
    stop_types = [
        stop.get("type") if isinstance(stop, dict) else None
        for stop in stops.values()
    ]
    if stop_types.count("Station") != 1 or any(
        kind not in {"Station", "Dropoff"} for kind in stop_types
    ):
        return None
    customer_count = stop_types.count("Dropoff")
    return (station_code, customer_count) if customer_count else None


def _cut_points(customer_counts: list[int]) -> tuple[int, int, int]:
    ordered = sorted(customer_counts)
    if not ordered:
        raise ValueError("no eligible routes")
    last = len(ordered) - 1
    return tuple(ordered[(last * quarter) // 4] for quarter in (1, 2, 3))


def _size_bin(customer_count: int, cuts: tuple[int, int, int]) -> str:
    for label, upper in zip(SIZE_BINS, cuts):
        if customer_count <= upper:
            return label
    return SIZE_BINS[-1]


def _allocate(
    sizes: dict[Hashable, int], total: int, *, at_least_one: bool
) -> dict[Hashable, int]:
    keys = sorted(sizes)
    if total < 0 or total > sum(sizes.values()):
        raise ValueError("quota is outside the available population")
    if at_least_one and total < len(keys):
        raise ValueError("sample size must cover every station")
    quota = {key: int(at_least_one) for key in keys}
    remaining = total - sum(quota.values())
    available = {key: sizes[key] - quota[key] for key in keys}
    weight_total = sum(available.values())
    if remaining == 0:
        return quota
    if weight_total <= 0:
        raise ValueError("not enough routes to allocate the requested sample")
    remainders: list[tuple[int, Hashable]] = []
    for key in keys:
        numerator = remaining * available[key]
        addition, remainder = divmod(numerator, weight_total)
        quota[key] += addition
        remainders.append((remainder, key))
    unassigned = total - sum(quota.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if unassigned == 0:
            break
        if quota[key] < sizes[key]:
            quota[key] += 1
            unassigned -= 1
    if unassigned:
        raise ValueError("quota allocation did not reach the requested sample size")
    return quota


def select_routes(
    routes: dict[str, Any], sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED
) -> tuple[list[Candidate], tuple[int, int, int], int]:
    metadata = []
    for route_id in sorted(routes):
        parsed = _route_metadata(route_id, routes[route_id])
        if parsed:
            metadata.append((route_id, *parsed))
    cuts = _cut_points([row[2] for row in metadata])
    candidates = [
        Candidate(route_id, station, count, _size_bin(count, cuts))
        for route_id, station, count in metadata
    ]
    by_station: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_station[candidate.station_code].append(candidate)
    station_quota = _allocate(
        {station: len(values) for station, values in by_station.items()},
        sample_size,
        at_least_one=True,
    )
    rng = random.Random(seed)
    selected: list[Candidate] = []
    for station in sorted(by_station):
        by_size: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in by_station[station]:
            by_size[candidate.size_bin].append(candidate)
        size_quota = _allocate(
            {label: len(values) for label, values in by_size.items()},
            station_quota[station],
            at_least_one=False,
        )
        for label in SIZE_BINS:
            group = sorted(by_size.get(label, []), key=lambda item: item.route_id)
            rng.shuffle(group)
            selected.extend(group[: size_quota.get(label, 0)])
    if len(selected) != sample_size or len({item.route_id for item in selected}) != sample_size:
        raise RuntimeError("selection is not a unique sample of the requested size")
    return selected, cuts, len(candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    try:
        with args.route_data.open(encoding="utf-8") as stream:
            routes = json.load(stream, parse_constant=lambda _value: None)
        if not isinstance(routes, dict):
            raise ValueError("route data must be a top-level object")
        selected, cuts, eligible = select_routes(routes, args.sample_size, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(f"{candidate.route_id}\n" for candidate in selected),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "eligible_routes": eligible,
                    "selected_routes": len(selected),
                    "stations": len({item.station_code for item in selected}),
                    "customer_count_cut_points": cuts,
                    "seed": args.seed,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
