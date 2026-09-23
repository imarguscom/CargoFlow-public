#!/usr/bin/env python3
"""Convert one Amazon last-mile route into the minimal CargoFlow contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cargoflow.contract import validate_instance, validate_matrix  # noqa: E402

DEFAULT_ROUTE_ID = "RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77"
DEFAULT_SHORT_NAME = "route_00143bdd"


class ConversionError(ValueError):
    """Raised when raw data cannot satisfy the frozen contract."""


class _ByteReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.buffer = b""
        self.position = 0

    def read_byte(self) -> int | None:
        if self.position >= len(self.buffer):
            self.buffer = self.stream.read(1024 * 1024)
            self.position = 0
            if not self.buffer:
                return None
        value = self.buffer[self.position]
        self.position += 1
        return value

    def nonspace(self) -> int | None:
        value = self.read_byte()
        while value is not None and value in b" \t\r\n":
            value = self.read_byte()
        return value


def _json_member(path: Path, wanted: str) -> Any:
    """Read only the requested top-level member of a JSON object.

    Values are buffered one member at a time, so the 1.7 GB travel file is
    never loaded as a whole. Python's decoder maps bare NaN/Infinity to None,
    allowing later validation to reject them explicitly.
    """

    def read_string(reader: _ByteReader, first: int) -> str:
        if first != ord('"'):
            raise ConversionError(f"{path}: expected a JSON member name")
        raw = bytearray(b'"')
        escaped = False
        while True:
            value = reader.read_byte()
            if value is None:
                raise ConversionError(f"{path}: unterminated JSON string")
            raw.append(value)
            if escaped:
                escaped = False
            elif value == ord('\\'):
                escaped = True
            elif value == ord('"'):
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise ConversionError(f"{path}: invalid JSON member name") from exc

    def read_value(reader: _ByteReader, first: int, capture: bool) -> bytes | None:
        raw = bytearray([first]) if capture else None
        if first in (ord('{'), ord('[')):
            depth = 1
            in_string = False
            escaped = False
            while depth:
                value = reader.read_byte()
                if value is None:
                    raise ConversionError(f"{path}: incomplete JSON value")
                if raw is not None:
                    raw.append(value)
                if in_string:
                    if escaped:
                        escaped = False
                    elif value == ord('\\'):
                        escaped = True
                    elif value == ord('"'):
                        in_string = False
                elif value == ord('"'):
                    in_string = True
                elif value in (ord('{'), ord('[')):
                    depth += 1
                elif value in (ord('}'), ord(']')):
                    depth -= 1
            return bytes(raw) if raw is not None else None
        while True:
            value = reader.read_byte()
            if value is None or value in b" \t\r\n,}":
                if value is not None and value not in b" \t\r\n":
                    reader.position -= 1
                return bytes(raw) if raw is not None else None
            if raw is not None:
                raw.append(value)

    with path.open("rb") as stream:
        reader = _ByteReader(stream)
        if reader.nonspace() != ord('{'):
            raise ConversionError(f"{path}: top level must be an object")
        while True:
            first = reader.nonspace()
            if first == ord('}'):
                break
            if first is None:
                raise ConversionError(f"{path}: incomplete top-level object")
            name = read_string(reader, first)
            if reader.nonspace() != ord(':'):
                raise ConversionError(f"{path}: missing colon after {name}")
            first_value = reader.nonspace()
            if first_value is None:
                raise ConversionError(f"{path}: missing value for {name}")
            raw_value = read_value(reader, first_value, name == wanted)
            if name == wanted:
                try:
                    return json.loads(raw_value or b"null", parse_constant=lambda _value: None)
                except json.JSONDecodeError as exc:
                    raise ConversionError(f"{path}: invalid value for {wanted}") from exc
            separator = reader.nonspace()
            if separator == ord('}'):
                break
            if separator != ord(','):
                raise ConversionError(f"{path}: invalid top-level separator")
    raise ConversionError(f"{path}: route {wanted!r} was not found")


def _number(value: Any, label: str, *, nonnegative: bool = True) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConversionError(f"{label} must be finite numeric data")
    if nonnegative and value < 0:
        raise ConversionError(f"{label} must be non-negative")
    return value


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ConversionError(f"{label} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConversionError(f"{label} is not parseable") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _input_dir(raw_dir: Path) -> Path:
    nested = raw_dir / "almrrc2021-data-training" / "model_build_inputs"
    direct = raw_dir / "model_build_inputs"
    if nested.is_dir():
        return nested
    if direct.is_dir():
        return direct
    raise ConversionError(f"raw directory does not contain model_build_inputs: {raw_dir}")


def convert_records(
    route_id: str,
    route: Any,
    packages: Any,
    travel: Any,
    actual: Any | None,
    output_dir: Path,
    *,
    emit_summary: bool = True,
) -> tuple[int, int]:
    """Convert already-loaded records for one route and write the contract."""

    if not all(isinstance(value, dict) for value in (route, packages, travel)) or (actual is not None and not isinstance(actual, dict)):
        raise ConversionError(f"route {route_id} has an invalid top-level record")

    stops = route.get("stops")
    if not isinstance(stops, dict):
        raise ConversionError(f"route {route_id} must contain stops")
    depot_ids = [
        key
        for key, value in stops.items()
        if isinstance(value, dict) and value.get("type") == "Station"
    ]
    if len(depot_ids) != 1:
        raise ConversionError(f"route {route_id} must contain exactly one Station depot")
    depot_id = depot_ids[0]
    node_ids = [depot_id] + sorted(
        key
        for key, value in stops.items()
        if key != depot_id
        and isinstance(value, dict)
        and value.get("type") == "Dropoff"
    )
    if len(node_ids) != len(stops):
        raise ConversionError("route contains a stop that is neither Station nor Dropoff")
    if not isinstance(packages, dict) or not isinstance(travel, dict):
        raise ConversionError("route package or travel data is not an object")

    departure = _parse_utc(f"{route.get('date_YYYY_MM_DD')} {route.get('departure_time_utc')}", "route departure")
    capacity = _number(route.get("executor_capacity_cm3"), "executor_capacity_cm3")
    nodes: list[dict[str, Any]] = []
    valid_window_customers = 0
    for node_id in node_ids:
        stop = stops[node_id]
        lat = _number(stop.get("lat"), f"stop {node_id} latitude", nonnegative=False)
        lng = _number(stop.get("lng"), f"stop {node_id} longitude", nonnegative=False)
        if node_id == depot_id:
            nodes.append({"id": depot_id, "kind": "depot", "lat": lat, "lng": lng, "demand_cm3": 0, "service_seconds": 0, "time_window": None})
            continue
        stop_packages = packages.get(node_id)
        if not isinstance(stop_packages, dict) or not stop_packages:
            raise ConversionError(f"stop {node_id} has no package records")
        demand: int | float = 0
        service: int | float = 0
        windows: list[tuple[datetime, datetime]] = []
        for package_id, package in stop_packages.items():
            if not isinstance(package, dict):
                raise ConversionError(f"package {package_id} is not an object")
            dimensions = package.get("dimensions")
            if not isinstance(dimensions, dict):
                raise ConversionError(f"package {package_id} has no dimensions")
            product = 1
            for dimension in ("depth_cm", "height_cm", "width_cm"):
                product *= _number(dimensions.get(dimension), f"package {package_id} {dimension}")
            demand += product
            service += _number(package.get("planned_service_time_seconds"), f"package {package_id} service")
            window = package.get("time_window")
            if not isinstance(window, dict):
                raise ConversionError(f"package {package_id} has an invalid time window")
            start_value = window.get("start_time_utc")
            end_value = window.get("end_time_utc")
            if start_value is None and end_value is None:
                continue
            if start_value is None or end_value is None:
                raise ConversionError(f"package {package_id} has an incomplete time window")
            windows.append((_parse_utc(start_value, f"package {package_id} window start"), _parse_utc(end_value, f"package {package_id} window end")))
        time_window = None
        if windows:
            start = max(pair[0] for pair in windows)
            end = min(pair[1] for pair in windows)
            if end < start:
                raise ConversionError(f"stop {node_id} has an empty time-window intersection")
            time_window = {"start_utc": _format_utc(start), "end_utc": _format_utc(end)}
            valid_window_customers += 1
        nodes.append({"id": node_id, "kind": "customer", "lat": lat, "lng": lng, "demand_cm3": demand, "service_seconds": service, "time_window": time_window})

    travel_rows = travel
    if set(travel_rows) != set(node_ids):
        raise ConversionError("travel-time row keys do not match converted node IDs")
    travel_seconds: list[list[int | float]] = []
    for origin in node_ids:
        row = travel_rows[origin]
        if not isinstance(row, dict) or set(row) != set(node_ids):
            raise ConversionError(f"travel-time columns for {origin} do not match node IDs")
        values = []
        for destination in node_ids:
            value = _number(row[destination], f"travel {origin}->{destination}")
            values.append(0 if origin == destination else value)
        travel_seconds.append(values)

    if actual is not None:
        actual_order = actual.get("actual")
        expected_customers = node_ids[1:]
        if not isinstance(actual_order, dict) or set(actual_order) != set(node_ids) or actual_order.get(depot_id) != 0:
            raise ConversionError("actual sequence keys or depot position do not match")
        positions = [actual_order[node_id] for node_id in expected_customers]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in positions) or sorted(positions) != list(range(1, len(expected_customers) + 1)):
            raise ConversionError("actual sequence customer positions are not a complete permutation")

    instance_id = f"amazon_lmrrc2021:{route_id}"
    instance = {"schema_version": "cargoflow.instance.v0.1", "instance_id": instance_id, "source": "amazon_lmrrc2021", "depot_id": depot_id, "vehicle_capacity_cm3": capacity, "departure_time_utc": _format_utc(departure), "nodes": nodes}
    matrix = {"schema_version": "cargoflow.matrix.v0.1", "instance_id": instance_id, "node_ids": node_ids, "travel_seconds": travel_seconds}
    validate_instance(instance)
    validate_matrix(matrix)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (("instance.json", instance), ("matrix.json", matrix)):
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    if emit_summary:
        print(f"route_id={route_id}")
        print(f"nodes={len(nodes)} customers={len(nodes) - 1}")
        print(f"matrix={len(node_ids)}x{len(node_ids)}")
        print(f"customers_with_time_window={valid_window_customers}")
    return len(nodes), valid_window_customers


def convert(raw_dir: Path, output_dir: Path, route_id: str = DEFAULT_ROUTE_ID) -> tuple[int, int]:
    inputs = _input_dir(raw_dir)
    return convert_records(
        route_id,
        _json_member(inputs / "route_data.json", route_id),
        _json_member(inputs / "package_data.json", route_id),
        _json_member(inputs / "travel_times.json", route_id),
        _json_member(inputs / "actual_sequences.json", route_id),
        output_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--route-id", default=DEFAULT_ROUTE_ID)
    args = parser.parse_args()
    try:
        convert(args.raw_dir, args.output_dir, args.route_id)
    except (ConversionError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
