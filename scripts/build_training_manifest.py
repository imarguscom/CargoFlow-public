#!/usr/bin/env python3
"""Stream the four Amazon build-input objects into converted route manifests.

The source files are top-level JSON objects keyed by route ID.  This command
keeps only one route's route/package/travel/actual records live at a time,
writes converted contracts and a small actual-sequence sidecar under an
ignored output directory, and records every conversion error instead of
silently substituting a route.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
try:
    import resource
except ModuleNotFoundError:  # Windows has no POSIX resource module.
    resource = None
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONVERTER_PATH = ROOT / "scripts" / "convert_amazon_route.py"
SPEC = importlib.util.spec_from_file_location("convert_amazon_route_manifest", CONVERTER_PATH)
if not SPEC or not SPEC.loader:
    raise RuntimeError(f"cannot load converter: {CONVERTER_PATH}")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _read_string(reader: converter._ByteReader, first: int, path: Path) -> str:
    if first != ord('"'):
        raise ValueError(f"{path}: expected a JSON object member name")
    raw = bytearray(b'"')
    escaped = False
    while True:
        value = reader.read_byte()
        if value is None:
            raise ValueError(f"{path}: unterminated JSON member name")
        raw.append(value)
        if escaped:
            escaped = False
        elif value == ord('\\'):
            escaped = True
        elif value == ord('"'):
            return json.loads(raw.decode("utf-8"))


def _read_value(reader: converter._ByteReader, first: int, path: Path) -> bytes:
    raw = bytearray([first])
    if first in (ord('{'), ord('[')):
        depth = 1
        in_string = False
        escaped = False
        while depth:
            value = reader.read_byte()
            if value is None:
                raise ValueError(f"{path}: incomplete JSON value")
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
        return bytes(raw)
    while True:
        value = reader.read_byte()
        if value is None or value in b" \t\r\n,}":
            if value is not None and value not in b" \t\r\n":
                reader.position -= 1
            return bytes(raw)
        raw.append(value)


def iter_object_members(path: Path) -> Iterator[tuple[str, Any]]:
    """Yield top-level JSON object members while retaining one value."""

    with path.open("rb") as stream:
        reader = converter._ByteReader(stream)
        if reader.nonspace() != ord('{'):
            raise ValueError(f"{path}: top level must be an object")
        while True:
            first = reader.nonspace()
            if first == ord('}'):
                return
            if first is None:
                raise ValueError(f"{path}: incomplete top-level object")
            name = _read_string(reader, first, path)
            if reader.nonspace() != ord(':'):
                raise ValueError(f"{path}: missing colon after {name}")
            first_value = reader.nonspace()
            if first_value is None:
                raise ValueError(f"{path}: missing value for {name}")
            raw = _read_value(reader, first_value, path)
            yield name, json.loads(raw, parse_constant=lambda _value: None)
            separator = reader.nonspace()
            if separator == ord('}'):
                return
            if separator != ord(','):
                raise ValueError(f"{path}: invalid top-level separator")


def _route_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith('#')}


def build(raw_dir: Path, output_dir: Path, *, benchmark_routes: Path | None = None, route_limit: int | None = None) -> dict[str, Any]:
    inputs = converter._input_dir(raw_dir)
    wanted = _route_ids(benchmark_routes) if benchmark_routes else None
    iterators = {
        label: iter_object_members(inputs / filename)
        for label, filename in {
            "route": "route_data.json",
            "packages": "package_data.json",
            "travel": "travel_times.json",
            "actual": "actual_sequences.json",
        }.items()
    }
    manifest: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen = 0
    peak_rss = 0
    while True:
        records: dict[str, Any] = {}
        finished = 0
        for label, iterator in iterators.items():
            try:
                route_id, record = next(iterator)
                records[label] = (route_id, record)
            except StopIteration:
                finished += 1
        if finished:
            if finished != len(iterators):
                raise ValueError("source files have different route-key counts/order")
            break
        route_ids = {value[0] for value in records.values()}
        if len(route_ids) != 1:
            raise ValueError(f"source files disagree on route key near record {seen + 1}")
        route_id = next(iter(route_ids))
        seen += 1
        if wanted is not None and route_id not in wanted:
            continue
        if route_limit is not None and len(manifest) >= route_limit:
            break
        route, packages, travel, actual = (records[label][1] for label in ("route", "packages", "travel", "actual"))
        try:
            station = str(route.get("station_code", ""))
            date = str(route.get("date_YYYY_MM_DD", ""))
            zone_by_stop = {
                str(stop_id): str(stop.get("zone_id"))
                for stop_id, stop in (route.get("stops") or {}).items()
                if isinstance(stop, dict) and isinstance(stop.get("zone_id"), str) and stop.get("zone_id")
            }
            count = sum(1 for value in (route.get("stops") or {}).values() if isinstance(value, dict) and value.get("type") == "Dropoff")
            route_dir = output_dir / "processed" / route_id
            converter.convert_records(route_id, route, packages, travel, actual, route_dir, emit_summary=False)
            actual_path = output_dir / "actual" / f"{route_id}.json"
            actual_path.parent.mkdir(parents=True, exist_ok=True)
            actual_path.write_text(json.dumps(actual, allow_nan=False) + "\n", encoding="utf-8")
            manifest.append({
                "route_id": route_id,
                "station_code": station,
                "date_YYYY_MM_DD": date,
                "customer_count": count,
                "zone_by_stop": zone_by_stop,
                "instance_path": str((route_dir / "instance.json").resolve()),
                "matrix_path": str((route_dir / "matrix.json").resolve()),
                "actual_path": str(actual_path.resolve()),
            })
        except (OSError, ValueError, TypeError, KeyError, converter.ConversionError) as exc:
            errors.append({"route_id": route_id, "error": str(exc)})
        if resource is not None:
            peak_rss = max(peak_rss, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if len(manifest) and len(manifest) % 100 == 0:
            print(f"converted={len(manifest)} seen={seen}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "source_route_records_seen": seen,
        "converted_routes": len(manifest),
        "conversion_errors": len(errors),
        "errors": errors,
        "peak_rss_units": peak_rss if resource is not None else None,
        "peak_rss_unit": "bytes on macOS; kilobytes on Linux; unavailable on Windows",
        "streaming": True,
    }
    (output_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-routes", type=Path)
    parser.add_argument("--route-limit", type=int)
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.raw_dir, args.output_dir, benchmark_routes=args.benchmark_routes, route_limit=args.route_limit), indent=2))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
