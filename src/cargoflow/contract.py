"""Small, dependency-free validators for the CargoFlow route contract."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any

__all__ = ["validate_instance", "validate_matrix"]


def _fail(message: str) -> None:
    raise ValueError(message)


def _finite_number(value: Any, path: str, *, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{path} must be a number")
    if not math.isfinite(value):
        _fail(f"{path} must be finite")
    if nonnegative and value < 0:
        _fail(f"{path} must be non-negative")


def _utc_timestamp(value: Any, path: str) -> datetime:
    if not isinstance(value, str):
        _fail(f"{path} must be an UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(f"{path} must use YYYY-MM-DDTHH:MM:SSZ") from exc
    return parsed


def validate_instance(instance: Any) -> bool:
    """Validate one CargoFlow instance and return ``True`` when valid.

    Invalid data raises ``ValueError`` with the first failing field.  The
    function deliberately checks only the frozen interchange contract; it
    does not inspect solver-specific options.
    """

    if not isinstance(instance, dict):
        _fail("instance must be an object")
    required = {
        "schema_version",
        "instance_id",
        "source",
        "depot_id",
        "vehicle_capacity_cm3",
        "departure_time_utc",
        "nodes",
    }
    if set(instance) != required:
        _fail("instance fields must be exactly " + ", ".join(sorted(required)))
    if instance["schema_version"] != "cargoflow.instance.v0.1":
        _fail("instance.schema_version is unsupported")
    if not isinstance(instance["instance_id"], str) or not instance["instance_id"]:
        _fail("instance.instance_id must be a non-empty string")
    if instance["source"] != "amazon_lmrrc2021":
        _fail("instance.source is unsupported")
    if not isinstance(instance["depot_id"], str) or not instance["depot_id"]:
        _fail("instance.depot_id must be a non-empty string")
    _finite_number(instance["vehicle_capacity_cm3"], "instance.vehicle_capacity_cm3", nonnegative=True)
    _utc_timestamp(instance["departure_time_utc"], "instance.departure_time_utc")

    nodes = instance["nodes"]
    if not isinstance(nodes, list) or not nodes:
        _fail("instance.nodes must be a non-empty array")
    node_ids: list[str] = []
    depot_count = 0
    for index, node in enumerate(nodes):
        path = f"instance.nodes[{index}]"
        if not isinstance(node, dict):
            _fail(f"{path} must be an object")
        keys = {"id", "kind", "lat", "lng", "demand_cm3", "service_seconds", "time_window"}
        if set(node) != keys:
            _fail(f"{path} fields are invalid")
        if not isinstance(node["id"], str) or not node["id"]:
            _fail(f"{path}.id must be a non-empty string")
        if node["id"] in node_ids:
            _fail(f"duplicate node id: {node['id']}")
        node_ids.append(node["id"])
        if node["kind"] not in {"depot", "customer"}:
            _fail(f"{path}.kind is invalid")
        if node["kind"] == "depot":
            depot_count += 1
            if node["id"] != instance["depot_id"]:
                _fail("the depot node id must equal instance.depot_id")
            if node["demand_cm3"] != 0 or node["service_seconds"] != 0:
                _fail(f"{path} depot demand and service must be zero")
        elif node["id"] == instance["depot_id"]:
            _fail("instance.depot_id cannot be a customer")
        _finite_number(node["lat"], f"{path}.lat")
        _finite_number(node["lng"], f"{path}.lng")
        if not -90 <= node["lat"] <= 90 or not -180 <= node["lng"] <= 180:
            _fail(f"{path} coordinates are out of range")
        _finite_number(node["demand_cm3"], f"{path}.demand_cm3", nonnegative=True)
        _finite_number(node["service_seconds"], f"{path}.service_seconds", nonnegative=True)
        window = node["time_window"]
        if window is not None:
            if not isinstance(window, dict) or set(window) != {"start_utc", "end_utc"}:
                _fail(f"{path}.time_window is invalid")
            start = _utc_timestamp(window["start_utc"], f"{path}.time_window.start_utc")
            end = _utc_timestamp(window["end_utc"], f"{path}.time_window.end_utc")
            if end < start:
                _fail(f"{path}.time_window ends before it starts")
    if depot_count != 1 or nodes[0]["id"] != instance["depot_id"]:
        _fail("instance.nodes must start with exactly one depot")
    return True


def validate_matrix(matrix: Any) -> bool:
    """Validate one directed travel-time matrix and return ``True``."""

    if not isinstance(matrix, dict):
        _fail("matrix must be an object")
    required = {"schema_version", "instance_id", "node_ids", "travel_seconds"}
    if set(matrix) != required:
        _fail("matrix fields must be exactly " + ", ".join(sorted(required)))
    if matrix["schema_version"] != "cargoflow.matrix.v0.1":
        _fail("matrix.schema_version is unsupported")
    if not isinstance(matrix["instance_id"], str) or not matrix["instance_id"]:
        _fail("matrix.instance_id must be a non-empty string")
    node_ids = matrix["node_ids"]
    if not isinstance(node_ids, list) or not node_ids or any(not isinstance(x, str) or not x for x in node_ids):
        _fail("matrix.node_ids must be a non-empty array of strings")
    if len(set(node_ids)) != len(node_ids):
        _fail("matrix.node_ids must be unique")
    values = matrix["travel_seconds"]
    n = len(node_ids)
    if not isinstance(values, list) or len(values) != n:
        _fail("matrix.travel_seconds must be square")
    for i, row in enumerate(values):
        if not isinstance(row, list) or len(row) != n:
            _fail("matrix.travel_seconds must be square")
        for j, value in enumerate(row):
            _finite_number(value, f"matrix.travel_seconds[{i}][{j}]", nonnegative=True)
            if i == j and value != 0:
                _fail("matrix diagonal must be zero")
    return True
