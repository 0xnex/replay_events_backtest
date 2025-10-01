"""Utilities for constructing an in-memory pool state from snapshot files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _twos_complement(value: int, bits: int) -> int:
    """Interpret ``value`` as a signed integer encoded with two's complement."""

    if bits <= 0:
        return value
    threshold = 1 << (bits - 1)
    if value >= threshold:
        return value - (1 << bits)
    return value


def _parse_move_int(value: Any, *, fallback_bits: Optional[int] = None, signed: Optional[bool] = None) -> int:
    """Best-effort parser for Sui Move integer representations."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if signed and fallback_bits:
            return _twos_complement(value, fallback_bits)
        return value
    if isinstance(value, str):
        number = int(value)
        if signed and fallback_bits:
            return _twos_complement(number, fallback_bits)
        return number

    if isinstance(value, Mapping):
        if "bits" in value:
            return _parse_move_int(value["bits"], fallback_bits=fallback_bits, signed=signed)
        if "fields" in value and isinstance(value["fields"], Mapping):
            fields = value["fields"]

            if "bits" in fields:
                number = int(fields["bits"])
                type_tag = value.get("type")
                bits = fallback_bits
                signed_flag = signed

                if isinstance(type_tag, str):
                    lowered = type_tag.lower()
                    if "::i128::" in lowered:
                        bits, signed_flag = 128, True
                    elif "::u128::" in lowered:
                        bits, signed_flag = 128, False
                    elif "::i64::" in lowered:
                        bits, signed_flag = 64, True
                    elif "::u64::" in lowered:
                        bits, signed_flag = 64, False
                    elif "::i32::" in lowered:
                        bits, signed_flag = 32, True
                    elif "::u32::" in lowered:
                        bits, signed_flag = 32, False
                    elif "::i16::" in lowered:
                        bits, signed_flag = 16, True
                    elif "::u16::" in lowered:
                        bits, signed_flag = 16, False
                    elif "::i8::" in lowered:
                        bits, signed_flag = 8, True
                    elif "::u8::" in lowered:
                        bits, signed_flag = 8, False

                if signed_flag is None:
                    signed_flag = False
                if bits is None and signed_flag:
                    bits = fallback_bits

                if signed_flag and bits:
                    return _twos_complement(number, bits)
                return number

            if "value" in fields:
                return _parse_move_int(fields["value"], fallback_bits=fallback_bits, signed=signed)

        if "value" in value:
            return _parse_move_int(value["value"], fallback_bits=fallback_bits, signed=signed)

    raise TypeError(f"Unsupported Move integer representation: {value!r}")


def _extract_fields(node: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``fields`` dictionary from a Move object container."""

    if not isinstance(node, Mapping):
        return {}
    content = node.get("content") or node.get("data")
    if isinstance(content, Mapping) and isinstance(content.get("fields"), Mapping):
        return content["fields"]
    if isinstance(node.get("fields"), Mapping):
        return node["fields"]
    return {}


def _get_nested(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


@dataclass(slots=True)
class TickInfo:
    index: int
    liquidity_gross: int
    liquidity_net: int
    fee_growth_outside_x: int
    fee_growth_outside_y: int
    reward_growths_outside: List[int]
    seconds_outside: int
    seconds_per_liquidity_outside: int
    tick_cumulative_outside: int

    @classmethod
    def from_move(cls, index: int, raw_fields: Mapping[str, Any]) -> "TickInfo":
        return cls(
            index=index,
            liquidity_gross=_parse_move_int(raw_fields.get("liquidity_gross", 0)),
            liquidity_net=_parse_move_int(raw_fields.get("liquidity_net", 0), fallback_bits=128, signed=True),
            fee_growth_outside_x=_parse_move_int(raw_fields.get("fee_growth_outside_x", 0)),
            fee_growth_outside_y=_parse_move_int(raw_fields.get("fee_growth_outside_y", 0)),
            reward_growths_outside=[_parse_move_int(val) for val in raw_fields.get("reward_growths_outside", [])],
            seconds_outside=_parse_move_int(raw_fields.get("seconds_out_side", 0)),
            seconds_per_liquidity_outside=_parse_move_int(raw_fields.get("seconds_per_liquidity_out_side", 0)),
            tick_cumulative_outside=_parse_move_int(
                raw_fields.get("tick_cumulative_out_side", 0), fallback_bits=64, signed=True
            ),
        )


@dataclass(slots=True)
class PoolState:
    pool_id: str
    version: int
    sqrt_price: int
    liquidity: int
    tick_current_index: int
    tick_spacing: int
    fee_growth_global_x: int
    fee_growth_global_y: int
    protocol_fee_x: int
    protocol_fee_y: int
    swap_fee_rate: int
    flash_loan_fee_rate: int
    protocol_fee_share: int
    protocol_flash_loan_fee_share: int
    reserve_x: int
    reserve_y: int
    coin_type_x: Optional[str]
    coin_type_y: Optional[str]
    reward_infos: List[Mapping[str, Any]]
    observation: Mapping[str, Any]
    tick_bitmap_handle: Optional[str]
    tick_bitmap_size: Optional[int]
    observation_cardinality: int
    observation_cardinality_next: int
    observation_index: int
    tick_handle_id: Optional[str]
    ticks: Dict[int, TickInfo] = field(default_factory=dict)
    raw_snapshot: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the pool state."""

        return {
            "pool_id": self.pool_id,
            "version": self.version,
            "sqrt_price": self.sqrt_price,
            "liquidity": self.liquidity,
            "tick_current_index": self.tick_current_index,
            "tick_spacing": self.tick_spacing,
            "fee_growth_global_x": self.fee_growth_global_x,
            "fee_growth_global_y": self.fee_growth_global_y,
            "protocol_fee_x": self.protocol_fee_x,
            "protocol_fee_y": self.protocol_fee_y,
            "swap_fee_rate": self.swap_fee_rate,
            "flash_loan_fee_rate": self.flash_loan_fee_rate,
            "protocol_fee_share": self.protocol_fee_share,
            "protocol_flash_loan_fee_share": self.protocol_flash_loan_fee_share,
            "reserve_x": self.reserve_x,
            "reserve_y": self.reserve_y,
            "coin_type_x": self.coin_type_x,
            "coin_type_y": self.coin_type_y,
            "reward_infos": self.reward_infos,
            "observation": self.observation,
            "tick_bitmap_handle": self.tick_bitmap_handle,
            "tick_bitmap_size": self.tick_bitmap_size,
            "observation_cardinality": self.observation_cardinality,
            "observation_cardinality_next": self.observation_cardinality_next,
            "observation_index": self.observation_index,
            "tick_handle_id": self.tick_handle_id,
            "ticks": {
                str(index): {
                    "index": tick.index,
                    "liquidity_gross": tick.liquidity_gross,
                    "liquidity_net": tick.liquidity_net,
                    "fee_growth_outside_x": tick.fee_growth_outside_x,
                    "fee_growth_outside_y": tick.fee_growth_outside_y,
                    "reward_growths_outside": tick.reward_growths_outside,
                    "seconds_outside": tick.seconds_outside,
                    "seconds_per_liquidity_outside": tick.seconds_per_liquidity_outside,
                    "tick_cumulative_outside": tick.tick_cumulative_outside,
                }
                for index, tick in sorted(self.ticks.items())
            },
            "raw_snapshot": self.raw_snapshot,
        }


def _extract_tick_handle_id(snapshot: Mapping[str, Any], pool_fields: Mapping[str, Any]) -> Optional[str]:
    if isinstance(snapshot.get("tick_handle_id"), str):
        return snapshot["tick_handle_id"]

    ticks_field = pool_fields.get("ticks") if isinstance(pool_fields, Mapping) else None
    ticks_handle = _get_nested(ticks_field, "fields", "id", "id")
    if isinstance(ticks_handle, str):
        return ticks_handle
    return None


def _parse_reward_infos(raw_infos: Iterable[Any]) -> List[Mapping[str, Any]]:
    parsed: List[Mapping[str, Any]] = []
    for info in raw_infos or []:
        fields = _extract_fields(info)
        if not fields:
            continue
        parsed.append(
            {
                "reward_coin_type": _get_nested(fields, "reward_coin_type", "fields", "name"),
                "reward_growth_global": _parse_move_int(fields.get("reward_growth_global", 0)),
                "reward_per_seconds": _parse_move_int(fields.get("reward_per_seconds", 0)),
                "total_reward": _parse_move_int(fields.get("total_reward", 0)),
                "total_reward_allocated": _parse_move_int(fields.get("total_reward_allocated", 0)),
                "last_update_time": _parse_move_int(fields.get("last_update_time", 0)),
                "ended_at_seconds": _parse_move_int(fields.get("ended_at_seconds", 0)),
            }
        )
    return parsed


def _parse_observation(raw_observations: Iterable[Any], tx_timestamp: Optional[int] = None) -> Mapping[str, Any]:
    observations = list(raw_observations or [])
    if not observations:
        return {}

    latest = _extract_fields(observations[0])
    if not latest:
        return {}

    # Use transaction timestamp if provided, otherwise fall back to observation timestamp
    timestamp_s = tx_timestamp if tx_timestamp is not None else _parse_move_int(latest.get("timestamp_s", 0))

    return {
        "initialized": bool(latest.get("initialized")),
        "seconds_per_liquidity_cumulative": _parse_move_int(latest.get("seconds_per_liquidity_cumulative", 0)),
        "tick_cumulative": _parse_move_int(latest.get("tick_cumulative"), fallback_bits=64, signed=True),
        "timestamp_s": timestamp_s,
    }


def _parse_tick_entries(snapshot: Mapping[str, Any]) -> Dict[int, TickInfo]:
    ticks: Dict[int, TickInfo] = {}
    for entry in snapshot.get("tick_map_objects", []):
        descriptor = entry.get("descriptor", {})
        name_node = descriptor.get("name") or descriptor.get("value")
        tick_index = None
        if isinstance(name_node, Mapping):
            if "value" in name_node:
                tick_index = _parse_move_int(name_node["value"], fallback_bits=32, signed=True)
            elif "fields" in name_node:
                tick_index = _parse_move_int(name_node["fields"], fallback_bits=32, signed=True)
            elif "bits" in name_node:
                tick_index = _parse_move_int(name_node["bits"], fallback_bits=32, signed=True)
        elif name_node is not None:
            tick_index = _parse_move_int(name_node, fallback_bits=32, signed=True)

        if tick_index is None:
            continue

        object_node = entry.get("object") or {}
        object_fields = _extract_fields(object_node)
        if not object_fields:
            # Some snapshots store the value under descriptor.value.fields
            value_node = descriptor.get("value") if isinstance(descriptor, Mapping) else None
            object_fields = _extract_fields(value_node or {})

        value_fields = _get_nested(object_fields, "value", "fields")
        if not isinstance(value_fields, Mapping):
            continue

        ticks[tick_index] = TickInfo.from_move(tick_index, value_fields)

    return ticks


def _extract_tx_timestamp_from_snapshot(data: Mapping[str, Any]) -> Optional[int]:
    """Extract transaction timestamp from snapshot metadata if available."""
    # Check if we have transaction data in the snapshot
    if "tx_data" in data:
        tx_data = data["tx_data"]
        if "timestampMs" in tx_data:
            return int(tx_data["timestampMs"]) // 1000
        if "timestamp" in tx_data:
            return int(tx_data["timestamp"])
    
    # If no transaction data, return None to use observation timestamp
    return None


def load_pool_state_with_tx_timestamp(path: str | Path, tx_timestamp: Optional[int] = None) -> PoolState:
    """Load a :class:`PoolState` instance from a snapshot JSON file with explicit transaction timestamp.
    
    This is useful when you want to override the pool state timestamp with a specific transaction timestamp,
    such as the timestamp of the transaction that created the pool.
    """
    snapshot_path = Path(path)
    data = json.loads(snapshot_path.read_text())

    pool_object = data.get("pool_object") or {}
    pool_fields = _extract_fields(pool_object)
    if not pool_fields:
        raise ValueError("Pool snapshot is missing Move object fields")

    sqrt_price = _parse_move_int(pool_fields.get("sqrt_price", 0))
    liquidity = _parse_move_int(pool_fields.get("liquidity", 0))
    tick_spacing = _parse_move_int(pool_fields.get("tick_spacing", 0))
    tick_index = _parse_move_int(pool_fields.get("tick_index", {}), fallback_bits=32, signed=True)

    tick_bitmap_handle = _get_nested(pool_fields, "tick_bitmap", "fields", "id", "id")
    tick_bitmap_size = _parse_move_int(_get_nested(pool_fields, "tick_bitmap", "fields", "size") or 0)

    tick_handle_id = _extract_tick_handle_id(data, pool_fields)

    return PoolState(
        pool_id=str(data.get("pool_id")),
        version=int(data.get("pool_version", 0)),
        sqrt_price=sqrt_price,
        liquidity=liquidity,
        tick_current_index=tick_index,
        tick_spacing=tick_spacing,
        fee_growth_global_x=_parse_move_int(pool_fields.get("fee_growth_global_x", 0)),
        fee_growth_global_y=_parse_move_int(pool_fields.get("fee_growth_global_y", 0)),
        protocol_fee_x=_parse_move_int(pool_fields.get("protocol_fee_x", 0)),
        protocol_fee_y=_parse_move_int(pool_fields.get("protocol_fee_y", 0)),
        swap_fee_rate=_parse_move_int(pool_fields.get("swap_fee_rate", 0)),
        flash_loan_fee_rate=_parse_move_int(pool_fields.get("flash_loan_fee_rate", 0)),
        protocol_fee_share=_parse_move_int(pool_fields.get("protocol_fee_share", 0)),
        protocol_flash_loan_fee_share=_parse_move_int(pool_fields.get("protocol_flash_loan_fee_share", 0)),
        reserve_x=_parse_move_int(pool_fields.get("reserve_x", 0)),
        reserve_y=_parse_move_int(pool_fields.get("reserve_y", 0)),
        coin_type_x=_get_nested(pool_fields, "type_x", "fields", "name"),
        coin_type_y=_get_nested(pool_fields, "type_y", "fields", "name"),
        reward_infos=_parse_reward_infos(pool_fields.get("reward_infos", [])),
        observation=_parse_observation(pool_fields.get("observations", []), tx_timestamp),
        tick_bitmap_handle=str(tick_bitmap_handle) if tick_bitmap_handle else None,
        tick_bitmap_size=int(tick_bitmap_size) if tick_bitmap_size is not None else None,
        observation_cardinality=_parse_move_int(pool_fields.get("observation_cardinality", 0)),
        observation_cardinality_next=_parse_move_int(pool_fields.get("observation_cardinality_next", 0)),
        observation_index=_parse_move_int(pool_fields.get("observation_index", 0)),
        tick_handle_id=tick_handle_id,
        ticks=_parse_tick_entries(data),
        raw_snapshot=data,
    )


def load_pool_state(path: str | Path) -> PoolState:
    """Load a :class:`PoolState` instance from a snapshot JSON file."""

    snapshot_path = Path(path)
    data = json.loads(snapshot_path.read_text())

    pool_object = data.get("pool_object") or {}
    pool_fields = _extract_fields(pool_object)
    if not pool_fields:
        raise ValueError("Pool snapshot is missing Move object fields")

    # Extract transaction timestamp from snapshot metadata
    tx_timestamp = _extract_tx_timestamp_from_snapshot(data)

    sqrt_price = _parse_move_int(pool_fields.get("sqrt_price", 0))
    liquidity = _parse_move_int(pool_fields.get("liquidity", 0))
    tick_spacing = _parse_move_int(pool_fields.get("tick_spacing", 0))
    tick_index = _parse_move_int(pool_fields.get("tick_index", {}), fallback_bits=32, signed=True)

    tick_bitmap_handle = _get_nested(pool_fields, "tick_bitmap", "fields", "id", "id")
    tick_bitmap_size = _parse_move_int(_get_nested(pool_fields, "tick_bitmap", "fields", "size") or 0)

    tick_handle_id = _extract_tick_handle_id(data, pool_fields)

    return PoolState(
        pool_id=str(data.get("pool_id")),
        version=int(data.get("pool_version", 0)),
        sqrt_price=sqrt_price,
        liquidity=liquidity,
        tick_current_index=tick_index,
        tick_spacing=tick_spacing,
        fee_growth_global_x=_parse_move_int(pool_fields.get("fee_growth_global_x", 0)),
        fee_growth_global_y=_parse_move_int(pool_fields.get("fee_growth_global_y", 0)),
        protocol_fee_x=_parse_move_int(pool_fields.get("protocol_fee_x", 0)),
        protocol_fee_y=_parse_move_int(pool_fields.get("protocol_fee_y", 0)),
        swap_fee_rate=_parse_move_int(pool_fields.get("swap_fee_rate", 0)),
        flash_loan_fee_rate=_parse_move_int(pool_fields.get("flash_loan_fee_rate", 0)),
        protocol_fee_share=_parse_move_int(pool_fields.get("protocol_fee_share", 0)),
        protocol_flash_loan_fee_share=_parse_move_int(pool_fields.get("protocol_flash_loan_fee_share", 0)),
        reserve_x=_parse_move_int(pool_fields.get("reserve_x", 0)),
        reserve_y=_parse_move_int(pool_fields.get("reserve_y", 0)),
        coin_type_x=_get_nested(pool_fields, "type_x", "fields", "name"),
        coin_type_y=_get_nested(pool_fields, "type_y", "fields", "name"),
        reward_infos=_parse_reward_infos(pool_fields.get("reward_infos", [])),
        observation=_parse_observation(pool_fields.get("observations", []), tx_timestamp),
        tick_bitmap_handle=str(tick_bitmap_handle) if tick_bitmap_handle else None,
        tick_bitmap_size=int(tick_bitmap_size) if tick_bitmap_size is not None else None,
        observation_cardinality=_parse_move_int(pool_fields.get("observation_cardinality", 0)),
        observation_cardinality_next=_parse_move_int(pool_fields.get("observation_cardinality_next", 0)),
        observation_index=_parse_move_int(pool_fields.get("observation_index", 0)),
        tick_handle_id=tick_handle_id,
        ticks=_parse_tick_entries(data),
        raw_snapshot=data,
    )


def load_processed_pool_state(path: str | Path) -> PoolState:
    """Load a :class:`PoolState` instance from a processed state JSON file."""
    
    state_path = Path(path)
    data = json.loads(state_path.read_text())
    
    # Convert ticks data back to TickInfo objects
    ticks = {}
    for tick_index_str, tick_data in data.get("ticks", {}).items():
        tick_index = int(tick_index_str)
        ticks[tick_index] = TickInfo(
            index=tick_data["index"],
            liquidity_gross=tick_data["liquidity_gross"],
            liquidity_net=tick_data["liquidity_net"],
            fee_growth_outside_x=tick_data["fee_growth_outside_x"],
            fee_growth_outside_y=tick_data["fee_growth_outside_y"],
            reward_growths_outside=tick_data["reward_growths_outside"],
            seconds_outside=tick_data["seconds_outside"],
            seconds_per_liquidity_outside=tick_data["seconds_per_liquidity_outside"],
            tick_cumulative_outside=tick_data["tick_cumulative_outside"],
        )
    
    return PoolState(
        pool_id=data["pool_id"],
        version=data["version"],
        sqrt_price=data["sqrt_price"],
        liquidity=data["liquidity"],
        tick_current_index=data["tick_current_index"],
        tick_spacing=data["tick_spacing"],
        fee_growth_global_x=data["fee_growth_global_x"],
        fee_growth_global_y=data["fee_growth_global_y"],
        protocol_fee_x=data["protocol_fee_x"],
        protocol_fee_y=data["protocol_fee_y"],
        swap_fee_rate=data["swap_fee_rate"],
        flash_loan_fee_rate=data["flash_loan_fee_rate"],
        protocol_fee_share=data["protocol_fee_share"],
        protocol_flash_loan_fee_share=data["protocol_flash_loan_fee_share"],
        reserve_x=data["reserve_x"],
        reserve_y=data["reserve_y"],
        coin_type_x=data["coin_type_x"],
        coin_type_y=data["coin_type_y"],
        reward_infos=data["reward_infos"],
        observation=data["observation"],
        tick_bitmap_handle=data["tick_bitmap_handle"],
        tick_bitmap_size=data["tick_bitmap_size"],
        observation_cardinality=data["observation_cardinality"],
        observation_cardinality_next=data["observation_cardinality_next"],
        observation_index=data["observation_index"],
        tick_handle_id=data["tick_handle_id"],
        ticks=ticks,
        raw_snapshot=data.get("raw_snapshot", {}),
    )


def dump_pool_state(state: PoolState, path: str | Path) -> Path:
    """Serialise ``state`` to ``path`` in JSON format."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(state.to_dict(), indent=2))
    return output_path


def compare_pool_states(expected: PoolState, actual: PoolState) -> Dict[str, Any]:
    """Return a diff between two pool states.

    The result contains mismatched scalar fields, tick-level differences, and metadata
    about missing/extra ticks. ``matches`` is ``True`` only when all tracked values
    are identical.
    """

    scalar_fields = [
        "pool_id",
        "version",
        "sqrt_price",
        "liquidity",
        "tick_current_index",
        "tick_spacing",
        "fee_growth_global_x",
        "fee_growth_global_y",
        "protocol_fee_x",
        "protocol_fee_y",
        "swap_fee_rate",
        "flash_loan_fee_rate",
        "protocol_fee_share",
        "protocol_flash_loan_fee_share",
        "reserve_x",
        "reserve_y",
        "tick_bitmap_handle",
        "tick_bitmap_size",
        "observation_cardinality",
        "observation_cardinality_next",
        "observation_index",
    ]

    field_mismatches: Dict[str, Dict[str, Any]] = {}
    for field in scalar_fields:
        exp = getattr(expected, field)
        act = getattr(actual, field)
        if exp != act:
            field_mismatches[field] = {"expected": exp, "actual": act}

    observation_keys = [
        "initialized",
        "seconds_per_liquidity_cumulative",
        "tick_cumulative",
        "timestamp_s",
    ]
    for key in observation_keys:
        exp = expected.observation.get(key)
        act = actual.observation.get(key)
        if exp != act:
            field_mismatches[f"observation.{key}"] = {"expected": exp, "actual": act}

    reward_infos_expected = expected.reward_infos
    reward_infos_actual = actual.reward_infos
    if reward_infos_expected != reward_infos_actual:
        field_mismatches["reward_infos"] = {
            "expected": reward_infos_expected,
            "actual": reward_infos_actual,
        }

    ticks_expected = expected.ticks
    ticks_actual = actual.ticks
    expected_indices = set(ticks_expected.keys())
    actual_indices = set(ticks_actual.keys())
    missing_ticks = sorted(expected_indices - actual_indices)
    extra_ticks = sorted(actual_indices - expected_indices)

    tick_fields = [
        "liquidity_gross",
        "liquidity_net",
        "fee_growth_outside_x",
        "fee_growth_outside_y",
        "reward_growths_outside",
        "seconds_outside",
        "seconds_per_liquidity_outside",
        "tick_cumulative_outside",
    ]

    tick_mismatches: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for index in sorted(expected_indices & actual_indices):
        diffs: Dict[str, Dict[str, Any]] = {}
        expected_tick = ticks_expected[index]
        actual_tick = ticks_actual[index]
        for attr in tick_fields:
            exp = getattr(expected_tick, attr)
            act = getattr(actual_tick, attr)
            if exp != act:
                diffs[attr] = {"expected": exp, "actual": act}
        if diffs:
            tick_mismatches[str(index)] = diffs

    matches = (
        not field_mismatches
        and not tick_mismatches
        and not missing_ticks
        and not extra_ticks
    )

    return {
        "matches": matches,
        "field_mismatches": field_mismatches,
        "tick_mismatches": tick_mismatches,
        "missing_ticks": missing_ticks,
        "extra_ticks": extra_ticks,
    }
