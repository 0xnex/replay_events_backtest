"""Utilities for downloading CLMM pool state snapshots from Sui."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_rpc_urls, load_env
from .sui_rpc import WaterfallSuiRPCClient

logger = logging.getLogger(__name__)


def _find_version(effects: Dict[str, Any], pool_id: str) -> Optional[int]:
    for key in ("mutated", "created", "unwrapped"):
        entries = effects.get(key) or []
        for entry in entries:
            reference = entry.get("reference") or entry.get("objectId")
            if isinstance(reference, dict):
                object_id = reference.get("objectId")
                version = reference.get("version") or entry.get("version")
            else:
                object_id = entry.get("objectId")
                version = entry.get("version")
            if object_id == pool_id and version is not None:
                try:
                    return int(version)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    continue
    return None


def _extract_table_object_id(node: Any) -> Optional[str]:
    if isinstance(node, dict):
        if "id" in node and isinstance(node["id"], dict):
            nested = node["id"].get("id")
            if isinstance(nested, str):
                return nested
        fields = node.get("fields")
        if isinstance(fields, dict):
            identifier = fields.get("id")
            if isinstance(identifier, dict):
                nested = identifier.get("id") or identifier.get("value")
                if isinstance(nested, str):
                    return nested
    return None


def _find_tick_handle_id(data: Any) -> Optional[str]:
    stack: List[Any] = [data]
    visited: set[int] = set()

    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            obj_id = id(current)
            if obj_id in visited:
                continue
            visited.add(obj_id)
            for key, value in current.items():
                if isinstance(key, str) and "tick" in key.lower():
                    table_type = None
                    if isinstance(value, dict):
                        table_type = value.get("type")
                    candidate = _extract_table_object_id(value)
                    if candidate:
                        if not isinstance(table_type, str) or "tick::tickinfo" in table_type.lower():
                            return candidate
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, (dict, list)):
                    stack.append(item)
    return None


def _normalise_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _normalise_json(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_normalise_json(value) for value in data]
    if isinstance(data, bytes):  # pragma: no cover - defensive
        return data.decode("utf-8")
    return data


def _extract_object_details(response: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(response, dict):
        return None
    details = response.get("details")
    if isinstance(details, dict):
        return details
    data = response.get("data")
    if isinstance(data, dict):
        return data
    return None


def _safe_filename_component(value: str) -> str:
    allowed = {"-", "_"}
    sanitized = [ch if ch.isalnum() or ch in allowed else "_" for ch in value]
    return "".join(sanitized)


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def fetch_pool_snapshot(
    pool_id: str,
    tx_digest: str,
    *,
    output_dir: Path | str = Path("data"),
    env_file: str = ".env",
    fetch_tick_map: bool = True,
    rpc_timeout: float = 30.0,
    tick_handle_override: Optional[str] = None,
) -> Path:
    """Download the pool object (and optionally its tick map) for a tx digest.

    Parameters
    ----------
    pool_id:
        The Sui object ID of the pool.
    tx_digest:
        Digest identifying the transaction whose post-state should be used.
    output_dir:
        Directory where the JSON snapshot is stored. Created if missing.
    env_file:
        Path to the dotenv file containing the RPC endpoint configuration.
    fetch_tick_map:
        When true, downloads all dynamic field objects under the tick handle.
    rpc_timeout:
        Request timeout in seconds.

    Returns
    -------
    Path to the stored JSON file.
    """

    env = load_env(env_file)
    rpc_urls = get_rpc_urls(env)

    verify = not _is_truthy(env.get("SUI_RPC_SKIP_TLS_VERIFY"))
    cafile = env.get("SUI_RPC_CA_BUNDLE") or env.get("SUI_RPC_CA_FILE")
    client_kwargs = {"verify": verify}
    if cafile and verify:
        client_kwargs["cafile"] = cafile

    client = WaterfallSuiRPCClient(
        urls=rpc_urls,
        timeout=rpc_timeout,
        client_kwargs=client_kwargs,
    )

    tx_data = client.get_transaction_block(tx_digest, show_effects=True, show_events=False)
    effects = tx_data.get("effects") or {}
    version = _find_version(effects, pool_id)
    if version is None:
        raise RuntimeError(
            "Pool object version could not be located in transaction effects. "
            "Verify that the pool_id was touched by the provided transaction."
        )

    past_object = client.try_get_past_object(pool_id, version)
    if past_object.get("status") != "VersionFound":
        raise RuntimeError(
            f"Unable to fetch pool object at version {version}: {past_object.get('status')}"
        )

    pool_details = _extract_object_details(past_object)
    if not (pool_details and pool_details.get("fields")):
        live_object = client.get_object(pool_id)
        live_details = _extract_object_details(live_object)
        if live_details:
            pool_details = live_details

    tick_handle_id: Optional[str] = tick_handle_override
    if fetch_tick_map and tick_handle_id is None:
        data_section: Any = pool_details
        if isinstance(pool_details, dict):
            if isinstance(pool_details.get("content"), dict):
                data_section = pool_details["content"]
            elif isinstance(pool_details.get("data"), dict):
                data_section = pool_details["data"]

        search_root: Any = data_section
        log_fields: Optional[Dict[str, Any]] = None
        if isinstance(data_section, dict) and "fields" in data_section:
            fields_candidate = data_section.get("fields")
            if isinstance(fields_candidate, (dict, list)):
                search_root = fields_candidate
                if isinstance(fields_candidate, dict):
                    log_fields = fields_candidate
        elif isinstance(data_section, dict):
            log_fields = data_section

        tick_handle_id = _find_tick_handle_id(search_root)
        if not tick_handle_id:
            field_keys = sorted(str(key) for key in (log_fields or {}).keys())
            logger.warning(
                "Tick handle ID could not be found in pool object. Available field keys: %s",
                ", ".join(field_keys) or "<none>",
            )

    tick_map_objects: List[Dict[str, Any]] = []
    missing_dynamic_fields: List[Dict[str, Any]] = []

    if fetch_tick_map and tick_handle_id:
        cursor: Optional[str] = None
        while True:
            response = client.get_dynamic_fields(tick_handle_id, cursor=cursor)
            entries = response.get("data", [])
            for entry in entries:
                object_id = entry.get("objectId")
                version_str = entry.get("version")
                name_descriptor = entry.get("name")
                if not object_id:
                    continue
                try:
                    df_version = int(version_str) if version_str is not None else version
                except (TypeError, ValueError):
                    df_version = version

                details: Optional[Dict[str, Any]] = None
                status: Optional[str] = None

                try:
                    dynamic_obj = client.try_get_past_object(object_id, df_version)
                except RuntimeError as exc:
                    missing_dynamic_fields.append(
                        {
                            "objectId": object_id,
                            "version": df_version,
                            "error": str(exc),
                        }
                    )
                else:
                    status = dynamic_obj.get("status") if isinstance(dynamic_obj, dict) else None
                    if status == "VersionFound":
                        details = _extract_object_details(dynamic_obj)

                if details is None:
                    if name_descriptor is None:
                        missing_dynamic_fields.append(
                            {
                                "objectId": object_id,
                                "version": df_version,
                                "status": status,
                                "reason": "Name descriptor unavailable for fallback fetch",
                            }
                        )
                        continue

                    try:
                        fallback = client.get_dynamic_field_object(tick_handle_id, name_descriptor)
                    except RuntimeError as exc:
                        missing_dynamic_fields.append(
                            {
                                "objectId": object_id,
                                "version": df_version,
                                "status": status,
                                "error": str(exc),
                                "reason": "fallback get_dynamic_field_object failed",
                            }
                        )
                        continue

                    details = _extract_object_details(fallback) or fallback

                tick_map_objects.append(
                    {
                        "descriptor": entry,
                        "object": details,
                    }
                )

            if not response.get("hasNextPage"):
                break
            cursor = response.get("nextCursor")
            if cursor is None:
                break

    snapshot = {
        "pool_id": pool_id,
        "tx_digest": tx_digest,
        "pool_version": version,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "pool_object": pool_details,
        "tick_handle_id": tick_handle_id,
        "tick_map_objects": tick_map_objects,
        "missing_dynamic_fields": missing_dynamic_fields,
    }

    file_name = f"{_safe_filename_component(pool_id)}_{_safe_filename_component(tx_digest)}.json"
    output_path = Path(output_dir) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_normalise_json(snapshot), indent=2))
    return output_path


def fetch_pool_snapshot_from_txs(
    pool_id: str,
    tx_digest: str,
    txs_dir: Path | str = Path("txs"),
    *,
    output_dir: Path | str = Path("data"),
    fetch_tick_map: bool = True,
    tick_handle_override: Optional[str] = None,
) -> Path:
    """Download the pool object from local txs directory instead of querying from chain.

    This function searches through the local transaction files to find the pool object
    at the specified digest, useful for offline analysis and testing.

    Parameters
    ----------
    pool_id:
        The Sui object ID of the pool.
    tx_digest:
        Digest identifying the transaction whose post-state should be used.
    txs_dir:
        Directory containing the transaction files (e.g., txs/{pool_id}/page_*.json).
    output_dir:
        Directory where the JSON snapshot is stored. Created if missing.
    fetch_tick_map:
        When true, attempts to extract tick map data from the transaction.
    tick_handle_override:
        Override the tick handle ID if known.

    Returns
    -------
    Path to the stored JSON file.

    Raises
    ------
    FileNotFoundError:
        If the transaction file is not found in the txs directory.
    RuntimeError:
        If the pool object cannot be found in the transaction data.
    """
    txs_path = Path(txs_dir)
    if not txs_path.exists():
        raise FileNotFoundError(f"Transaction directory not found: {txs_path}")

    # Search for the transaction in the pool's directory
    pool_txs_dir = txs_path / pool_id
    if not pool_txs_dir.exists():
        raise FileNotFoundError(f"Pool transaction directory not found: {pool_txs_dir}")

    # Look for the transaction in page files
    tx_data = None
    tx_file = None
    
    # First, try to find it in page files
    for page_file in sorted(pool_txs_dir.glob("page_*.json")):
        try:
            with open(page_file, 'r') as f:
                page_data = json.load(f)
            
            # Check if this page contains our transaction
            if isinstance(page_data, dict):
                # Handle different page formats
                transactions = page_data.get("data", [])
                if not transactions and "result" in page_data:
                    transactions = [page_data["result"]]
                
                for tx in transactions:
                    if isinstance(tx, dict) and tx.get("digest") == tx_digest:
                        tx_data = tx
                        tx_file = page_file
                        break
            elif isinstance(page_data, list):
                for tx in page_data:
                    if isinstance(tx, dict) and tx.get("digest") == tx_digest:
                        tx_data = tx
                        tx_file = page_file
                        break
            
            if tx_data:
                break
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error reading page file {page_file}: {e}")
            continue

    if not tx_data:
        raise RuntimeError(f"Transaction {tx_digest} not found in txs directory {pool_txs_dir}")

    logger.info(f"Found transaction {tx_digest} in {tx_file}")

    # Check if this is an event-based format (like from Sui RPC events)
    if "events" in tx_data and "effects" not in tx_data:
        logger.info("Detected event-based transaction format")
        
        # For event-based format, we can't extract the full pool object
        # This is a limitation - we would need the full transaction data
        # For now, create a minimal pool object from available data
        events = tx_data.get("events", [])
        
        # Look for pool-related events
        pool_events = []
        for event in events:
            if isinstance(event, dict):
                parsed_json = event.get("parsedJson", {})
                if parsed_json.get("pool_id") == pool_id:
                    pool_events.append(event)
        
        if not pool_events:
            raise RuntimeError(f"No pool-related events found for {pool_id} in transaction {tx_digest}")
        
        # Create a minimal pool object from event data
        # This is a simplified representation
        pool_object = {
            "dataType": "moveObject",
            "type": f"0x70285592c97965e811e0c6f98dccc3a9c2b4ad854b3594faab9597ada267b860::pool::Pool",
            "hasPublicTransfer": False,
            "fields": {
                "id": {"id": pool_id},
                "version": "1",  # Default version
                "digest": tx_digest,
                "events": pool_events
            }
        }
        
        # Set version to 1 as default for event-based format
        version = 1
        tick_handle_id = tick_handle_override
        
    else:
        # Handle full transaction format with effects
        effects = tx_data.get("effects", {})
        if not effects:
            raise RuntimeError(f"No effects found in transaction {tx_digest}")

        # Find the pool object version from effects
        version = _find_version(effects, pool_id)
        if version is None:
            raise RuntimeError(
                f"Pool object version could not be located in transaction effects for {pool_id}. "
                "Verify that the pool_id was touched by the provided transaction."
            )

        # Extract pool object from transaction data
        pool_object = None
        tick_handle_id = tick_handle_override
        
        # Look for the pool object in the transaction's object changes
        object_changes = effects.get("mutated", []) + effects.get("created", []) + effects.get("unwrapped", [])
        
        for change in object_changes:
            if change.get("objectId") == pool_id:
                # Try to get the object data from the transaction
                if "object" in change:
                    pool_object = change["object"]
                elif "reference" in change:
                    ref = change["reference"]
                    if isinstance(ref, dict) and "objectId" in ref:
                        # This is a reference, we need to extract from transaction data
                        # Look in the transaction's object data
                        objects = tx_data.get("objectChanges", [])
                        for obj in objects:
                            if obj.get("objectId") == pool_id:
                                pool_object = obj.get("object")
                                break
                break

        if not pool_object:
            # Fallback: try to find in transaction's object data
            objects = tx_data.get("objectChanges", [])
            for obj in objects:
                if obj.get("objectId") == pool_id:
                    pool_object = obj.get("object")
                    break

        if not pool_object:
            raise RuntimeError(f"Pool object {pool_id} not found in transaction {tx_digest}")

    # Extract tick handle if needed
    if fetch_tick_map and tick_handle_id is None:
        tick_handle_id = _find_tick_handle_id(pool_object)

    # Extract tick map objects from transaction data
    tick_map_objects: List[Dict[str, Any]] = []
    missing_dynamic_fields: List[Dict[str, Any]] = []

    if fetch_tick_map and tick_handle_id:
        # Look for dynamic fields in the transaction's object changes
        objects = tx_data.get("objectChanges", [])
        for obj in objects:
            if obj.get("objectId") == tick_handle_id:
                # This is the tick handle, look for its dynamic fields
                # In a real implementation, you might need to parse the dynamic fields
                # from the transaction data or make additional queries
                logger.info(f"Found tick handle {tick_handle_id} in transaction")
                break

    # Create snapshot
    snapshot = {
        "pool_id": pool_id,
        "tx_digest": tx_digest,
        "pool_version": version,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "local_txs",
        "tx_file": str(tx_file) if tx_file else None,
        "pool_object": pool_object,
        "tick_handle_id": tick_handle_id,
        "tick_map_objects": tick_map_objects,
        "missing_dynamic_fields": missing_dynamic_fields,
    }

    # Save to output directory
    file_name = f"{_safe_filename_component(pool_id)}_{_safe_filename_component(tx_digest)}.json"
    output_path = Path(output_dir) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_normalise_json(snapshot), indent=2))
    
    logger.info(f"Pool snapshot saved to: {output_path}")
    return output_path
