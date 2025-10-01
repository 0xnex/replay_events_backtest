"""Transaction processor for maintaining pool state by replaying transactions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .pool_state import PoolState, TickInfo, _parse_move_int

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TransactionEvent:
    """Represents a single event from a transaction."""
    
    tx_digest: str
    event_seq: int
    event_type: str
    parsed_json: Dict[str, Any]
    timestamp: Optional[int] = None
    
    @classmethod
    def from_raw_event(cls, tx_digest: str, raw_event: Dict[str, Any], tx_timestamp: Optional[int] = None) -> "TransactionEvent":
        """Create a TransactionEvent from raw event data."""
        event_id = raw_event.get("id", {})
        return cls(
            tx_digest=tx_digest,
            event_seq=int(event_id.get("eventSeq", 0)),
            event_type=raw_event.get("type", ""),
            parsed_json=raw_event.get("parsedJson", {}),
            timestamp=tx_timestamp,
        )


@dataclass(slots=True)
class SwapEvent:
    """Represents a swap event with parsed data."""
    
    pool_id: str
    amount_x: int
    amount_y: int
    fee_amount: int
    protocol_fee: int
    liquidity: int
    reserve_x: int
    reserve_y: int
    sqrt_price_before: int
    sqrt_price_after: int
    tick_index: Dict[str, int]
    x_for_y: bool
    sender: str
    
    @classmethod
    def from_parsed_json(cls, parsed_json: Dict[str, Any]) -> "SwapEvent":
        """Create SwapEvent from parsed JSON data."""
        return cls(
            pool_id=parsed_json.get("pool_id", ""),
            amount_x=_parse_move_int(parsed_json.get("amount_x", 0)),
            amount_y=_parse_move_int(parsed_json.get("amount_y", 0)),
            fee_amount=_parse_move_int(parsed_json.get("fee_amount", 0)),
            protocol_fee=_parse_move_int(parsed_json.get("protocol_fee", 0)),
            liquidity=_parse_move_int(parsed_json.get("liquidity", 0)),
            reserve_x=_parse_move_int(parsed_json.get("reserve_x", 0)),
            reserve_y=_parse_move_int(parsed_json.get("reserve_y", 0)),
            sqrt_price_before=_parse_move_int(parsed_json.get("sqrt_price_before", 0)),
            sqrt_price_after=_parse_move_int(parsed_json.get("sqrt_price_after", 0)),
            sender=parsed_json.get("sender", ""),
            tick_index=parsed_json.get("tick_index", 0),
            x_for_y=parsed_json.get("x_for_y", True),
        )


@dataclass(slots=True)
class AddLiquidityEvent:
    """Represents an add liquidity event with parsed data."""

    pool_id: str
    position_id: str
    tick_lower: int
    tick_upper: int
    amount_x: int
    amount_y: int
    liquidity: int
    sender: str
    
    @classmethod
    def from_parsed_json(cls, parsed_json: Dict[str, Any]) -> "AddLiquidityEvent":
        """Create AddLiquidityEvent from parsed JSON data."""
        tick_lower_raw = parsed_json.get("tick_lower") or parsed_json.get("lower_tick_index")
        tick_upper_raw = parsed_json.get("tick_upper") or parsed_json.get("upper_tick_index")
        return cls(
            pool_id=parsed_json.get("pool_id", ""),
            position_id=parsed_json.get("position_id", ""),
            tick_lower=_parse_move_int(tick_lower_raw or 0, fallback_bits=32, signed=True),
            tick_upper=_parse_move_int(tick_upper_raw or 0, fallback_bits=32, signed=True),
            amount_x=_parse_move_int(parsed_json.get("amount_x", 0)),
            amount_y=_parse_move_int(parsed_json.get("amount_y", 0)),
            liquidity=_parse_move_int(parsed_json.get("liquidity", 0)),
            sender=parsed_json.get("sender", ""),
        )


@dataclass(slots=True)
class RemoveLiquidityEvent:
    """Represents a remove liquidity event with parsed data."""
    
    pool_id: str
    position_id: str
    tick_lower: int
    tick_upper: int
    amount_x: int
    amount_y: int
    liquidity: int
    sender: str
    
    @classmethod
    def from_parsed_json(cls, parsed_json: Dict[str, Any]) -> "RemoveLiquidityEvent":
        """Create RemoveLiquidityEvent from parsed JSON data."""
        tick_lower_raw = parsed_json.get("tick_lower") or parsed_json.get("lower_tick_index")
        tick_upper_raw = parsed_json.get("tick_upper") or parsed_json.get("upper_tick_index")
        return cls(
            pool_id=parsed_json.get("pool_id", ""),
            position_id=parsed_json.get("position_id", ""),
            tick_lower=_parse_move_int(tick_lower_raw or 0, fallback_bits=32, signed=True),
            tick_upper=_parse_move_int(tick_upper_raw or 0, fallback_bits=32, signed=True),
            amount_x=_parse_move_int(parsed_json.get("amount_x", 0)),
            amount_y=_parse_move_int(parsed_json.get("amount_y", 0)),
            liquidity=_parse_move_int(parsed_json.get("liquidity", 0)),
            sender=parsed_json.get("sender", ""),
        )


@dataclass(slots=True)
class CollectFeeEvent:
    """Represents a collect fee event with parsed data."""
    
    pool_id: str
    position_id: str
    amount_x: int
    amount_y: int
    sender: str
    
    @classmethod
    def from_parsed_json(cls, parsed_json: Dict[str, Any]) -> "CollectFeeEvent":
        """Create CollectFeeEvent from parsed JSON data."""
        return cls(
            pool_id=parsed_json.get("pool_id", ""),
            position_id=parsed_json.get("position_id", ""),
            amount_x=_parse_move_int(parsed_json.get("amount_x", 0)),
            amount_y=_parse_move_int(parsed_json.get("amount_y", 0)),
            sender=parsed_json.get("sender", ""),
        )




class TransactionProcessor:
    """Processes transactions to maintain pool state."""
    
    def __init__(self, pool_state: PoolState):
        self.pool_state = pool_state
        self.processed_events: List[TransactionEvent] = []
        self.validation_errors: List[str] = []
    
    def process_swap_event(self, event: SwapEvent, tx_event: TransactionEvent) -> None:
        """Process a swap event and update pool state."""
        logger.debug(f"Processing swap event: {tx_event.tx_digest}, event: {event}")
        
        # Update pool state with swap results
        if event.sqrt_price_after > 0:
            self.pool_state.sqrt_price = event.sqrt_price_after
        else:
            logger.warning(
                "Swap %s:%s reported non-positive sqrt price (%s); retaining previous value",
                tx_event.tx_digest,
                tx_event.event_seq,
                event.sqrt_price_after,
            )
        self.pool_state.liquidity = event.liquidity
        self.pool_state.reserve_x = event.reserve_x
        self.pool_state.reserve_y = event.reserve_y
        
        # Update fee growth (simplified - in reality this would be more complex)
        if event.amount_x > 0:  # X to Y swap
            fee_growth_delta = (event.fee_amount * (2**128)) // max(event.liquidity, 1)
            self.pool_state.fee_growth_global_x += fee_growth_delta
        else:  # Y to X swap
            fee_growth_delta = (event.fee_amount * (2**128)) // max(event.liquidity, 1)
            self.pool_state.fee_growth_global_y += fee_growth_delta
        
        # Update protocol fees
        self.pool_state.protocol_fee_x += event.protocol_fee if event.amount_x > 0 else 0
        self.pool_state.protocol_fee_y += event.protocol_fee if event.amount_y > 0 else 0
        
        # Update current tick based on sqrt price
        # This is a simplified calculation - in reality it would be more precise
        self.pool_state.tick_current_index = self._sqrt_price_to_tick(event.sqrt_price_after)
        
        # Validate state after swap
        self._validate_swap_state(event)
    
    def process_add_liquidity_event(self, event: AddLiquidityEvent, tx_event: TransactionEvent) -> None:
        """Process an add liquidity event and update pool state."""
        logger.info(f"Processing add liquidity event: {tx_event.tx_digest}")
        
        # Update tick information
        lower_tick = self._ensure_tick(event.tick_lower)
        upper_tick = self._ensure_tick(event.tick_upper)
        lower_tick.liquidity_gross = max(0, lower_tick.liquidity_gross + event.liquidity)
        lower_tick.liquidity_net = max(
            -2**127, min(2**127 - 1, lower_tick.liquidity_net + event.liquidity)
        )
        upper_tick.liquidity_gross = max(0, upper_tick.liquidity_gross + event.liquidity)
        upper_tick.liquidity_net = max(
            -2**127, min(2**127 - 1, upper_tick.liquidity_net - event.liquidity)
        )

        # Update reserves
        self.pool_state.reserve_x += event.amount_x
        self.pool_state.reserve_y += event.amount_y

        # Update total liquidity
        self.pool_state.liquidity += event.liquidity
        
        # Validate state after add liquidity
        self._validate_liquidity_state(event)
    
    def process_remove_liquidity_event(self, event: RemoveLiquidityEvent, tx_event: TransactionEvent) -> None:
        """Process a remove liquidity event and update pool state."""
        logger.debug(f"Processing remove liquidity event: {tx_event.tx_digest}")
        
        # Update tick information
        lower_tick = self._ensure_tick(event.tick_lower)
        upper_tick = self._ensure_tick(event.tick_upper)
        lower_tick.liquidity_gross = max(0, lower_tick.liquidity_gross - event.liquidity)
        lower_tick.liquidity_net = max(
            -2**127, min(2**127 - 1, lower_tick.liquidity_net - event.liquidity)
        )
        upper_tick.liquidity_gross = max(0, upper_tick.liquidity_gross - event.liquidity)
        upper_tick.liquidity_net = max(
            -2**127, min(2**127 - 1, upper_tick.liquidity_net + event.liquidity)
        )
        
        # Update reserves
        self.pool_state.reserve_x -= event.amount_x
        self.pool_state.reserve_y -= event.amount_y
        
        # Update total liquidity
        self.pool_state.liquidity -= event.liquidity
        
        # Validate state after remove liquidity
        self._validate_liquidity_state(event)
    
    def process_collect_fee_event(self, event: CollectFeeEvent, tx_event: TransactionEvent) -> None:
        """Process a collect fee event and update pool state."""
        logger.debug(f"Processing collect fee event: {tx_event.tx_digest}")

        # Fees are collected from the pool, reducing reserves
        self.pool_state.reserve_x -= event.amount_x
        self.pool_state.reserve_y -= event.amount_y

    def process_repay_flash_swap_event(self, event: RepayFlashSwapEvent, tx_event: TransactionEvent) -> None:
        """Process a repay flash swap event and update pool reserves."""
        logger.debug(f"Processing repay flash swap event: {tx_event.tx_digest}")

        # Repayment returns borrowed assets to the vault, so align reserves with event payload
        self.pool_state.reserve_x = event.reserve_x
        self.pool_state.reserve_y = event.reserve_y
    
    def _ensure_tick(self, tick_index: int) -> TickInfo:
        tick = self.pool_state.ticks.get(tick_index)
        if tick is None:
            tick = TickInfo(
                index=tick_index,
                liquidity_gross=0,
                liquidity_net=0,
                fee_growth_outside_x=0,
                fee_growth_outside_y=0,
                reward_growths_outside=[],
                seconds_outside=0,
                seconds_per_liquidity_outside=0,
                tick_cumulative_outside=0,
            )
            self.pool_state.ticks[tick_index] = tick
        return tick
    
    def _sqrt_price_to_tick(self, sqrt_price: int) -> int:
        """Convert sqrt price to tick index (simplified)."""
        # This is a simplified calculation
        # In reality, this would use the proper tick calculation from the CLMM math
        import math
        
        if sqrt_price <= 0:
            logger.debug("Tick conversion skipped due to non-positive sqrt_price=%s", sqrt_price)
            return self.pool_state.tick_current_index
        
        price = (sqrt_price / (2**64)) ** 2
        if price <= 0:
            logger.warning(f"Invalid calculated price: {price}, using current tick")
            return self.pool_state.tick_current_index
            
        tick = int(math.log(price) / math.log(1.0001))
        return tick
    
    def _validate_swap_state(self, event: SwapEvent) -> None:
        """Validate pool state after a swap event."""
        # Check that reserves match the event data
        if self.pool_state.reserve_x != event.reserve_x:
            error = f"Reserve X mismatch: state={self.pool_state.reserve_x}, event={event.reserve_x}"
            self.validation_errors.append(error)
            logger.warning(error)
        
        if self.pool_state.reserve_y != event.reserve_y:
            error = f"Reserve Y mismatch: state={self.pool_state.reserve_y}, event={event.reserve_y}"
            self.validation_errors.append(error)
            logger.warning(error)
        
        # Check that liquidity matches
        if self.pool_state.liquidity != event.liquidity:
            error = f"Liquidity mismatch: state={self.pool_state.liquidity}, event={event.liquidity}"
            self.validation_errors.append(error)
            logger.warning(error)
        
        # Check that sqrt price matches
        if self.pool_state.sqrt_price != event.sqrt_price_after:
            error = f"Sqrt price mismatch: state={self.pool_state.sqrt_price}, event={event.sqrt_price_after}"
            self.validation_errors.append(error)
            logger.warning(error)
    
    def _validate_liquidity_state(self, event: Union[AddLiquidityEvent, RemoveLiquidityEvent]) -> None:
        """Validate pool state after a liquidity event."""
        # Check that liquidity is non-negative
        if self.pool_state.liquidity < 0:
            error = f"Negative liquidity detected: {self.pool_state.liquidity}"
            self.validation_errors.append(error)
            logger.error(error)
        
        # Check that reserves are non-negative
        if self.pool_state.reserve_x < 0:
            error = f"Negative reserve X detected: {self.pool_state.reserve_x}"
            self.validation_errors.append(error)
            logger.error(error)
        
        if self.pool_state.reserve_y < 0:
            error = f"Negative reserve Y detected: {self.pool_state.reserve_y}"
            self.validation_errors.append(error)
            logger.error(error)
        
        # Check tick bounds
        if hasattr(event, 'tick_lower') and hasattr(event, 'tick_upper'):
            if event.tick_lower >= event.tick_upper:
                error = f"Invalid tick range: lower={event.tick_lower}, upper={event.tick_upper}"
                self.validation_errors.append(error)
                logger.error(error)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            "total_events": len(self.processed_events),
            "validation_errors": len(self.validation_errors),
            "errors": self.validation_errors,
        }
    
    def process_event(self, event: TransactionEvent) -> None:
        """Process a single transaction event."""
        self.processed_events.append(event)
        
        event_type = event.event_type

        if event_type.endswith("::SwapEvent"):
            swap_event = SwapEvent.from_parsed_json(event.parsed_json)
            self.process_swap_event(swap_event, event)
        elif event_type.endswith("::RepayFlashSwapEvent"):
            repay_event = RepayFlashSwapEvent.from_parsed_json(event.parsed_json)
            self.process_repay_flash_swap_event(repay_event, event)
        elif event_type.endswith("::AddLiquidityEvent"):
            add_event = AddLiquidityEvent.from_parsed_json(event.parsed_json)
            self.process_add_liquidity_event(add_event, event)
        elif event_type.endswith("::RemoveLiquidityEvent"):
            remove_event = RemoveLiquidityEvent.from_parsed_json(event.parsed_json)
            self.process_remove_liquidity_event(remove_event, event)
        elif event_type.endswith("::CollectFeeEvent"):
            collect_event = CollectFeeEvent.from_parsed_json(event.parsed_json)
            self.process_collect_fee_event(collect_event, event)
        else:
            logger.debug("Skipping unsupported event type %s for %s", event.event_type, event.tx_digest)
        self.last_processed_event = event


class TransactionReplayer:
    """Replays transactions to maintain pool state."""

    def __init__(self, pool_state: PoolState, event_filter: Optional[Dict[str, List[int]]] = None):
        self.processor = TransactionProcessor(pool_state)
        self.pool_state = pool_state
        self.pool_state_timestamp = self._extract_pool_state_timestamp()
        self._event_filter: Optional[Dict[str, set[int]]] = None
        self._events_filtered: int = 0
        self._event_map: Dict[str, Set[int]] = {}
        self.last_processed_event: Optional[TransactionEvent] = None
        if event_filter:
            self._event_filter = {
                digest: {int(seq) for seq in sequences}
                for digest, sequences in event_filter.items()
            }
    
    def _extract_pool_state_timestamp(self) -> Optional[int]:
        """Extract timestamp from pool state observation data."""
        observation = self.pool_state.observation
        if isinstance(observation, dict):
            timestamp = observation.get("timestamp_s")
            if timestamp is not None:
                return int(timestamp)
        return None
    
    def _extract_transaction_timestamp(self, tx_data: Dict[str, Any]) -> Optional[int]:
        """Extract timestamp from transaction data."""
        # Try to get timestamp from transaction metadata
        if "timestampMs" in tx_data:
            return int(tx_data["timestampMs"]) // 1000  # Convert to seconds
        if "timestamp" in tx_data:
            return int(tx_data["timestamp"])
        
        # Try to get from effects or other metadata
        effects = tx_data.get("effects", {})
        if "timestampMs" in effects:
            return int(effects["timestampMs"]) // 1000
        
        # If no timestamp found, return None
        return None
    
    def load_transaction_page(self, page_path: Path) -> List[TransactionEvent]:
        """Load events from a transaction page file, filtering for pool-relevant events."""
        with open(page_path, 'r') as f:
            data = json.load(f)
        
        events = []
        pool_id = self.pool_state.pool_id
        
        for tx_data in data.get("data", []):
            tx_digest = tx_data.get("digest", "")
            tx_timestamp = self._extract_transaction_timestamp(tx_data)

            # Check if this transaction contains events for our pool
            has_pool_events = False
            for raw_event in tx_data.get("events", []):
                if self._is_pool_relevant_event(tx_digest, raw_event, pool_id):
                    has_pool_events = True
                    break
            
            # Only process transactions that have events for our pool
            if has_pool_events:
                logger.debug(f"Processing transaction {tx_digest} (contains pool events)")
                for raw_event in tx_data.get("events", []):
                    if self._is_pool_relevant_event(tx_digest, raw_event, pool_id):
                        event = TransactionEvent.from_raw_event(tx_digest, raw_event, tx_timestamp)
                        events.append(event)
                        self._event_map.setdefault(tx_digest, set()).add(event.event_seq)
            else:
                logger.debug(f"Skipping transaction {tx_digest} (no pool events)")
        
        return events
    
    def _is_pool_relevant_event(self, tx_digest: str, raw_event: Dict[str, Any], pool_id: str) -> bool:
        """Check if an event is relevant to the specified pool."""
        event_type = raw_event.get("type", "")
        parsed_json = raw_event.get("parsedJson", {})

        supported_suffixes = (
            "::SwapEvent",
            "::RepayFlashSwapEvent",
            "::AddLiquidityEvent",
            "::RemoveLiquidityEvent",
            "::CollectFeeEvent",
        )

        if not any(event_type.endswith(suffix) for suffix in supported_suffixes):
            return False

        event_pool_id = (
            parsed_json.get("pool_id")
            or parsed_json.get("pool")
            or parsed_json.get("event", {}).get("pool_id")
        )
        if event_pool_id != pool_id:
            return False

        if self._event_filter is None:
            return True

        allowed_sequences = self._event_filter.get(tx_digest)
        if allowed_sequences is None:
            return False

        event_seq_raw = raw_event.get("id", {}).get("eventSeq")
        try:
            event_seq = int(event_seq_raw)
        except (TypeError, ValueError):
            logger.debug("Skipping event with invalid eventSeq %s in %s", event_seq_raw, tx_digest)
            return False

        if event_seq in allowed_sequences:
            return True

        self._events_filtered += 1
        logger.debug(
            "Skipping eventSeq %s for %s - whitelist %s",
            event_seq,
            tx_digest,
            sorted(allowed_sequences),
        )
        return False
    
    def replay_transactions(
        self,
        txs_dir: Path,
        start_page: int = 1,
        end_page: Optional[int] = None,
        *,
        process_events: bool = True,
    ) -> None:
        """Replay transactions from the specified page range in chronological order.

        Set ``process_events`` to ``False`` to only collect the relevant events without
        mutating the pool state.
        """
        pool_id = self.pool_state.pool_id
        pool_txs_dir = txs_dir / pool_id
        
        if not pool_txs_dir.exists():
            raise ValueError(f"Transaction directory not found: {pool_txs_dir}")
        
        # Get all page files
        page_files = sorted([f for f in pool_txs_dir.glob("page_*.json")])
        
        if end_page is None:
            end_page = len(page_files)
        
        logger.info(f"Replaying transactions from page {start_page} to {end_page}")
        if self.pool_state_timestamp is not None:
            logger.info(f"Pool state timestamp: {self.pool_state_timestamp}")
            logger.info("Filtering transactions to only include those after pool state timestamp")
        
        # Collect all events from all pages first
        all_events = []
        processed_transactions = 0
        skipped_transactions = 0
        
        for i in range(start_page - 1, min(end_page, len(page_files))):
            page_file = page_files[i]
            logger.debug(f"Loading page: {page_file.name}")
            
            # Count transactions before and after filtering
            with open(page_file, 'r') as f:
                page_data = json.load(f)
            total_txs = len(page_data.get("data", []))
            
            events = self.load_transaction_page(page_file)
            all_events.extend(events)
            
            # Count how many transactions had pool events
            pool_tx_count = 0
            for tx_data in page_data.get("data", []):
                tx_digest = tx_data.get("digest", "")
                has_pool_events = False
                for raw_event in tx_data.get("events", []):
                    if self._is_pool_relevant_event(tx_digest, raw_event, pool_id):
                        has_pool_events = True
                        break
                if has_pool_events:
                    pool_tx_count += 1
            
            processed_transactions += pool_tx_count
            skipped_transactions += total_txs - pool_tx_count
        
        logger.info(f"Processed {processed_transactions} transactions with pool events (skipped {skipped_transactions} irrelevant transactions)")
        
        # Filter events by timestamp if pool state timestamp is available
        if self.pool_state_timestamp is not None:
            filtered_events = []
            for event in all_events:
                if event.timestamp is not None and event.timestamp > self.pool_state_timestamp:
                    filtered_events.append(event)
                elif event.timestamp is None:
                    # If no timestamp available, include the event (assume it's after)
                    logger.warning(f"Event {event.tx_digest} has no timestamp, including it")
                    filtered_events.append(event)
            
            logger.info(f"Filtered {len(all_events)} events to {len(filtered_events)} events after pool state timestamp")
            all_events = filtered_events
        
        # Sort events chronologically by timestamp
        all_events.sort(key=lambda e: e.timestamp or 0)
        
        # Process events in chronological order
        if process_events:
            for event in all_events:
                self.processor.process_event(event)
            logger.info(f"Processed {len(self.processor.processed_events)} events")
        else:
            logger.info(f"Identified {len(all_events)} relevant events (processing skipped)")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current pool state."""
        validation_summary = self.processor.get_validation_summary()
        return {
            "pool_id": self.pool_state.pool_id,
            "sqrt_price": self.pool_state.sqrt_price,
            "liquidity": self.pool_state.liquidity,
            "tick_current_index": self.pool_state.tick_current_index,
            "reserve_x": self.pool_state.reserve_x,
            "reserve_y": self.pool_state.reserve_y,
            "fee_growth_global_x": self.pool_state.fee_growth_global_x,
            "fee_growth_global_y": self.pool_state.fee_growth_global_y,
            "protocol_fee_x": self.pool_state.protocol_fee_x,
            "protocol_fee_y": self.pool_state.protocol_fee_y,
            "processed_events": len(self.processor.processed_events),
            "validation_errors": validation_summary["validation_errors"],
            "validation_summary": validation_summary,
            "pool_state_timestamp": self.pool_state_timestamp,
            "temporal_filtering": self.pool_state_timestamp is not None,
            "event_filter_applied": self._event_filter is not None,
            "events_filtered_out": self._events_filtered,
            "digests_with_events": len(self._event_map),
        }

    def get_event_map(self) -> Dict[str, List[int]]:
        """Return mapping of transaction digests to sorted event sequence numbers."""

        return {digest: sorted(seqs) for digest, seqs in self._event_map.items()}
@dataclass(slots=True)
class RepayFlashSwapEvent:
    """Represents a repay-FlashSwap event."""

    pool_id: str
    amount_x_debt: int
    amount_y_debt: int
    paid_x: int
    paid_y: int
    reserve_x: int
    reserve_y: int
    sender: str

    @classmethod
    def from_parsed_json(cls, parsed_json: Dict[str, Any]) -> "RepayFlashSwapEvent":
        return cls(
            pool_id=parsed_json.get("pool_id", ""),
            amount_x_debt=_parse_move_int(parsed_json.get("amount_x_debt", 0)),
            amount_y_debt=_parse_move_int(parsed_json.get("amount_y_debt", 0)),
            paid_x=_parse_move_int(parsed_json.get("paid_x", 0)),
            paid_y=_parse_move_int(parsed_json.get("paid_y", 0)),
            reserve_x=_parse_move_int(parsed_json.get("reserve_x", 0)),
            reserve_y=_parse_move_int(parsed_json.get("reserve_y", 0)),
            sender=parsed_json.get("sender", ""),
        )
