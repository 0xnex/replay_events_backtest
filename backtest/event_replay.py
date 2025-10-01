from ast import Tuple
from typing import Callable, Optional

from backtest.pool_state import PoolState


TickFromSqrtFn = Callable[[int], int]

class EventReplayer:
    def __init__(self, pool: PoolState, tick_from_sqrt: TickFromSqrtFn):
        self.pool = pool
        self.tick_from_sqrt = tick_from_sqrt

    @staticmethod
    def _read_before_after_sqrt(ev: dict) -> Optional[Tuple[int, int]]:
        # Momentum: sqrt_price_before/after
        if "sqrt_price_before" in ev and "sqrt_price_after" in ev:
            return int(ev["sqrt_price_before"]), int(ev["sqrt_price_after"])
        # Cetus: before_sqrt_price/after_sqrt_price
        if "before_sqrt_price" in ev and "after_sqrt_price" in ev:
            return int(ev["before_sqrt_price"]), int(ev["after_sqrt_price"])
        return None

    def process_swap(self, ev: dict):
        ba = self._read_before_after_sqrt(ev)
        if ba is None:
            raise ValueError("Swap event missing before/after sqrt price.")
        spb, spa = ba
        tb = self.pool.align_tick(self.tick_from_sqrt(spb))
        ta = self.pool.align_tick(self.tick_from_sqrt(spa))

        # sync event before state
        self.pool.tick_current = tb
        self.pool.sqrt_price_q64x64 = spb

        # accumulate global fees (event may only have fee_amount / or split X/Y)
        fee_x = int(ev.get("fee_amount_x", ev.get("fee_amount", 0)))
        fee_y = int(ev.get("fee_amount_y", 0))
        if fee_x or fee_y:
            self.pool.accrue_fees_global(fee_x, fee_y)

        up = spa > spb
        cur = tb
        while True:
            nxt = self.pool.tickmap.next_initialized(cur, up)
            if nxt is None:
                break
            if up:
                if nxt > ta:
                    break
                self.pool.cross_tick(nxt, True)
                cur = nxt
            else:
                if nxt <= ta:
                    break
                self.pool.cross_tick(nxt, False)
                cur = nxt

        # event after
        self.pool.sqrt_price_q64x64 = spa
        self.pool.tick_current = ta

        # after crossing, optional: refresh once for all active positions after each swap
        # optional: refresh once for all active positions after each swap
        for pos in self.pool.positions.values():
            self.pool.position_update_fees(pos.owner, pos.lower, pos.upper)

    @staticmethod
    def _read_tick_bits_like(ev: dict, lower_key="lower_tick_index", upper_key="upper_tick_index") -> Tuple[int, int]:
        def _read(v):
            if isinstance(v, dict) and "bits" in v:
                return int(v["bits"])
            return int(v)
        lo = ev.get(lower_key, ev.get("tick_lower_index"))
        up = ev.get(upper_key, ev.get("tick_upper_index"))
        return _read(lo), _read(up)

    def process_open_position(self, ev: dict, owner: Optional[str] = None):
        lo_bits, up_bits = self._read_tick_bits_like(ev, "tick_lower_index", "tick_upper_index")
        lower = self.pool.align_tick(lo_bits)
        upper = self.pool.align_tick(up_bits)
        liq = int(ev.get("liquidity", 0))
        owner = owner or ev.get("sender") or "owner"
        if liq > 0:
            self.pool.position_update_add(owner, lower, upper, liq)

    def process_add_liquidity(self, ev: dict, owner: Optional[str] = None):
        lo_bits, up_bits = self._read_tick_bits_like(ev, "lower_tick_index", "upper_tick_index")
        lower = self.pool.align_tick(lo_bits)
        upper = self.pool.align_tick(up_bits)
        liq = int(ev["liquidity"])
        owner = owner or ev.get("sender") or "owner"
        self.pool.position_update_add(owner, lower, upper, liq)

    def apply_event(self, ev_type: str, ev: dict, owner: Optional[str] = None):
        t = ev_type.lower()
        if "swap" in t:
            self.process_swap(ev)
        elif "openposition" in t:
            self.process_open_position(ev, owner)
        elif "addliquidity" in t:
            self.process_add_liquidity(ev, owner)
        else:
            # extendable remove_liquidity / collect etc.
            pass