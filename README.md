# Replaying Event Trigger Backtest Framework

## Overview

The Replaying Event Trigger Backtest Framework is a comprehensive system designed for backtesting Concentrated Liquidity Market Maker (CLMM) strategies on the Sui network. The framework enables historical analysis by downloading on-chain pool data, replaying historical events (swaps, liquidity changes, fee collections, etc.), and maintaining accurate pool state throughout the replay process.

## Core Purpose

The framework provides the following key capabilities:

1. **Download historical pool data and events** for a specified pool and start timestamp
2. **Initialize pool state** from local data and update it incrementally as events are replayed
3. **Support position operations**: `add_liquidity`, `remove_liquidity`, `collect_fee`, `RepayFlashSwap` and `swap`
4. **Evaluate current value of positions** based on the maintained pool state
5. **Verify state consistency** against on-chain snapshots

## Architecture Overview

### Core Components

The framework is built around several key components:

#### 1. Transaction Processing System (`backtest/tx_processor.py`)

**Key Classes:**

- `TransactionEvent`: Represents individual transaction events with parsed data
- `SwapEvent`, `AddLiquidityEvent`, `RemoveLiquidityEvent`, `CollectFeeEvent`: Typed event classes for different operations
- `TransactionProcessor`: Core processor that updates pool state based on events
- `TransactionReplayer`: High-level interface for replaying transaction pages

**Features:**

- ✅ Event parsing from raw transaction data
- ✅ **Timestamp extraction from transaction data**
- ✅ **Temporal filtering (only transactions after pool state timestamp)**
- ✅ **Chronological ordering (time ascending)**
- ✅ **Per-transaction event whitelists via `--event-filter`**
- ✅ State updates for all major operations (swap, add/remove liquidity, collect fees)
- ✅ Tick management for liquidity positions
- ✅ Fee growth tracking and protocol fee accumulation
- ✅ Comprehensive validation system

#### 2. Pool State Management (`backtest/pool_state.py`)

The `PoolState` class maintains the complete state of a CLMM pool including:

- Current sqrt price and liquidity
- Reserve amounts for both tokens
- Tick information and liquidity distribution
- Fee growth tracking (global and per-tick)
- Protocol fee accumulation
- Oracle observations and timestamps

#### 3. Event Replay System (`backtest/event_replay.py`)

The `EventReplayer` class provides specialized event processing:

- Swap event processing with tick crossing logic
- Liquidity position management
- Fee accumulation and distribution
- Support for multiple DEX formats (Momentum, Cetus)

#### 4. CLI Interface (`backtest/cli.py`)

Comprehensive command-line interface with multiple commands:

- `download-pool-state`: Download pool snapshots from chain
- `init-pool-state`: Initialize pool state from snapshots
- `replay-transactions`: Replay historical transactions
- `list-events`: Generate event filters
- `verify-state`: Compare states against references

## Supported Event Types

The framework supports five main event types:

### 1. SwapEvent

- **Purpose**: Token swaps with price updates
- **Processing**: Updates sqrt_price, liquidity, reserves, fee growth, and protocol fees
- **Validation**: Ensures state consistency after swaps

### 2. AddLiquidityEvent

- **Purpose**: Adding liquidity to tick ranges
- **Processing**: Updates reserves, total liquidity, and tick information
- **Validation**: Ensures non-negative values and valid tick ranges

### 3. RemoveLiquidityEvent

- **Purpose**: Removing liquidity from tick ranges
- **Processing**: Updates reserves, total liquidity, and tick information
- **Validation**: Ensures non-negative values and valid tick ranges

### 4. CollectFeeEvent

- **Purpose**: Collecting accumulated fees
- **Processing**: Reduces reserves by collected fees and updates fee tracking
- **Validation**: Ensures proper fee accounting

### 5. RepayFlashSwapEvent

- **Purpose**: Flash swap repayments
- **Processing**: Handles flash swap completion and fee settlement
- **Validation**: Ensures proper repayment accounting

## File Structure

```
backtest/
├── data/                                    # Pool snapshots
│   └── ${pool_id}_${digest}.json
├── txs/                                     # Transaction files
│   └── ${pool_id}/
│       ├── page_00001.json
│       ├── page_00002.json
│       └── ...
└── state/                                   # State files
    └── ${pool_id}_${digest}_state.json
```

## Usage Workflow

### 1. Download Pool Snapshot

```bash
python -m backtest.cli download-pool-state <POOL_ID> <TX_DIGEST>
```

Downloads pool object and tick map snapshot at a specific transaction to `data/${pool_id}_${digest}.json`.

**Options:**

- `--env-file`: Path to .env file (default: .env)
- `--output-dir`: Directory for snapshot JSON (default: ./data)
- `--skip-tick-map`: Skip downloading tick map
- `--tick-handle-id`: Override tick handle ID
- `--timeout`: RPC timeout in seconds (default: 30)
- `-v, --verbose`: Enable debug logging

### 2. Initialize Pool State

```bash
python -m backtest.cli init-pool-state data/${pool_id}_${digest}.json --summary
```

Creates processed pool state file at `state/${pool_id}_${digest}_state.json`.

### 3. List Events (Optional)

```bash
python -m backtest.cli list-events state/${pool_id}_${digest}_state.json txs/ \
    --output event_filter.json --summary
```

Generates event filter mapping transaction digests to event sequences for selective replay.

### 4. Replay Transactions

```bash
python -m backtest.cli replay-transactions \
    state/${pool_id}_${digest}_state.json \
    txs/ \
    --event-filter event_filter.json \
    --verify --env-file .env \
    --summary
```

**Options:**

- `--start-page`: Start page number (default: 1)
- `--end-page`: End page number (default: all)
- `--output`: Output file for updated state
- `--verify`: Verify end state against reference
- `--summary`: Print summary of replay results
- `--event-filter`: JSON file with event filter
- `--env-file`: Path to .env file
- `--timeout`: RPC timeout for verification
- `-v, --verbose`: Enable debug logging

## Advanced Features

### Temporal Filtering

The framework automatically filters transactions to only include those after the pool state timestamp, ensuring chronological accuracy.

### Event Filtering

Users can create selective event filters to replay only specific events:

```json
{
  "0xabc123...": [1, 3, 5],
  "0xdef456...": [2, 4]
}
```

This allows for precise control over which events are processed during replay.

### State Validation

The framework performs comprehensive validation:

- **State Consistency**: Ensures pool state matches event data
- **Non-negative Values**: Prevents negative liquidity or reserves
- **Tick Range Validation**: Ensures valid tick ranges for liquidity operations
- **Error Tracking**: Collects and reports all validation errors
- **Summary Reporting**: Provides detailed validation summaries

### Verification System

The `verify-state` command compares replayed state against reference snapshots:

```bash
python -m backtest.cli verify-state <REFERENCE> <CANDIDATE> [options]
```

This ensures the replay process maintains accuracy against on-chain data.

## Programmatic Usage

### Basic Usage

```python
from backtest.pool_state import load_pool_state
from backtest.tx_processor import TransactionReplayer

# Load initial state
pool_state = load_pool_state("state/test_state.json")

# Create replayer
replayer = TransactionReplayer(pool_state)

# Replay transactions
replayer.replay_transactions(
    txs_dir=Path("txs"),
    start_page=1,
    end_page=10
)

# Get results
summary = replayer.get_state_summary()
print(f"Processed {summary['processed_events']} events")
print(f"Validation errors: {summary['validation_errors']}")
```

### Advanced Usage with Event Filters

```python
# Load event filter
with open("event_filter.json", "r") as f:
    event_filter = json.load(f)

# Create replayer with filter
replayer = TransactionReplayer(pool_state, event_filter=event_filter)

# Replay with filtering
replayer.replay_transactions(txs_dir=Path("txs"))
```

## Validation Features

The system performs comprehensive validation throughout the replay process:

### State Consistency Checks

- Reserve consistency after swaps
- Liquidity consistency across operations
- Sqrt price consistency with tick calculations
- Fee growth accuracy

### Data Integrity Checks

- Non-negative liquidity and reserves
- Valid tick ranges for liquidity operations
- Proper fee accounting
- Event sequence validation

### Error Reporting

- Detailed error messages with context
- Validation summary with error counts
- Transaction and event-level error tracking
- State comparison reports

## Performance Considerations

### Memory Management

- Efficient tick data structures
- Lazy loading of transaction pages
- Optimized state updates

### Processing Speed

- Parallel event processing where possible
- Efficient tick crossing algorithms
- Minimal state copying

### Storage Optimization

- Compressed transaction storage
- Incremental state updates
- Efficient JSON serialization

## Error Handling

The framework includes robust error handling:

### Transaction Processing Errors

- Invalid event data handling
- Missing timestamp recovery
- Malformed transaction data

### State Validation Errors

- Inconsistent state detection
- Mathematical overflow protection
- Boundary condition validation

### Network and I/O Errors

- RPC timeout handling
- File system error recovery
- Network connectivity issues

## Extensibility

The framework is designed for easy extension:

### Adding New Event Types

1. Create new event class inheriting from base event
2. Implement parsing logic in `from_parsed_json`
3. Add processing logic in `TransactionProcessor`
4. Update event type detection in `_is_pool_relevant_event`

### Custom Validation

- Add validation methods to `TransactionProcessor`
- Implement custom state checks
- Extend error reporting system

### Integration with Other Systems

- Export state to external formats
- Integrate with strategy backtesting systems
- Connect to real-time data feeds

## Best Practices

### Data Management

1. Always verify downloaded snapshots against on-chain data
2. Use event filters to focus on relevant transactions
3. Regularly validate state consistency during replay
4. Maintain backup copies of critical state files

### Performance Optimization

1. Use appropriate page ranges for large transaction sets
2. Enable verbose logging only when debugging
3. Consider memory usage for large pools with extensive tick data
4. Use verification sparingly to avoid unnecessary RPC calls

### Error Handling

1. Always check validation results after replay
2. Investigate and resolve validation errors before proceeding
3. Use summary reports to understand replay results
4. Maintain logs for debugging complex issues

## Troubleshooting

### Common Issues

1. **Timestamp Mismatches**: Ensure pool state timestamp is correct
2. **Missing Events**: Verify event filters include all relevant events
3. **Validation Failures**: Check for data inconsistencies in source transactions
4. **Memory Issues**: Use smaller page ranges for large transaction sets

### Debugging Tips

1. Enable verbose logging with `-v` flag
2. Use `--summary` to get detailed replay information
3. Check event filters with `list-events` command
4. Verify state consistency with `verify-state` command

## Future Enhancements

The framework is designed to support future enhancements:

1. **Position Tracking**: Individual position management and P&L calculation
2. **Strategy Integration**: Direct integration with trading strategies
3. **Real-time Updates**: Live state updates from real-time feeds
4. **Advanced Analytics**: Performance metrics and risk analysis
5. **Multi-Pool Support**: Cross-pool arbitrage and analysis

## Conclusion

The Replaying Event Trigger Backtest Framework provides a comprehensive solution for backtesting CLMM strategies on the Sui network. With its robust event processing, state management, and validation systems, it enables accurate historical analysis and strategy evaluation. The framework's extensible design and comprehensive CLI interface make it suitable for both research and production use cases.

The combination of temporal filtering, event selection, and state validation ensures that backtesting results are accurate and reliable, providing a solid foundation for algorithmic trading strategy development and evaluation.
