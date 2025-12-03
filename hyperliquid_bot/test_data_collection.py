#!/usr/bin/env python3
"""Simple test to verify data collection is working."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_pipeline.hyperliquid_client import HyperliquidClient

async def test_data_collection():
    """Test if the bot can collect data via REST or WebSocket."""
    print("=" * 60)
    print("Testing Hyperliquid Data Collection")
    print("=" * 60)

    # Create client
    client = HyperliquidClient(
        api_url="https://api.hyperliquid-testnet.xyz",
        symbols=["BTC-PERP"],
        buffer_size=1000
    )

    # Track received data
    candle_count = 0

    async def on_candle(candle):
        nonlocal candle_count
        candle_count += 1
        if candle_count <= 3:
            print(f"✓ Candle {candle_count}: {candle['symbol']} @ ${candle['close']:.2f}")

    client.add_callback("candle", on_candle)

    print("\n1. Starting client...")
    print("   - Will try WebSocket first")
    print("   - Falls back to REST API if WebSocket blocked")
    print("   - Waiting 70 seconds for data...\n")

    # Start client
    client_task = asyncio.create_task(client.start())

    # Wait for data collection
    await asyncio.sleep(70)

    # Check results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    candles_df = client.get_buffer_dataframe("candles")

    print(f"Mode: {'REST API' if client.use_rest_fallback else 'WebSocket'}")
    print(f"Candles collected: {len(candles_df)}")
    print(f"Candles via callback: {candle_count}")

    if len(candles_df) > 0:
        print(f"\n✓ SUCCESS: Data collection is working!")
        print(f"  Latest price: ${candles_df['close'][-1]:.2f}")
        print(f"  Timestamp: {candles_df['timestamp'][-1]}")
        print(f"\n✓ The model retrainer will now have data to work with!")
        success = True
    else:
        print(f"\n❌ FAILURE: No data collected")
        print(f"   Check that the API is accessible from this machine")
        success = False

    # Stop client
    await client.stop()

    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(test_data_collection())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
