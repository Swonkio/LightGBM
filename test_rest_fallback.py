#!/usr/bin/env python3
"""Test REST API fallback for Hyperliquid data collection."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "hyperliquid_bot" / "src"))

from data_pipeline.hyperliquid_client import HyperliquidClient

async def test_rest_fallback():
    """Test the REST API fallback mechanism."""
    print("=" * 60)
    print("Testing Hyperliquid REST API Fallback")
    print("=" * 60)

    # Create client
    client = HyperliquidClient(
        api_url="https://api.hyperliquid-testnet.xyz",
        symbols=["BTC-PERP"],
        buffer_size=1000
    )

    # Track candle count
    candle_count = 0

    def on_candle(candle):
        nonlocal candle_count
        candle_count += 1
        print(f"Candle {candle_count}: {candle['symbol']} @ {candle['close']:.2f} (ts: {candle['timestamp']})")

    client.add_callback("candle", on_candle)

    print("\nStarting client (will use REST fallback due to proxy)...")
    print("WebSocket will fail, then switch to REST polling...")
    print("Waiting 15 seconds for REST polling to fetch data...\n")

    # Start client in background
    client_task = asyncio.create_task(client.start())

    # Wait 15 seconds to allow REST polling to run at least once
    await asyncio.sleep(15)

    # Stop client
    await client.stop()

    # Get buffer data
    candles_df = client.get_buffer_dataframe("candles")

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Candles received via callback: {candle_count}")
    print(f"Candles in buffer: {len(candles_df)}")
    print(f"REST fallback mode: {client.use_rest_fallback}")

    if len(candles_df) > 0:
        print("\n✓ SUCCESS: REST API fallback is working!")
        print(f"  Latest candle close: ${candles_df['close'][-1]:.2f}")
        print(f"  Timestamp: {candles_df['timestamp'][-1]}")
        return True
    else:
        print("\n❌ FAILURE: No data received via REST API")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_rest_fallback())
    sys.exit(0 if success else 1)
