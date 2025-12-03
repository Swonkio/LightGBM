#!/usr/bin/env python3
"""Quick test that forces REST API fallback mode."""

import asyncio
import sys
from pathlib import Path

# Add hyperliquid_bot/src to path
script_dir = Path(__file__).resolve().parent
src_path = script_dir / "hyperliquid_bot" / "src"

if not src_path.exists():
    src_path = Path("/home/user/LightGBM/hyperliquid_bot/src")
    if not src_path.exists():
        print(f"ERROR: Cannot find hyperliquid_bot/src directory")
        sys.exit(1)

sys.path.insert(0, str(src_path))

from data_pipeline.hyperliquid_client import HyperliquidClient

async def test_rest_mode():
    """Test REST API data collection with reduced retry attempts."""
    print("=" * 60)
    print("Testing REST API Data Collection")
    print("=" * 60)

    # Create client with FEWER reconnect attempts so it switches to REST faster
    client = HyperliquidClient(
        api_url="https://api.hyperliquid-testnet.xyz",
        symbols=["BTC-PERP"],
        buffer_size=1000,
        reconnect_delay=2,  # Shorter delay
        max_reconnect_attempts=2  # Only 2 attempts before switching to REST
    )

    candle_count = 0

    async def on_candle(candle):
        nonlocal candle_count
        candle_count += 1
        if candle_count <= 5:
            print(f"✓ Candle {candle_count}: {candle['symbol']} @ ${candle['close']:.2f}")

    client.add_callback("candle", on_candle)

    print("\nStarting client...")
    print("- WebSocket will try 2 times and fail")
    print("- Will switch to REST API polling")
    print("- REST polls every 60 seconds")
    print("- Waiting 70 seconds for REST to fetch data...\n")

    # Start client
    client_task = asyncio.create_task(client.start())

    # Wait for REST to fetch at least once
    await asyncio.sleep(70)

    # Check results
    candles_df = client.get_buffer_dataframe("candles")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"REST fallback mode: {client.use_rest_fallback}")
    print(f"Candles received: {len(candles_df)}")
    print(f"Callback count: {candle_count}")

    if client.use_rest_fallback and len(candles_df) > 0:
        print("\n✓✓✓ SUCCESS! ✓✓✓")
        print(f"✓ REST API fallback is working!")
        print(f"✓ Latest price: ${candles_df['close'][-1]:.2f}")
        print(f"\n✓ Your bot will now collect data via REST API")
        print(f"✓ Model retraining will have data to work with!")
        success = True
    elif client.use_rest_fallback:
        print("\n⚠ Switched to REST mode but no data received")
        print("  Check your internet connection and API access")
        success = False
    else:
        print("\n⚠ Still trying to reconnect WebSocket")
        print("  The max_reconnect_attempts might need to be lower")
        success = False

    await client.stop()
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(test_rest_mode())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
