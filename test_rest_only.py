#!/usr/bin/env python3
"""Test REST API data collection by forcing REST mode (skip WebSocket entirely)."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

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

async def test_rest_only():
    """Test REST API polling directly without trying WebSocket."""
    print("=" * 60)
    print("Testing REST API Data Collection (WebSocket disabled)")
    print("=" * 60)

    # Create client
    client = HyperliquidClient(
        api_url="https://api.hyperliquid-testnet.xyz",
        symbols=["BTC-PERP"],
        buffer_size=1000
    )

    # Force REST mode immediately (skip WebSocket)
    client.use_rest_fallback = True
    client.is_running = True

    candle_count = 0

    async def on_candle(candle):
        nonlocal candle_count
        candle_count += 1
        if candle_count <= 5:
            print(f"✓ Candle {candle_count}: {candle['symbol']} @ ${candle['close']:.2f}")

    client.add_callback("candle", on_candle)

    print("\n🔧 Forcing REST API mode (skipping WebSocket)")
    print("📡 Starting REST polling...")
    print("⏱️  Waiting 70 seconds for first poll...\n")

    # Start REST polling directly
    polling_task = asyncio.create_task(client._rest_polling_loop())

    # Wait for at least one poll cycle
    await asyncio.sleep(70)

    # Stop
    client.is_running = False
    await asyncio.sleep(1)  # Give it time to stop

    # Check results
    candles_df = client.get_buffer_dataframe("candles")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Candles in buffer: {len(candles_df)}")
    print(f"Candles via callback: {candle_count}")

    if len(candles_df) > 0:
        print(f"\n✓✓✓ SUCCESS! ✓✓✓")
        print(f"✓ REST API polling is working!")
        print(f"✓ Collected {len(candles_df)} candles")
        print(f"✓ Latest price: ${candles_df['close'][-1]:.2f}")
        print(f"\n💡 The bot will work with REST API fallback!")
        print(f"💡 To use it, set max_reconnect_attempts to 1 or 2 in config")
        return True
    else:
        print(f"\n❌ FAILURE: No candles collected")
        print(f"   This is unexpected since direct API test worked")
        print(f"   Check for errors above")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_rest_only())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
