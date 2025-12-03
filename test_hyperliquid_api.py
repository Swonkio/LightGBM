#!/usr/bin/env python3
"""Test Hyperliquid API connectivity and data availability."""

import asyncio
import json
import requests
import websockets
from datetime import datetime, timedelta

TESTNET_REST = "https://api.hyperliquid-testnet.xyz"
TESTNET_WS = "wss://api.hyperliquid-testnet.xyz/ws"

async def test_websocket():
    """Test WebSocket connection and subscriptions."""
    print("=" * 60)
    print("Testing Hyperliquid Testnet WebSocket")
    print("=" * 60)

    try:
        print(f"Connecting to: {TESTNET_WS}")
        async with websockets.connect(TESTNET_WS, ping_interval=20, ping_timeout=10) as ws:
            print("✓ WebSocket connected successfully")

            # Subscribe to candles
            subscription = {
                "method": "subscribe",
                "subscription": {
                    "type": "candle",
                    "coin": "BTC",
                    "interval": "1m"
                }
            }

            print(f"\nSubscribing to 1m candles for BTC...")
            await ws.send(json.dumps(subscription))

            # Wait for messages (timeout after 30 seconds)
            print("Waiting for messages (30 second timeout)...\n")
            message_count = 0

            try:
                async with asyncio.timeout(30):
                    async for message in ws:
                        message_count += 1
                        data = json.loads(message)
                        print(f"Message {message_count}: {json.dumps(data, indent=2)[:500]}")

                        if message_count >= 5:
                            break
            except asyncio.TimeoutError:
                print(f"\n⚠ Timeout after 30 seconds. Received {message_count} messages.")

            if message_count == 0:
                print("❌ No messages received from WebSocket")
                return False
            else:
                print(f"\n✓ Received {message_count} messages")
                return True

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        return False

def test_rest_api():
    """Test REST API for historical data."""
    print("\n" + "=" * 60)
    print("Testing Hyperliquid Testnet REST API")
    print("=" * 60)

    # Test 1: Get meta info
    try:
        print("\n1. Testing /info endpoint (meta)...")
        response = requests.post(
            f"{TESTNET_REST}/info",
            json={"type": "meta"},
            timeout=10
        )

        if response.status_code == 200:
            print(f"✓ Meta endpoint working: {response.status_code}")
            data = response.json()
            print(f"  Available assets: {len(data.get('universe', []))} assets")
        else:
            print(f"❌ Meta endpoint failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Meta endpoint error: {e}")

    # Test 2: Get historical candles
    try:
        print("\n2. Testing candleSnapshot endpoint...")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": "BTC",
                "interval": "1m",
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000)
            }
        }

        response = requests.post(
            f"{TESTNET_REST}/info",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                print(f"✓ Candles endpoint working: received {len(data)} candles")
                print(f"  Latest candle: {data[-1]}")
                return True
            else:
                print(f"⚠ Candles endpoint returned empty data: {data}")
                return False
        else:
            print(f"❌ Candles endpoint failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Candles endpoint error: {e}")
        return False

async def main():
    """Run all tests."""
    print("\n🧪 Hyperliquid Testnet API Diagnostics")
    print(f"Time: {datetime.now()}\n")

    # Test REST API
    rest_works = test_rest_api()

    # Test WebSocket
    ws_works = await test_websocket()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"REST API: {'✓ Working' if rest_works else '❌ Not working'}")
    print(f"WebSocket: {'✓ Working' if ws_works else '❌ Not working'}")

    if not rest_works and not ws_works:
        print("\n⚠ VERDICT: Hyperliquid testnet API appears to be down or unreachable")
    elif not ws_works:
        print("\n⚠ VERDICT: WebSocket not working - check subscription format or API changes")
    elif not rest_works:
        print("\n⚠ VERDICT: REST API not working - but WebSocket is functional")
    else:
        print("\n✓ VERDICT: API is functional - issue may be in bot configuration")

if __name__ == "__main__":
    asyncio.run(main())
