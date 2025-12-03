#!/usr/bin/env python3
"""Test Hyperliquid WebSocket candle subscription with exact bot format."""

import asyncio
import json
import websockets
from datetime import datetime

TESTNET_WS = "wss://api.hyperliquid-testnet.xyz/ws"

async def test_exact_subscription():
    """Test with the exact subscription format used by the bot."""
    print("=" * 60)
    print("Testing WebSocket Candle Subscription")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    try:
        print(f"\nConnecting to: {TESTNET_WS}")
        async with websockets.connect(TESTNET_WS, ping_interval=20, ping_timeout=10) as ws:
            print("✓ WebSocket connected")

            # Use EXACT format from hyperliquid_client.py line 101-103
            subscription = {
                "type": "subscribe",
                "channel": "candle",
                "coin": "BTC",
                "interval": "1m"
            }

            print(f"\nSending subscription: {json.dumps(subscription, indent=2)}")
            await ws.send(json.dumps(subscription))
            print("✓ Subscription sent")

            print("\nWaiting for messages (60 second timeout)...\n")
            message_count = 0

            try:
                async with asyncio.timeout(60):
                    async for message in ws:
                        message_count += 1
                        data = json.loads(message)

                        print(f"\n--- Message {message_count} ---")
                        print(json.dumps(data, indent=2)[:1000])

                        if message_count >= 5:
                            print("\n✓ Received 5 messages, test successful")
                            return True

            except asyncio.TimeoutError:
                print(f"\n⏱ Timeout after 60 seconds")
                print(f"Messages received: {message_count}")

            if message_count == 0:
                print("\n❌ PROBLEM: No messages received")
                print("This explains why retrainer has no data!")
                return False
            else:
                print(f"\n✓ Received {message_count} messages")
                return True

    except Exception as e:
        print(f"\n❌ WebSocket error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def test_multiple_subscriptions():
    """Test subscribing to multiple channels like the bot does."""
    print("\n" + "=" * 60)
    print("Testing Multiple Channel Subscriptions")
    print("=" * 60)

    try:
        print(f"\nConnecting to: {TESTNET_WS}")
        async with websockets.connect(TESTNET_WS) as ws:
            print("✓ Connected")

            # Subscribe to multiple channels like the bot
            channels = ["l2Book", "trades", "candle"]

            for channel in channels:
                if channel == "candle":
                    sub = {"type": "subscribe", "channel": channel, "coin": "BTC", "interval": "1m"}
                else:
                    sub = {"type": "subscribe", "channel": channel, "coin": "BTC"}

                print(f"Subscribing to {channel}...")
                await ws.send(json.dumps(sub))
                await asyncio.sleep(0.5)

            print("\n✓ All subscriptions sent")
            print("Listening for 30 seconds...\n")

            channel_counts = {}

            try:
                async with asyncio.timeout(30):
                    async for message in ws:
                        data = json.loads(message)
                        channel = data.get("channel", "unknown")

                        if channel not in channel_counts:
                            channel_counts[channel] = 0
                        channel_counts[channel] += 1

                        print(f"Received: {channel} (count: {channel_counts[channel]})")

            except asyncio.TimeoutError:
                pass

            print("\n📊 Summary:")
            for channel, count in channel_counts.items():
                print(f"  {channel}: {count} messages")

            if "candle" not in channel_counts or channel_counts["candle"] == 0:
                print("\n❌ PROBLEM: No candle messages received!")
                return False

            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    """Run all tests."""
    print("\n🧪 Hyperliquid WebSocket Diagnostics")
    print(f"Testing from: modeltrainer environment")
    print()

    # Test 1: Single candle subscription
    test1 = await test_exact_subscription()

    # Test 2: Multiple subscriptions
    test2 = await test_multiple_subscriptions()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    if test1 and test2:
        print("✓ WebSocket working correctly")
        print("  Issue must be in bot configuration or data flow")
    elif not test1:
        print("❌ WebSocket candle subscription not working")
        print("  This explains 'No new data for retraining'")
    else:
        print("⚠ Partial success - needs investigation")

if __name__ == "__main__":
    asyncio.run(main())
