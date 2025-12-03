"""Hyperliquid WebSocket and REST API client for real-time data ingestion."""

import asyncio
import json
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Deque
import logging

import websockets
import requests
import pandas as pd
import polars as pl
from pathlib import Path

logger = logging.getLogger(__name__)


class HyperliquidClient:
    """
    Real-time Hyperliquid data client with WebSocket and REST support.

    Handles:
    - L2 orderbook snapshots and updates
    - Trade stream
    - 1-minute candles
    - Funding rates
    - Open interest
    - User state (positions, balances)
    """

    def __init__(
        self,
        api_url: str,
        symbols: List[str],
        buffer_size: int = 100000,
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 10
    ):
        self.api_url = api_url.replace("https://", "wss://").replace("http://", "ws://")
        self.rest_url = api_url.replace("wss://", "https://").replace("ws://", "https://")
        self.symbols = symbols
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        # Data buffers
        self.orderbook_buffer: Deque[Dict] = deque(maxlen=buffer_size)
        self.trades_buffer: Deque[Dict] = deque(maxlen=buffer_size)
        self.candles_buffer: Deque[Dict] = deque(maxlen=buffer_size)
        self.funding_buffer: Deque[Dict] = deque(maxlen=buffer_size)
        self.oi_buffer: Deque[Dict] = deque(maxlen=buffer_size)

        # Current state
        self.current_orderbook: Dict[str, Dict] = {}
        self.current_funding: Dict[str, float] = {}
        self.current_oi: Dict[str, float] = {}

        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            "orderbook": [],
            "trade": [],
            "candle": [],
            "funding": [],
            "oi": []
        }

        # Connection state
        self.ws_connection = None
        self.is_running = False
        self.reconnect_count = 0

    def add_callback(self, event_type: str, callback: Callable):
        """Register a callback for specific event types."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)

    async def connect(self):
        """Establish WebSocket connection to Hyperliquid."""
        ws_url = f"{self.api_url}/ws"
        logger.info(f"Connecting to Hyperliquid WebSocket: {ws_url}")

        try:
            self.ws_connection = await websockets.connect(ws_url)
            self.reconnect_count = 0
            logger.info("✓ WebSocket connected successfully")

            # Subscribe to channels
            await self._subscribe_channels()
            return True

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def _subscribe_channels(self):
        """Subscribe to all required data channels."""
        for symbol in self.symbols:
            subscriptions = [
                {"type": "subscribe", "channel": "l2Book", "coin": symbol},
                {"type": "subscribe", "channel": "trades", "coin": symbol},
                {"type": "subscribe", "channel": "candle", "coin": symbol, "interval": "1m"},
                {"type": "subscribe", "channel": "funding", "coin": symbol},
                {"type": "subscribe", "channel": "openInterest", "coin": symbol}
            ]

            for sub in subscriptions:
                await self.ws_connection.send(json.dumps(sub))
                logger.info(f"Subscribed to {sub['channel']} for {symbol}")

    async def start(self):
        """Start the WebSocket data stream."""
        self.is_running = True

        while self.is_running:
            try:
                if not self.ws_connection:
                    connected = await self.connect()
                    if not connected:
                        await asyncio.sleep(self.reconnect_delay)
                        continue

                # Process incoming messages
                async for message in self.ws_connection:
                    await self._handle_message(message)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, attempting reconnect...")
                self.ws_connection = None
                self.reconnect_count += 1

                if self.reconnect_count >= self.max_reconnect_attempts:
                    logger.error("Max reconnect attempts reached, stopping...")
                    self.is_running = False
                    break

                await asyncio.sleep(self.reconnect_delay)

            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                await asyncio.sleep(self.reconnect_delay)

    async def _handle_message(self, message: str):
        """Parse and route incoming WebSocket messages."""
        try:
            data = json.loads(message)
            channel = data.get("channel")

            if channel == "l2Book":
                await self._handle_orderbook(data)
            elif channel == "trades":
                await self._handle_trade(data)
            elif channel == "candle":
                await self._handle_candle(data)
            elif channel == "funding":
                await self._handle_funding(data)
            elif channel == "openInterest":
                await self._handle_oi(data)

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_orderbook(self, data: Dict):
        """Process L2 orderbook updates."""
        symbol = data.get("coin")
        timestamp = time.time()

        # Extract bids and asks
        bids = data.get("data", {}).get("levels", [[]])[0]  # Buy side
        asks = data.get("data", {}).get("levels", [[]])[1]  # Sell side

        if not bids or not asks:
            return

        # Store current orderbook state
        self.current_orderbook[symbol] = {
            "timestamp": timestamp,
            "bids": bids,
            "asks": asks,
            "best_bid": float(bids[0]["px"]) if bids else 0,
            "best_ask": float(asks[0]["px"]) if asks else 0,
            "bid_size": float(bids[0]["sz"]) if bids else 0,
            "ask_size": float(asks[0]["sz"]) if asks else 0
        }

        # Buffer for feature calculation
        self.orderbook_buffer.append({
            "timestamp": timestamp,
            "symbol": symbol,
            **self.current_orderbook[symbol]
        })

        # Trigger callbacks
        for callback in self.callbacks["orderbook"]:
            await callback(self.current_orderbook[symbol])

    async def _handle_trade(self, data: Dict):
        """Process trade executions."""
        symbol = data.get("coin")
        trades = data.get("data", [])

        for trade in trades:
            trade_data = {
                "timestamp": float(trade.get("time", time.time())),
                "symbol": symbol,
                "price": float(trade.get("px")),
                "size": float(trade.get("sz")),
                "side": trade.get("side"),  # "B" or "A" (buyer or seller was aggressor)
            }

            self.trades_buffer.append(trade_data)

            for callback in self.callbacks["trade"]:
                await callback(trade_data)

    async def _handle_candle(self, data: Dict):
        """Process 1-minute candle data."""
        symbol = data.get("coin")
        candle = data.get("data", {})

        candle_data = {
            "timestamp": float(candle.get("t", time.time())),
            "symbol": symbol,
            "open": float(candle.get("o")),
            "high": float(candle.get("h")),
            "low": float(candle.get("l")),
            "close": float(candle.get("c")),
            "volume": float(candle.get("v"))
        }

        self.candles_buffer.append(candle_data)

        for callback in self.callbacks["candle"]:
            await callback(candle_data)

    async def _handle_funding(self, data: Dict):
        """Process funding rate updates."""
        symbol = data.get("coin")
        funding_rate = float(data.get("data", {}).get("fundingRate", 0))

        self.current_funding[symbol] = funding_rate

        funding_data = {
            "timestamp": time.time(),
            "symbol": symbol,
            "funding_rate": funding_rate
        }

        self.funding_buffer.append(funding_data)

        for callback in self.callbacks["funding"]:
            await callback(funding_data)

    async def _handle_oi(self, data: Dict):
        """Process open interest updates."""
        symbol = data.get("coin")
        open_interest = float(data.get("data", {}).get("oi", 0))

        self.current_oi[symbol] = open_interest

        oi_data = {
            "timestamp": time.time(),
            "symbol": symbol,
            "open_interest": open_interest
        }

        self.oi_buffer.append(oi_data)

        for callback in self.callbacks["oi"]:
            await callback(oi_data)

    def get_historical_candles(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pl.DataFrame:
        """
        Fetch historical candle data via REST API.

        Args:
            symbol: Trading pair (e.g., "BTC-PERP")
            interval: Candle interval (default "1m")
            start_time: Start datetime
            end_time: End datetime

        Returns:
            Polars DataFrame with OHLCV data
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)
        if end_time is None:
            end_time = datetime.now()

        endpoint = f"{self.rest_url}/info"

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000)
            }
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            candles = []

            for candle in data:
                candles.append({
                    "timestamp": candle["t"] / 1000,
                    "open": float(candle["o"]),
                    "high": float(candle["h"]),
                    "low": float(candle["l"]),
                    "close": float(candle["c"]),
                    "volume": float(candle["v"])
                })

            df = pl.DataFrame(candles)
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

            logger.info(f"Fetched {len(df)} historical candles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical candles: {e}")
            return pl.DataFrame()

    def get_buffer_dataframe(self, buffer_type: str) -> pl.DataFrame:
        """Convert buffer to Polars DataFrame."""
        buffers = {
            "orderbook": self.orderbook_buffer,
            "trades": self.trades_buffer,
            "candles": self.candles_buffer,
            "funding": self.funding_buffer,
            "oi": self.oi_buffer
        }

        if buffer_type not in buffers:
            raise ValueError(f"Unknown buffer type: {buffer_type}")

        buffer = buffers[buffer_type]
        if not buffer:
            return pl.DataFrame()

        return pl.DataFrame(list(buffer))

    async def stop(self):
        """Stop the WebSocket connection."""
        self.is_running = False
        if self.ws_connection:
            await self.ws_connection.close()
        logger.info("WebSocket client stopped")


if __name__ == "__main__":
    # Test the client
    async def test_client():
        client = HyperliquidClient(
            api_url="https://api.hyperliquid-testnet.xyz",
            symbols=["BTC-PERP"]
        )

        # Add test callbacks
        async def on_trade(trade):
            print(f"Trade: {trade['price']} @ {trade['size']}")

        client.add_callback("trade", on_trade)

        # Run for 30 seconds
        task = asyncio.create_task(client.start())
        await asyncio.sleep(30)
        await client.stop()

    asyncio.run(test_client())
