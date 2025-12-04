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


def normalize_symbol(symbol: str) -> str:
    """Convert symbol format to Hyperliquid API format (e.g., BTC-PERP -> BTC)."""
    return symbol.split("-")[0]


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
        max_reconnect_attempts: int = 10,
        parquet_path: Optional[str] = None
    ):
        self.api_url = api_url.replace("https://", "wss://").replace("http://", "ws://")
        self.rest_url = api_url.replace("wss://", "https://").replace("ws://", "https://")
        self.symbols = symbols
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.parquet_path = Path(parquet_path) if parquet_path else None

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
        self.use_rest_fallback = False
        self.last_candle_fetch = None

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
            api_symbol = normalize_symbol(symbol)

            # Hyperliquid WebSocket uses {"method": "subscribe", "subscription": {...}} format
            subscriptions = [
                {"method": "subscribe", "subscription": {"type": "l2Book", "coin": api_symbol}},
                {"method": "subscribe", "subscription": {"type": "trades", "coin": api_symbol}},
                {"method": "subscribe", "subscription": {"type": "candle", "coin": api_symbol, "interval": "1m"}},
            ]

            for sub in subscriptions:
                await self.ws_connection.send(json.dumps(sub))
                sub_type = sub["subscription"]["type"]
                logger.info(f"Subscribed to {sub_type} for {symbol} (API: {api_symbol})")

    async def start(self):
        """Start the WebSocket data stream or REST polling fallback."""
        self.is_running = True

        # Load persisted buffer from parquet (if exists)
        self.load_buffer_from_parquet()

        # Start periodic save task (every 5 minutes)
        if self.parquet_path:
            asyncio.create_task(self._periodic_save_task())

        # Try WebSocket first
        ws_connected = await self.connect()

        if not ws_connected:
            logger.warning("WebSocket connection failed, switching to REST API polling fallback")
            self.use_rest_fallback = True
            await self._rest_polling_loop()
            return

        # WebSocket mode
        while self.is_running:
            try:
                if not self.ws_connection:
                    connected = await self.connect()
                    if not connected:
                        # Switch to REST fallback after max retries
                        if self.reconnect_count >= self.max_reconnect_attempts:
                            logger.warning("WebSocket unavailable, switching to REST polling fallback")
                            self.use_rest_fallback = True
                            await self._rest_polling_loop()
                            return
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
                    logger.warning("Max reconnect attempts reached, switching to REST fallback")
                    self.use_rest_fallback = True
                    await self._rest_polling_loop()
                    return

                await asyncio.sleep(self.reconnect_delay)

            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                await asyncio.sleep(self.reconnect_delay)

    async def _handle_message(self, message: str):
        """Parse and route incoming WebSocket messages."""
        try:
            data = json.loads(message)
            channel = data.get("channel")

            # Log all received messages for debugging
            logger.debug(f"Received message - channel: {channel}, data: {str(data)[:200]}")

            if channel == "l2Book":
                await self._handle_orderbook(data)
            elif channel == "trades":
                await self._handle_trade(data)
            elif channel == "candle":
                logger.info(f"Processing candle message: {str(data)[:150]}")
                await self._handle_candle(data)
            elif channel == "funding":
                await self._handle_funding(data)
            elif channel == "openInterest":
                await self._handle_oi(data)
            elif channel == "subscriptionResponse":
                logger.debug(f"Subscription confirmed: {data}")
            else:
                logger.warning(f"Unknown channel: {channel}, data: {str(data)[:200]}")

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

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
        try:
            candle = data.get("data", {})
            symbol = candle.get("s", "")  # Symbol is 's' inside 'data'

            logger.info(f"Parsing candle - symbol: {symbol}, candle data keys: {list(candle.keys())}")

            # Convert symbol back to internal format (BTC -> BTC-PERP)
            # Find matching symbol from our list
            matching_symbol = None
            for sym in self.symbols:
                if normalize_symbol(sym) == symbol:
                    matching_symbol = sym
                    break

            if not matching_symbol:
                matching_symbol = f"{symbol}-PERP"  # Default format

            candle_data = {
                "timestamp": float(candle.get("t", time.time())) / 1000,  # Convert ms to seconds
                "symbol": matching_symbol,
                "open": float(candle.get("o")),
                "high": float(candle.get("h")),
                "low": float(candle.get("l")),
                "close": float(candle.get("c")),
                "volume": float(candle.get("v", 0))
            }

            logger.info(f"✓ Processed candle: {matching_symbol} @ ${candle_data['close']}")

            self.candles_buffer.append(candle_data)

            for callback in self.callbacks["candle"]:
                await callback(candle_data)

        except Exception as e:
            logger.error(f"Error in _handle_candle: {e}", exc_info=True)

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

    async def _rest_polling_loop(self):
        """REST API polling fallback when WebSocket is unavailable."""
        logger.info("Starting REST API polling mode (polls every 60 seconds)")

        while self.is_running:
            try:
                for symbol in self.symbols:
                    api_symbol = normalize_symbol(symbol)

                    # Fetch recent candles
                    end_time = datetime.now()
                    start_time = end_time - timedelta(minutes=60)

                    candles_df = self.get_historical_candles(
                        symbol=api_symbol,
                        interval="1m",
                        start_time=start_time,
                        end_time=end_time
                    )

                    if not candles_df.is_empty():
                        # Add candles to buffer
                        for row in candles_df.iter_rows(named=True):
                            candle_data = {
                                "timestamp": row["timestamp"].timestamp() if hasattr(row["timestamp"], "timestamp") else row["timestamp"],
                                "symbol": symbol,
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": float(row["volume"])
                            }
                            self.candles_buffer.append(candle_data)

                            # Trigger callbacks
                            for callback in self.callbacks["candle"]:
                                await callback(candle_data)

                        logger.info(f"✓ Fetched {len(candles_df)} candles for {symbol} via REST API")

                    # Fetch meta info for funding/OI (if available via REST)
                    # Note: Some data may not be available via REST API
                    # You may need to adjust based on actual Hyperliquid REST endpoints

                # Poll every 60 seconds
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in REST polling loop: {e}")
                await asyncio.sleep(30)

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

        # Save buffer to parquet before stopping
        self.save_buffer_to_parquet()
        logger.info("WebSocket client stopped")

    async def _periodic_save_task(self):
        """Periodically save buffer to parquet every 5 minutes."""
        while self.is_running:
            await asyncio.sleep(300)  # 5 minutes
            self.save_buffer_to_parquet()

    def save_buffer_to_parquet(self):
        """Save candles buffer to parquet file for persistence across restarts."""
        if not self.parquet_path or len(self.candles_buffer) == 0:
            return

        try:
            # Convert deque to list then to polars DataFrame
            candles_list = list(self.candles_buffer)
            df = pl.DataFrame(candles_list)

            # Create directory if it doesn't exist
            self.parquet_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to parquet with compression
            df.write_parquet(self.parquet_path, compression="snappy")
            logger.info(f"✓ Saved {len(candles_list)} candles to {self.parquet_path}")

        except Exception as e:
            logger.error(f"Error saving buffer to parquet: {e}")

    def load_buffer_from_parquet(self):
        """Load candles buffer from parquet file on startup."""
        if not self.parquet_path or not self.parquet_path.exists():
            return

        try:
            # Load from parquet
            df = pl.read_parquet(self.parquet_path)

            # Convert to list of dicts and add to buffer
            candles_list = df.to_dicts()
            for candle in candles_list:
                self.candles_buffer.append(candle)

            logger.info(f"✓ Loaded {len(candles_list)} candles from {self.parquet_path}")

        except Exception as e:
            logger.error(f"Error loading buffer from parquet: {e}")


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
