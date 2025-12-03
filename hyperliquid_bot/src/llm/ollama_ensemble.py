"""
Ollama LLM ensemble layer for CPU-based inference.

Uses:
- Main model: Llama 3.1 70B Instruct Q4_K_M (~10-14 t/s on Threadripper 3960X)
- Fast model: Gemma-2-27B or Phi-3-Medium (~30-50 t/s)

Provides:
- Macro market bias analysis (every 15-30 min)
- Order flow narration (every 1 min)
- Confidence-weighted ensemble with LightGBM
"""

import json
import time
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

import requests
import numpy as np

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for Ollama API running locally on CPU."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: int = 120
    ):
        self.host = host
        self.timeout = timeout
        self.session = requests.Session()

        # Verify Ollama is running
        self._check_connection()

    def _check_connection(self):
        """Check if Ollama API is accessible."""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                logger.info(f"✓ Connected to Ollama. Available models: {model_names}")
            else:
                logger.warning("Ollama API responded with non-200 status")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: systemctl status ollama")

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 512
    ) -> Tuple[str, float]:
        """
        Generate text using Ollama model.

        Args:
            model: Model name (e.g., "llama3.1:70b-instruct-q4_K_M")
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate

        Returns:
            (generated_text, inference_time_seconds)
        """
        start_time = time.time()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
        }

        try:
            response = self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "")
            inference_time = time.time() - start_time

            logger.debug(f"Ollama inference: {inference_time:.2f}s ({len(generated_text)} chars)")
            return generated_text, inference_time

        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timeout after {self.timeout}s")
            return "", 0.0
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return "", 0.0


class LLMEnsemble:
    """
    LLM ensemble system for trading signal enhancement.

    Combines:
    - 70B model for deep macro analysis
    - 27B model for fast order flow insights
    - LightGBM predictions with confidence weighting
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        main_model_config: Dict = None,
        fast_model_config: Dict = None,
        ensemble_weight: float = 0.4,
        lgbm_weight: float = 0.6,
        override_threshold: float = 0.85
    ):
        self.client = OllamaClient(host=ollama_host)

        # Model configurations
        self.main_model = main_model_config or {
            "name": "llama3.1:70b-instruct-q4_K_M",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 512,
            "interval_minutes": 30
        }

        self.fast_model = fast_model_config or {
            "name": "gemma2:27b",
            "temperature": 0.2,
            "top_p": 0.85,
            "max_tokens": 256,
            "interval_minutes": 1
        }

        # Ensemble weights
        self.ensemble_weight = ensemble_weight
        self.lgbm_weight = lgbm_weight
        self.override_threshold = override_threshold

        # State tracking
        self.last_macro_analysis = None
        self.last_macro_time = 0
        self.last_orderflow_analysis = None
        self.last_orderflow_time = 0

        # Inference cache
        self.macro_bias_cache = {"bias": "neutral", "confidence": 0.5, "reasoning": ""}
        self.orderflow_cache = {"momentum": "neutral", "confidence": 0.5}

        # Background thread for periodic macro analysis
        self.macro_thread = None
        self.is_running = False

    def analyze_macro(
        self,
        symbol: str,
        price_change_pct: float,
        funding_rate: float,
        oi_change_pct: float,
        whale_flow: str,
        volume_profile: str
    ) -> Dict:
        """
        Deep macro market analysis using 70B model.

        Args:
            symbol: Trading symbol
            price_change_pct: 6-hour price change
            funding_rate: Current funding rate
            oi_change_pct: Open interest change
            whale_flow: Description of large trades
            volume_profile: Volume distribution

        Returns:
            {"bias": "bullish|neutral|bearish", "confidence": 0-1, "reasoning": "..."}
        """
        prompt = f"""You are a crypto derivatives market analyst. Analyze the following 6-hour market summary and provide a trading bias (bullish/neutral/bearish) with confidence 0-1.

Market Data:
- Symbol: {symbol}
- Price change 6h: {price_change_pct:.2f}%
- Funding rate: {funding_rate:.4f}%
- Open Interest change: {oi_change_pct:.2f}%
- Large trade flow: {whale_flow}
- Volume profile: {volume_profile}

Provide response in JSON: {{"bias": "bullish|neutral|bearish", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

Response:"""

        logger.info("Running macro analysis with 70B model...")
        response, inference_time = self.client.generate(
            model=self.main_model["name"],
            prompt=prompt,
            temperature=self.main_model["temperature"],
            top_p=self.main_model["top_p"],
            max_tokens=self.main_model["max_tokens"]
        )

        # Parse JSON response
        try:
            # Extract JSON from response
            result = self._extract_json(response)

            if result:
                self.macro_bias_cache = result
                self.last_macro_time = time.time()
                logger.info(f"✓ Macro analysis: {result['bias']} (confidence: {result['confidence']:.2f})")
                return result
            else:
                logger.warning("Failed to parse macro analysis, using default")
                return {"bias": "neutral", "confidence": 0.5, "reasoning": "Parse error"}

        except Exception as e:
            logger.error(f"Error parsing macro analysis: {e}")
            return {"bias": "neutral", "confidence": 0.5, "reasoning": "Error"}

    def analyze_orderflow(
        self,
        cum_delta: float,
        buy_sell_ratio: float,
        imbalance: float,
        large_trades: str
    ) -> Dict:
        """
        Fast order flow analysis using 27B model.

        Args:
            cum_delta: Cumulative volume delta
            buy_sell_ratio: Buy/sell volume ratio
            imbalance: Orderbook imbalance
            large_trades: Recent large trades description

        Returns:
            {"momentum": "strong_buy|weak_buy|neutral|weak_sell|strong_sell", "confidence": 0-1}
        """
        prompt = f"""You are monitoring real-time order flow. Based on the last 60 seconds of data, assess short-term momentum.

Data:
- Cumulative delta: {cum_delta:.2f}
- Buy/sell ratio: {buy_sell_ratio:.2f}
- Orderbook imbalance: {imbalance:.2f}
- Recent large trades: {large_trades}

Respond with JSON: {{"momentum": "strong_buy|weak_buy|neutral|weak_sell|strong_sell", "confidence": 0.0-1.0}}

Response:"""

        response, inference_time = self.client.generate(
            model=self.fast_model["name"],
            prompt=prompt,
            temperature=self.fast_model["temperature"],
            top_p=self.fast_model["top_p"],
            max_tokens=self.fast_model["max_tokens"]
        )

        # Parse JSON response
        try:
            result = self._extract_json(response)

            if result:
                self.orderflow_cache = result
                self.last_orderflow_time = time.time()
                return result
            else:
                return {"momentum": "neutral", "confidence": 0.5}

        except Exception as e:
            logger.error(f"Error parsing orderflow analysis: {e}")
            return {"momentum": "neutral", "confidence": 0.5}

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON object from LLM response."""
        try:
            # Try direct parse
            return json.loads(text)
        except:
            # Try to find JSON in text
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                try:
                    return json.loads(json_str)
                except:
                    pass

        return None

    def combine_signals(
        self,
        lgbm_probs: np.ndarray,
        use_llm: bool = True
    ) -> Tuple[int, float, Dict]:
        """
        Combine LightGBM and LLM signals into final prediction.

        Args:
            lgbm_probs: LightGBM class probabilities [short, flat, long]
            use_llm: Whether to include LLM signals

        Returns:
            (final_class, final_confidence, details)
        """
        # LightGBM prediction
        lgbm_class = np.argmax(lgbm_probs)
        lgbm_conf = lgbm_probs[lgbm_class]

        if not use_llm:
            return lgbm_class, lgbm_conf, {"source": "lgbm_only"}

        # Map LLM bias to class weights
        macro_bias = self.macro_bias_cache["bias"]
        macro_conf = self.macro_bias_cache["confidence"]

        orderflow_momentum = self.orderflow_cache["momentum"]
        orderflow_conf = self.orderflow_cache["confidence"]

        # Convert LLM signals to class probabilities
        llm_probs = np.zeros(3)  # [short, flat, long]

        # Macro bias
        if macro_bias == "bullish":
            llm_probs[2] += macro_conf  # Long
        elif macro_bias == "bearish":
            llm_probs[0] += macro_conf  # Short
        else:
            llm_probs[1] += macro_conf  # Flat

        # Order flow momentum
        momentum_map = {
            "strong_buy": (0, 0, 1.0),
            "weak_buy": (0, 0.3, 0.7),
            "neutral": (0.3, 0.4, 0.3),
            "weak_sell": (0.7, 0.3, 0),
            "strong_sell": (1.0, 0, 0)
        }

        if orderflow_momentum in momentum_map:
            weights = momentum_map[orderflow_momentum]
            llm_probs += np.array(weights) * orderflow_conf

        # Normalize LLM probabilities
        llm_probs = llm_probs / (llm_probs.sum() + 1e-10)

        # Ensemble: weighted combination
        ensemble_probs = (
            self.lgbm_weight * lgbm_probs +
            self.ensemble_weight * llm_probs
        )

        final_class = np.argmax(ensemble_probs)
        final_conf = ensemble_probs[final_class]

        # LLM override logic: if LLM has very high confidence, override LightGBM
        llm_class = np.argmax(llm_probs)
        llm_max_conf = llm_probs[llm_class]

        if llm_max_conf > self.override_threshold:
            logger.info(f"LLM override triggered: {llm_max_conf:.2f} > {self.override_threshold}")
            final_class = llm_class
            final_conf = llm_max_conf

        details = {
            "lgbm_class": lgbm_class,
            "lgbm_conf": float(lgbm_conf),
            "llm_class": llm_class,
            "llm_conf": float(llm_max_conf),
            "ensemble_class": final_class,
            "ensemble_conf": float(final_conf),
            "macro_bias": macro_bias,
            "orderflow_momentum": orderflow_momentum
        }

        return final_class, final_conf, details

    def should_run_macro_analysis(self) -> bool:
        """Check if it's time to run macro analysis."""
        interval = self.main_model["inference_interval_minutes"] * 60
        return (time.time() - self.last_macro_time) >= interval

    def should_run_orderflow_analysis(self) -> bool:
        """Check if it's time to run orderflow analysis."""
        interval = self.fast_model["inference_interval_minutes"] * 60
        return (time.time() - self.last_orderflow_time) >= interval

    def get_cache_age_seconds(self) -> Dict[str, float]:
        """Get age of cached analyses."""
        return {
            "macro": time.time() - self.last_macro_time if self.last_macro_time else float('inf'),
            "orderflow": time.time() - self.last_orderflow_time if self.last_orderflow_time else float('inf')
        }


if __name__ == "__main__":
    # Test the LLM ensemble
    ensemble = LLMEnsemble()

    # Test macro analysis
    macro_result = ensemble.analyze_macro(
        symbol="BTC-PERP",
        price_change_pct=2.5,
        funding_rate=0.01,
        oi_change_pct=5.0,
        whale_flow="Large buy orders at 40000",
        volume_profile="Heavy accumulation below 39800"
    )

    print(f"\n✓ Macro analysis: {macro_result}")

    # Test order flow analysis
    orderflow_result = ensemble.analyze_orderflow(
        cum_delta=150000,
        buy_sell_ratio=1.3,
        imbalance=0.2,
        large_trades="3 large buys > 1 BTC in last minute"
    )

    print(f"\n✓ Order flow analysis: {orderflow_result}")

    # Test signal combination
    lgbm_probs = np.array([0.2, 0.3, 0.5])  # [short, flat, long]
    final_class, final_conf, details = ensemble.combine_signals(lgbm_probs)

    print(f"\n✓ Final signal: class={final_class}, confidence={final_conf:.2f}")
    print(f"  Details: {details}")
