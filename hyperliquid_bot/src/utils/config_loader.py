"""Configuration loader and validator for Hyperliquid trading bot."""

import json
import os
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class Config:
    """Main configuration container."""

    environment: Dict[str, Any]
    hyperliquid: Dict[str, Any]
    data_pipeline: Dict[str, Any]
    features: Dict[str, Any]
    model: Dict[str, Any]
    llm: Dict[str, Any]
    execution: Dict[str, Any]
    risk: Dict[str, Any]
    backtest: Dict[str, Any]
    monitoring: Dict[str, Any]
    logging: Dict[str, Any]
    performance: Dict[str, Any]

    _raw_config: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, config_path: str = "./config/config.json") -> "Config":
        """Load configuration from JSON file."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r") as f:
            raw_config = json.load(f)

        # Validate required sections
        required_sections = [
            "environment", "hyperliquid", "data_pipeline", "features",
            "model", "llm", "execution", "risk", "backtest",
            "monitoring", "logging", "performance"
        ]

        for section in required_sections:
            if section not in raw_config:
                raise ValueError(f"Missing required config section: {section}")

        config = cls(
            environment=raw_config["environment"],
            hyperliquid=raw_config["hyperliquid"],
            data_pipeline=raw_config["data_pipeline"],
            features=raw_config["features"],
            model=raw_config["model"],
            llm=raw_config["llm"],
            execution=raw_config["execution"],
            risk=raw_config["risk"],
            backtest=raw_config["backtest"],
            monitoring=raw_config["monitoring"],
            logging=raw_config["logging"],
            performance=raw_config["performance"],
            _raw_config=raw_config
        )

        config._validate()
        return config

    def _validate(self):
        """Validate configuration values."""
        # Validate environment mode
        if self.environment["mode"] not in ["testnet", "mainnet"]:
            raise ValueError("Environment mode must be 'testnet' or 'mainnet'")

        # Validate model parameters
        if self.model["num_classes"] != len(self.model["class_labels"]):
            raise ValueError("num_classes must match length of class_labels")

        # Validate risk parameters
        if self.risk["max_leverage"] > 20:
            print("WARNING: max_leverage > 20 is extremely risky")

        if self.risk["kelly_fraction"] > 1.0:
            raise ValueError("kelly_fraction should be <= 1.0")

        # Validate LLM weights
        llm_weight = self.llm["ensemble_weight"]
        lgbm_weight = self.llm["lgbm_weight"]

        if abs(llm_weight + lgbm_weight - 1.0) > 1e-6:
            raise ValueError("LLM + LGBM weights must sum to 1.0")

        # Check private key path
        if self.execution["enabled"]:
            pk_path = Path(self.hyperliquid["private_key_path"])
            if not pk_path.exists():
                print(f"WARNING: Private key file not found: {pk_path}")
                print("Execution is enabled but credentials missing!")

        # Validate Ollama connection if LLM enabled
        if self.llm["enabled"]:
            import requests
            try:
                response = requests.get(
                    f"{self.llm['ollama_host']}/api/tags",
                    timeout=5
                )
                if response.status_code != 200:
                    print("WARNING: Ollama API not responding correctly")
            except Exception as e:
                print(f"WARNING: Cannot connect to Ollama: {e}")

    def get_api_url(self) -> str:
        """Get the appropriate API URL based on environment mode."""
        mode = self.environment["mode"]
        if mode == "testnet":
            return self.environment["testnet_api_url"]
        return self.environment["mainnet_api_url"]

    def is_testnet(self) -> bool:
        """Check if running in testnet mode."""
        return self.environment["mode"] == "testnet"

    def get_model_path(self, symbol: str) -> Path:
        """Get the model file path for a given symbol."""
        base_path = Path(self.model["model_save_path"])
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / f"{symbol.replace('-', '_')}_lgbm.pkl"

    def save(self, config_path: str):
        """Save current configuration to file."""
        with open(config_path, "w") as f:
            json.dump(self._raw_config, f, indent=2)


def load_config(config_path: str = "./config/config.json") -> Config:
    """Convenience function to load configuration."""
    return Config.load(config_path)


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        print("✓ Configuration loaded successfully")
        print(f"  Mode: {config.environment['mode']}")
        print(f"  Symbols: {config.hyperliquid['symbols']}")
        print(f"  API URL: {config.get_api_url()}")
        print(f"  LLM enabled: {config.llm['enabled']}")
        print(f"  Execution enabled: {config.execution['enabled']}")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
