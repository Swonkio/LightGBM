# Hyperliquid Perpetuals Trading Bot

A production-ready, CPU-optimized algorithmic trading system for Hyperliquid perpetual futures. Combines incremental LightGBM machine learning with local LLM ensemble (Ollama) for intelligent trade signals.

## 🚀 Key Features

### Hardware Optimization
- **CPU-Only Operation**: No GPU required - optimized for AMD Threadripper 3960X (24c/48t)
- **Memory Efficient**: Stays under 200GB RAM with 10M+ row datasets
- **Fast Retraining**: Sub-3-minute LightGBM retrains on 15M rows
- **Local LLM Inference**: Llama 3.1 70B @ 10-14 t/s via Ollama

### Machine Learning
- **Incremental LightGBM**: True online learning with `continue_training`
- **45+ Real-Time Features**: Orderbook, volume, volatility, funding, technical indicators
- **Multiclass Prediction**: Long/Short/Flat signals
- **Walk-Forward Backtesting**: Realistic performance validation

### LLM Ensemble Layer
- **Dual Model System**:
  - Main: Llama 3.1 70B Instruct (Q4_K_M) for macro analysis every 30min
  - Fast: Gemma-2-27B for order flow analysis every 1min
- **Confidence Weighting**: LightGBM × 0.6 + LLM × 0.4 with override logic
- **CPU-Only Inference**: All via Ollama local API

### Risk Management
- **Kelly Criterion Position Sizing**: Dynamic allocation based on edge
- **Multi-Level Stops**: Hard stop-loss, take-profit, trailing stops
- **Drawdown Protection**: Automatic trading halt at 15% drawdown
- **Daily Loss Limits**: Circuit breaker at 5% daily loss

### Execution
- **Post-Only Orders (ALO)**: Maker rebate optimization
- **Testnet/Mainnet Toggle**: Safe testing environment
- **Atomic Model Swaps**: Zero-downtime model updates
- **Discord Alerts**: Real-time trade and error notifications

## 📁 Project Structure

```
hyperliquid_bot/
├── config/
│   └── config.json              # Main configuration file
├── data/
│   ├── live/                    # Real-time data buffers
│   ├── historical/              # Historical candles
│   ├── models/                  # Saved LightGBM models
│   └── backtests/               # Backtest results
├── src/
│   ├── data_pipeline/
│   │   └── hyperliquid_client.py    # WebSocket + REST client
│   ├── features/
│   │   └── feature_engine.py        # 45+ feature computation
│   ├── models/
│   │   └── lgbm_trainer.py          # Incremental LightGBM
│   ├── llm/
│   │   └── ollama_ensemble.py       # LLM ensemble layer
│   ├── execution/
│   │   └── order_manager.py         # Order execution
│   ├── risk/
│   │   └── risk_manager.py          # Risk & position sizing
│   ├── backtest/
│   │   └── backtester.py            # Walk-forward backtest
│   ├── monitoring/
│   │   └── alerts.py                # Discord alerts & health checks
│   ├── utils/
│   │   └── config_loader.py         # Config management
│   ├── scripts/
│   │   └── run_backtest.py          # Backtest runner
│   └── main.py                      # Main orchestrator
├── logs/                        # Log files
├── start.sh                     # Production launcher
├── stop.sh                      # Graceful shutdown
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Installation

### 1. System Requirements

**Hardware:**
- CPU: AMD Threadripper 3960X or equivalent (24+ cores recommended)
- RAM: 256GB (minimum 64GB for smaller datasets)
- Storage: 1TB+ NVMe SSD
- Network: Low-latency connection for real-time trading

**Software:**
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.11+
- Ollama (for local LLM inference)

### 2. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.1:70b-instruct-q4_K_M  # Main model (~40GB)
ollama pull gemma2:27b                     # Fast model (~16GB)

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Verify
curl http://localhost:11434/api/tags
```

### 3. Install Python Dependencies

```bash
cd hyperliquid_bot

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure the Bot

**Edit `config/config.json`:**

```json
{
  "environment": {
    "mode": "testnet"  // Start with testnet!
  },

  "hyperliquid": {
    "wallet_address": "YOUR_WALLET_ADDRESS",
    "private_key_path": "./config/private_key.txt",
    "symbols": ["BTC-PERP"]
  },

  "execution": {
    "enabled": false  // Set to true when ready to trade
  },

  "monitoring": {
    "discord_webhook_url": "YOUR_WEBHOOK_URL"  // Optional
  }
}
```

**Add your private key:**

```bash
echo "YOUR_PRIVATE_KEY" > config/private_key.txt
chmod 600 config/private_key.txt
```

## 🚦 Usage

### Quick Start

```bash
# Start in dry-run mode (no execution)
./start.sh dry-run

# Monitor logs
tail -f logs/trading_bot.log

# Stop the bot
./stop.sh
```

### Live Trading

```bash
# 1. Test on testnet first
# Set config.json: "mode": "testnet", "execution": {"enabled": true}
./start.sh live

# 2. Once validated, switch to mainnet
# Set config.json: "mode": "mainnet"
./start.sh live
```

### Run Backtest

```bash
# Full year backtest
python -m src.scripts.run_backtest --start 2024-01-01 --end 2024-12-01

# Results saved to:
# - data/backtests/equity_curve_<timestamp>.csv
# - data/backtests/trades_<timestamp>.csv
```

## 📊 Features Explained

### 45+ Real-Time Features

**Orderbook Microstructure:**
- Micro-price (weighted mid)
- Spread in basis points
- Order flow imbalance
- Book depth ratio

**Volume & Order Flow:**
- Volume delta (buy - sell)
- Cumulative delta
- Volume MA ratios (15/30/60)
- Buy/sell volume ratio

**Volatility Estimators:**
- Realized volatility (15min, 1hr)
- Parkinson volatility (high-low range)
- Garman-Klass volatility

**Funding & Open Interest:**
- Current funding rate
- Funding rate EMA
- Open interest delta %

**Technical Indicators:**
- RSI (14, 28)
- Bollinger Bands distance
- ATR normalized
- MACD + signal

**Temporal Encoding:**
- Hour sin/cos (cyclical)
- Day of week sin/cos

**Lag & Rolling Features:**
- Price lags (1, 5, 15, 30)
- Return lags
- Rolling mean/skew

## 🧠 LLM Ensemble Strategy

### Macro Analysis (Every 30 min)

**Llama 3.1 70B analyzes:**
- 6-hour price change
- Funding rate trends
- Open interest dynamics
- Whale trade flow
- Volume profile

**Output:** `{bias: "bullish|neutral|bearish", confidence: 0-1}`

### Order Flow Analysis (Every 1 min)

**Gemma-2-27B monitors:**
- Cumulative delta
- Buy/sell ratio
- Orderbook imbalance
- Large recent trades

**Output:** `{momentum: "strong_buy|weak_buy|neutral|weak_sell|strong_sell", confidence: 0-1}`

### Signal Combination

```
final_signal = (LightGBM × 0.6) + (LLM × 0.4)

if LLM_confidence > 0.85:
    final_signal = LLM_signal  // Override
```

## ⚙️ Configuration Reference

### Model Training

```json
"model": {
  "incremental_learning": true,
  "retrain_interval_minutes": 5,
  "max_train_samples": 15000000,
  "training_params": {
    "num_threads": 24,  // Use all CPU cores
    "num_leaves": 127,
    "learning_rate": 0.03
  }
}
```

### Risk Parameters

```json
"risk": {
  "max_position_size_usd": 10000,
  "max_leverage": 8.0,
  "kelly_fraction": 0.25,
  "stop_loss_pct": 2.0,
  "take_profit_pct": 4.0,
  "trailing_stop_pct": 1.5
}
```

### LLM Settings

```json
"llm": {
  "main_model": {
    "name": "llama3.1:70b-instruct-q4_K_M",
    "inference_interval_minutes": 30
  },
  "fast_model": {
    "name": "gemma2:27b",
    "inference_interval_minutes": 1
  },
  "ensemble_weight": 0.4,
  "override_threshold": 0.85
}
```

## 📈 Performance Monitoring

### Real-Time Metrics

The bot logs comprehensive metrics every 5 minutes:

```
System Metrics:
  Predictions: 1234
  Trades: 45
  Model retrains: 12
  Account balance: $10,500.23
  Win rate: 62.3%
```

### Health Checks

Automated monitoring of:
- Data pipeline freshness (< 5 min)
- Model loaded status
- LLM API availability
- Error rate tracking

### Discord Alerts

Receive instant notifications for:
- Position opened/closed
- Stop loss triggered
- System errors
- Daily performance summary

## 🔒 Security Best Practices

1. **Start with Testnet**: Always validate strategies on testnet first
2. **Private Key Security**:
   - Never commit private keys to git
   - Use `chmod 600` on key files
   - Consider hardware wallet integration
3. **Position Limits**: Set conservative limits initially
4. **Monitoring**: Enable Discord alerts for all critical events
5. **Regular Audits**: Review trade logs and performance weekly

## 🐛 Troubleshooting

### Bot won't start

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check Python dependencies
pip install -r requirements.txt --upgrade

# View startup errors
cat logs/stdout.log
```

### LLM inference slow

```bash
# Check Ollama performance
ollama run llama3.1:70b-instruct-q4_K_M "test"

# Monitor CPU usage
htop

# Reduce model size if needed (use Q3 quantization)
ollama pull llama3.1:70b-instruct-q3_K_M
```

### Model retraining timeout

Adjust in `config.json`:

```json
"model": {
  "max_retrain_time_seconds": 300,  // Increase if needed
  "max_train_samples": 10000000     // Reduce dataset size
}
```

## 📚 Advanced Usage

### Custom Features

Edit `src/features/feature_engine.py` to add custom features:

```python
def _add_custom_features(self, df: pl.DataFrame) -> pl.DataFrame:
    """Add your custom features here."""
    df = df.with_columns([
        # Example: Custom momentum indicator
        (pl.col("close") / pl.col("close").shift(20) - 1).alias("momentum_20")
    ])
    return df
```

### Multiple Symbols

Trade multiple perpetuals simultaneously:

```json
"hyperliquid": {
  "symbols": ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
}
```

### Hyperparameter Optimization

Use the backtest module to optimize:

```python
# Create grid search script
for kelly_fraction in [0.1, 0.2, 0.25, 0.3]:
    for stop_loss in [1.5, 2.0, 2.5]:
        # Run backtest with different params
        results = run_backtest(...)
```

## 🤝 Contributing

This is a private trading system. For questions or issues:

1. Check logs: `logs/trading_bot.log`
2. Review configuration: `config/config.json`
3. Test on testnet first
4. Document all modifications

## ⚖️ License & Disclaimer

**DISCLAIMER**: This software is for educational and research purposes only. Cryptocurrency trading carries substantial risk of loss. The authors are not responsible for any financial losses incurred through use of this software.

**USE AT YOUR OWN RISK.**

- Past performance does not guarantee future results
- Always test strategies on testnet before mainnet
- Never trade with money you cannot afford to lose
- Consult financial advisors before trading

## 🙏 Acknowledgments

- **Hyperliquid**: For the high-performance perpetuals DEX
- **LightGBM**: Microsoft's gradient boosting framework
- **Ollama**: For making local LLM inference accessible
- **Polars**: Blazing fast DataFrame library

## 📞 Support

For technical issues:
- Check logs: `tail -f logs/trading_bot.log`
- Review health status: `grep "Health check" logs/trading_bot.log`
- Test components individually (see module docstrings)

---

**Built for AMD Threadripper 3960X | CPU-Only | Production Ready**

Last Updated: 2025-12-03
