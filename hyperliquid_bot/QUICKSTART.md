# Quick Start Guide

Get your Hyperliquid trading bot running in 5 minutes.

## Prerequisites Checklist

- [ ] Python 3.11+ installed
- [ ] 64GB+ RAM available
- [ ] 100GB+ free disk space
- [ ] Hyperliquid testnet wallet created
- [ ] Ollama installed and running

## Step-by-Step Setup

### 1. Install Ollama and Models (15-20 minutes)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download models (this will take a while - ~60GB total)
ollama pull llama3.1:70b-instruct-q4_K_M  # Main model (~40GB)
ollama pull gemma2:27b                     # Fast model (~16GB)

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

Expected output:
```json
{
  "models": [
    {"name": "llama3.1:70b-instruct-q4_K_M", ...},
    {"name": "gemma2:27b", ...}
  ]
}
```

### 2. Install Python Dependencies (2-3 minutes)

```bash
cd hyperliquid_bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3. Configure for Testnet (1 minute)

**Get your Hyperliquid testnet wallet:**
1. Visit https://app.hyperliquid-testnet.xyz
2. Connect wallet or create new one
3. Get testnet funds from faucet
4. Copy your wallet address

**Update config/config.json:**

```json
{
  "environment": {
    "mode": "testnet"  // IMPORTANT: Start with testnet!
  },

  "hyperliquid": {
    "wallet_address": "0xYourWalletAddressHere",
    "private_key_path": "./config/private_key.txt"
  },

  "execution": {
    "enabled": false  // Disable for first run
  },

  "llm": {
    "enabled": true
  }
}
```

**Add your private key:**

```bash
# Create private key file
echo "0xYourPrivateKeyHere" > config/private_key.txt

# Secure it
chmod 600 config/private_key.txt
```

**WARNING:** Never commit this file to version control!

### 4. Test Run - Dry Mode (30 seconds)

```bash
# Start bot in dry-run mode (no trading)
./start.sh dry-run

# In another terminal, watch the logs
tail -f logs/trading_bot.log
```

You should see:
```
✓ Configuration loaded successfully
✓ Connected to Ollama
✓ WebSocket connected successfully
✓ Model loaded
Trading loop started
```

Press `Ctrl+C` or run `./stop.sh` to stop.

### 5. Run a Backtest (5-10 minutes)

Test the strategy on historical data:

```bash
# Run backtest for recent period
python -m src.scripts.run_backtest \
  --start 2024-10-01 \
  --end 2024-11-01 \
  --symbol BTC-PERP

# Check results
cat data/backtests/trades_*.csv | wc -l
```

Review metrics:
- Win rate > 55%: Good
- Sharpe ratio > 1.0: Good
- Max drawdown < 20%: Good

### 6. Paper Trading (Testnet)

Once backtest looks good:

**Edit config/config.json:**

```json
{
  "execution": {
    "enabled": true  // Enable execution on testnet
  },

  "risk": {
    "max_position_size_usd": 100  // Start small!
  }
}
```

**Start the bot:**

```bash
./start.sh live
```

**Monitor closely:**

```bash
# Watch logs
tail -f logs/trading_bot.log

# Check positions (in Python)
python3 -c "
from src.execution.order_manager import HyperliquidExecutor
executor = HyperliquidExecutor(
    api_url='https://api.hyperliquid-testnet.xyz',
    wallet_address='YOUR_ADDRESS',
    private_key='',
    is_testnet=True
)
executor.sync_positions()
print(executor.positions)
"
```

### 7. Production (Mainnet) - When Ready

**ONLY after weeks of successful testnet trading:**

```json
{
  "environment": {
    "mode": "mainnet"  // Switch to mainnet
  },

  "risk": {
    "max_position_size_usd": 1000  // Conservative start
  }
}
```

```bash
./start.sh live
```

## Common Issues & Solutions

### Issue: "Ollama API not responding"

```bash
# Check Ollama status
systemctl status ollama

# Restart if needed
sudo systemctl restart ollama

# Test inference
ollama run gemma2:27b "test"
```

### Issue: "WebSocket connection failed"

- Check internet connection
- Verify API URL in config
- Test with: `curl https://api.hyperliquid-testnet.xyz/info`

### Issue: "Model training timeout"

Reduce dataset size in config:

```json
"model": {
  "max_train_samples": 5000000  // Reduce from 15M
}
```

### Issue: "Out of memory"

Reduce buffer sizes:

```json
"data_pipeline": {
  "buffer_size": 50000  // Reduce from 100000
}
```

## Monitoring Checklist

Daily checks:
- [ ] Bot process running: `ps aux | grep main.py`
- [ ] Recent predictions: `grep "Signal:" logs/trading_bot.log | tail -20`
- [ ] Account balance increasing
- [ ] Error count low: `grep "ERROR" logs/trading_bot.log | wc -l`
- [ ] Disk space available: `df -h`

Weekly checks:
- [ ] Review win rate and metrics
- [ ] Check model retrain frequency
- [ ] Verify Discord alerts working
- [ ] Backup trade data: `cp -r data/backtests backups/`

## Performance Benchmarks

On AMD Threadripper 3960X (24c/48t):

| Operation | Target | Typical |
|-----------|--------|---------|
| Feature computation (1000 rows) | < 100ms | ~50ms |
| LightGBM prediction | < 10ms | ~5ms |
| LightGBM full retrain (10M rows) | < 3min | ~2min |
| Llama 3.1 70B inference | < 30s | ~20s |
| Gemma-2 27B inference | < 5s | ~3s |

If your system is slower:
1. Check CPU frequency: `lscpu | grep MHz`
2. Reduce thread count in config
3. Use smaller LLM quantization (Q3 instead of Q4)

## Next Steps

1. ✅ Run for 1 week on testnet
2. ✅ Analyze performance metrics
3. ✅ Tune risk parameters
4. ✅ Enable Discord alerts
5. ✅ Test stop-loss behavior
6. ✅ Document your modifications
7. 🚀 Consider mainnet (with caution)

## Support

- Documentation: `README.md`
- Logs: `logs/trading_bot.log`
- Health check: `grep "System Metrics" logs/trading_bot.log -A 6`

**Remember: This is real trading with real money. Start small, test thoroughly, and never risk more than you can afford to lose.**
