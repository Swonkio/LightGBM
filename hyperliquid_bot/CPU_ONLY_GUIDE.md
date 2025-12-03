# CPU-Only Operation Guide

## Your Setup: GTX 1060 for Display Only

Your system has a GTX 1060 GPU that's only for display. Here's how to ensure **all ML inference stays on your Threadripper 3960X CPU**.

---

## ✅ Quick Verification

Run this script to check CPU-only configuration:

```bash
./scripts/verify_cpu_only.sh
```

Expected output:
```
✓ System is configured for CPU-only operation
Your GTX 1060 will only be used for display.
All ML inference will run on the Threadripper CPU.
```

---

## 🔒 Force CPU-Only Mode

### Step 1: Configure Ollama for CPU-Only

**Option A: Systemd Service (Recommended)**

```bash
# Edit Ollama service
sudo systemctl edit ollama
```

Add these lines:
```ini
[Service]
Environment="CUDA_VISIBLE_DEVICES=-1"
Environment="OLLAMA_NUM_GPU=0"
```

Save and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**Option B: Manual Launch**

If running Ollama manually:
```bash
CUDA_VISIBLE_DEVICES=-1 OLLAMA_NUM_GPU=0 ollama serve
```

### Step 2: Verify Ollama is CPU-Only

```bash
# Terminal 1: Monitor GPU memory
watch -n 0.5 nvidia-smi

# Terminal 2: Test inference
ollama run deepseek-r1:32b "test inference"
```

**GPU memory should stay at 0 MB** during inference.

If you see GPU memory increase:
- Ollama is using GPU ❌
- Go back to Step 1 and reconfigure

### Step 3: Set Environment Variable for Bot

Add to your `~/.bashrc` or `~/.profile`:

```bash
export CUDA_VISIBLE_DEVICES=-1
```

Then reload:
```bash
source ~/.bashrc
```

Or set it each time you start the bot:
```bash
CUDA_VISIBLE_DEVICES=-1 ./start.sh live
```

---

## 🧪 Test CPU-Only Operation

### Test 1: Ollama CPU Usage

```bash
# Start inference
ollama run deepseek-r1:32b "Explain quantum computing in one sentence"

# In another terminal, check CPU usage
htop

# You should see:
# - High CPU usage on Threadripper cores
# - ~20-30% utilization across all 24 cores
# - Ollama process using 100-300% CPU (multiple cores)
```

### Test 2: GPU Stays Idle

```bash
# Monitor GPU continuously
watch -n 1 'nvidia-smi | grep -A 2 "Processes:"'

# Should show:
# | No running processes found |

# GTX 1060 should only show your display server (Xorg/Wayland)
```

### Test 3: LightGBM Training

```bash
# Run a test training
cd hyperliquid_bot
python3 -c "
from src.models.lgbm_trainer import IncrementalLGBMTrainer
import numpy as np
import polars as pl

# Generate test data
n = 100000
X = np.random.randn(n, 45)
y = np.random.randint(0, 3, n)

df = pl.DataFrame({
    **{f'f{i}': X[:, i] for i in range(45)},
    'target': y,
    'timestamp': pl.datetime_range(pl.datetime(2024,1,1), pl.datetime(2024,1,1,0,n), interval='1m', eager=True)
})

trainer = IncrementalLGBMTrainer(num_threads=24)
train_data, val_data, _ = trainer.prepare_training_data(df)
model = trainer.train_from_scratch(train_data, val_data, num_boost_round=100)
print('✓ Training completed on CPU')
"

# Watch GPU memory during this - should stay at 0
```

---

## 📊 Expected Performance (CPU-Only)

On your Threadripper 3960X:

| Operation | Expected Performance |
|-----------|---------------------|
| DeepSeek R1 32B inference | ~15-25 tokens/sec |
| GPT-OSS 20B inference | ~30-50 tokens/sec |
| LightGBM training (10M rows) | ~2 minutes |
| LightGBM prediction | ~5ms per row |
| All 24 cores utilized | Yes ✓ |
| GPU memory used | 0 MB ✓ |

---

## 🚨 Troubleshooting

### Problem: Ollama is using GPU

**Symptoms:**
- `nvidia-smi` shows memory usage during Ollama inference
- Inference is faster than expected (GPU is faster)

**Fix:**
```bash
# 1. Stop Ollama
sudo systemctl stop ollama

# 2. Force CPU-only
sudo systemctl edit ollama
# Add:
#   [Service]
#   Environment="CUDA_VISIBLE_DEVICES=-1"
#   Environment="OLLAMA_NUM_GPU=0"

# 3. Restart
sudo systemctl daemon-reload
sudo systemctl start ollama

# 4. Verify
./scripts/verify_cpu_only.sh
```

### Problem: Bot starts but GPU memory increases

**Symptoms:**
- Bot starts successfully
- `nvidia-smi` shows increasing GPU memory

**Fix:**
```bash
# Kill the bot
./stop.sh

# Set environment variable
export CUDA_VISIBLE_DEVICES=-1

# Verify it's set
echo $CUDA_VISIBLE_DEVICES  # Should output: -1

# Restart bot
./start.sh live
```

### Problem: CUDA errors in logs

**Symptoms:**
- Logs show CUDA-related errors
- "CUDA out of memory" messages

**Fix:**
This means something is trying to use the GPU. Force disable:

```bash
# Permanent fix - add to startup
echo 'export CUDA_VISIBLE_DEVICES=-1' >> ~/.bashrc
source ~/.bashrc

# Verify
python3 -c "import os; print('CUDA disabled:', os.environ.get('CUDA_VISIBLE_DEVICES') == '-1')"
```

---

## ✅ Final Checklist

Before running the bot in production:

- [ ] Run `./scripts/verify_cpu_only.sh` - all checks pass
- [ ] Ollama service has `CUDA_VISIBLE_DEVICES=-1` set
- [ ] Environment variable exported: `echo $CUDA_VISIBLE_DEVICES` returns `-1`
- [ ] GPU memory stays at 0 MB during Ollama test
- [ ] `htop` shows high CPU usage during inference
- [ ] Bot starts without CUDA errors in logs
- [ ] Your display (GTX 1060) continues working normally

---

## 🎯 Why CPU-Only?

**Advantages:**
- ✅ More consistent inference times
- ✅ No GPU memory management issues
- ✅ Can run bigger batch sizes in CPU RAM
- ✅ Better for production (GPUs can crash)
- ✅ Your GTX 1060 (6GB) couldn't fit 32B models anyway

**Your Threadripper 3960X:**
- 24 cores / 48 threads @ 3.8 GHz
- 256 GB RAM (massive advantage over GPU memory)
- Perfect for quantized LLMs (Q4, Q5)
- Excellent parallel performance for LightGBM

---

## 📈 Monitoring CPU Usage

While bot is running:

```bash
# Real-time CPU usage
htop

# Per-core usage
mpstat -P ALL 1

# Process-specific
top -p $(cat bot.pid)

# Temperature (if available)
sensors

# Memory usage
free -h
```

Expect:
- **Idle**: 5-10% CPU (data collection)
- **During DeepSeek inference**: 60-80% CPU (15-20 cores active)
- **During LightGBM training**: 95-100% CPU (all 24 cores maxed)
- **GPU**: Always idle except for display

---

**Your GTX 1060 will only handle graphics output. All ML work stays on the CPU.** ✅
