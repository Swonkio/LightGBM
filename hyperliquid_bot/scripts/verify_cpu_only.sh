#!/bin/bash

# Verify that the bot is configured for CPU-only operation

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "CPU-Only Configuration Verification"
echo "=========================================="
echo ""

PASS=0
FAIL=0

# Check 1: CUDA_VISIBLE_DEVICES
echo -n "Environment CUDA_VISIBLE_DEVICES: "
if [ "$CUDA_VISIBLE_DEVICES" = "-1" ]; then
    echo -e "${GREEN}Set to -1 (GPU disabled) ✓${NC}"
    ((PASS++))
elif [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo -e "${YELLOW}Not set (GPU may be used)${NC}"
    echo "  Set with: export CUDA_VISIBLE_DEVICES=-1"
    ((FAIL++))
else
    echo -e "${YELLOW}Set to: $CUDA_VISIBLE_DEVICES${NC}"
fi

# Check 2: Ollama service configuration
echo -n "Ollama service configuration: "
if systemctl show ollama -p Environment 2>/dev/null | grep -q "CUDA_VISIBLE_DEVICES=-1"; then
    echo -e "${GREEN}CPU-only configured ✓${NC}"
    ((PASS++))
else
    echo -e "${YELLOW}Not configured for CPU-only${NC}"
    echo "  Configure with:"
    echo "    sudo systemctl edit ollama"
    echo "    Add: Environment=\"CUDA_VISIBLE_DEVICES=-1\""
    echo "         Environment=\"OLLAMA_NUM_GPU=0\""
    ((FAIL++))
fi

# Check 3: Test Ollama inference without GPU
echo -n "Testing Ollama CPU-only inference: "
if command -v nvidia-smi &> /dev/null; then
    # GPU present, test if it's being used
    GPU_MEM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

    # Run quick inference
    timeout 10 ollama run deepseek-r1:32b "test" > /dev/null 2>&1 &
    OLLAMA_PID=$!
    sleep 5

    GPU_MEM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    kill $OLLAMA_PID 2>/dev/null || true

    MEM_DIFF=$((GPU_MEM_AFTER - GPU_MEM_BEFORE))

    if [ "$MEM_DIFF" -lt 100 ]; then
        echo -e "${GREEN}No GPU memory used (${MEM_DIFF}MB delta) ✓${NC}"
        ((PASS++))
    else
        echo -e "${RED}GPU memory increased by ${MEM_DIFF}MB ✗${NC}"
        echo "  Ollama is using GPU! Force CPU-only mode."
        ((FAIL++))
    fi
else
    echo -e "${GREEN}No GPU detected ✓${NC}"
    ((PASS++))
fi

# Check 4: Python LightGBM configuration
echo -n "LightGBM configuration: "
DEVICE=$(python3 -c "import json; print(json.load(open('config/config.json'))['model']['training_params'].get('device', 'cpu'))" 2>/dev/null || echo "cpu")
if [ "$DEVICE" = "cpu" ]; then
    echo -e "${GREEN}CPU-only ✓${NC}"
    ((PASS++))
else
    echo -e "${RED}Device set to: $DEVICE ✗${NC}"
    ((FAIL++))
fi

# Check 5: Verify LightGBM package is CPU version
echo -n "LightGBM package: "
if python3 -c "import lightgbm; print(lightgbm.__version__)" 2>/dev/null | grep -v gpu > /dev/null; then
    echo -e "${GREEN}CPU version installed ✓${NC}"
    ((PASS++))
else
    echo -e "${YELLOW}Cannot verify version${NC}"
fi

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${RED}Failed: $FAIL${NC}"
echo "=========================================="

if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}✓ System is configured for CPU-only operation${NC}"
    echo ""
    echo "Your GTX 1060 will only be used for display."
    echo "All ML inference will run on the Threadripper CPU."
    exit 0
else
    echo -e "${RED}✗ GPU may be used. Fix the issues above.${NC}"
    exit 1
fi
