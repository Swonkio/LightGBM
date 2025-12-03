#!/bin/bash

# System compatibility checker for Hyperliquid trading bot

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Hyperliquid Bot - System Compatibility"
echo "=========================================="
echo ""

PASS=0
FAIL=0
WARN=0

# Check 1: CPU cores
echo -n "CPU cores: "
CORES=$(nproc)
if [ "$CORES" -ge 24 ]; then
    echo -e "${GREEN}$CORES (excellent)${NC}"
    ((PASS++))
elif [ "$CORES" -ge 12 ]; then
    echo -e "${YELLOW}$CORES (acceptable, but 24+ recommended)${NC}"
    ((WARN++))
else
    echo -e "${RED}$CORES (insufficient, 12+ required)${NC}"
    ((FAIL++))
fi

# Check 2: RAM
echo -n "Total RAM: "
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM" -ge 200 ]; then
    echo -e "${GREEN}${TOTAL_RAM}GB (excellent)${NC}"
    ((PASS++))
elif [ "$TOTAL_RAM" -ge 64 ]; then
    echo -e "${YELLOW}${TOTAL_RAM}GB (acceptable for smaller datasets)${NC}"
    ((WARN++))
else
    echo -e "${RED}${TOTAL_RAM}GB (insufficient, 64GB+ required)${NC}"
    ((FAIL++))
fi

# Check 3: Available RAM
echo -n "Available RAM: "
AVAIL_RAM=$(free -g | awk '/^Mem:/{print $7}')
if [ "$AVAIL_RAM" -ge 50 ]; then
    echo -e "${GREEN}${AVAIL_RAM}GB${NC}"
    ((PASS++))
else
    echo -e "${YELLOW}${AVAIL_RAM}GB (may need to free up memory)${NC}"
    ((WARN++))
fi

# Check 4: Disk space
echo -n "Free disk space: "
FREE_DISK=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$FREE_DISK" -ge 500 ]; then
    echo -e "${GREEN}${FREE_DISK}GB${NC}"
    ((PASS++))
elif [ "$FREE_DISK" -ge 100 ]; then
    echo -e "${YELLOW}${FREE_DISK}GB (may need more for long-term data)${NC}"
    ((WARN++))
else
    echo -e "${RED}${FREE_DISK}GB (insufficient, 100GB+ required)${NC}"
    ((FAIL++))
fi

# Check 5: Python version
echo -n "Python version: "
if command -v python3 &> /dev/null; then
    PYTHON_VER=$(python3 --version | awk '{print $2}')
    MAJOR=$(echo $PYTHON_VER | cut -d. -f1)
    MINOR=$(echo $PYTHON_VER | cut -d. -f2)

    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 11 ]; then
        echo -e "${GREEN}$PYTHON_VER${NC}"
        ((PASS++))
    else
        echo -e "${RED}$PYTHON_VER (3.11+ required)${NC}"
        ((FAIL++))
    fi
else
    echo -e "${RED}Not found${NC}"
    ((FAIL++))
fi

# Check 6: Ollama installation
echo -n "Ollama: "
if command -v ollama &> /dev/null; then
    OLLAMA_VER=$(ollama --version 2>&1 | head -n1)
    echo -e "${GREEN}Installed ($OLLAMA_VER)${NC}"
    ((PASS++))
else
    echo -e "${RED}Not installed${NC}"
    echo "  Install: curl -fsSL https://ollama.ai/install.sh | sh"
    ((FAIL++))
fi

# Check 7: Ollama API
echo -n "Ollama API: "
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}Running${NC}"
    ((PASS++))
else
    echo -e "${RED}Not accessible${NC}"
    echo "  Start: sudo systemctl start ollama"
    ((FAIL++))
fi

# Check 8: Required models
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4)

    echo -n "Llama 3.1 70B: "
    if echo "$MODELS" | grep -q "llama3.1:70b"; then
        echo -e "${GREEN}Installed${NC}"
        ((PASS++))
    else
        echo -e "${RED}Not found${NC}"
        echo "  Install: ollama pull llama3.1:70b-instruct-q4_K_M"
        ((FAIL++))
    fi

    echo -n "Gemma 2 27B: "
    if echo "$MODELS" | grep -q "gemma2:27b"; then
        echo -e "${GREEN}Installed${NC}"
        ((PASS++))
    else
        echo -e "${RED}Not found${NC}"
        echo "  Install: ollama pull gemma2:27b"
        ((FAIL++))
    fi
fi

# Check 9: Network latency
echo -n "Network latency (Hyperliquid): "
LATENCY=$(ping -c 3 api.hyperliquid.xyz 2>/dev/null | tail -1 | awk -F '/' '{print $5}')
if [ -z "$LATENCY" ]; then
    echo -e "${YELLOW}Cannot measure (offline?)${NC}"
    ((WARN++))
else
    LATENCY_INT=${LATENCY%.*}
    if [ "$LATENCY_INT" -lt 50 ]; then
        echo -e "${GREEN}${LATENCY}ms (excellent)${NC}"
        ((PASS++))
    elif [ "$LATENCY_INT" -lt 150 ]; then
        echo -e "${YELLOW}${LATENCY}ms (acceptable)${NC}"
        ((WARN++))
    else
        echo -e "${RED}${LATENCY}ms (high, may affect trading)${NC}"
        ((FAIL++))
    fi
fi

# Summary
echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${YELLOW}Warnings: $WARN${NC}"
echo -e "${RED}Failed: $FAIL${NC}"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}✓ System is ready for trading bot${NC}"
    exit 0
elif [ "$FAIL" -le 2 ]; then
    echo -e "${YELLOW}⚠ System may work but with limitations${NC}"
    echo "  Address failures above before production use"
    exit 1
else
    echo -e "${RED}✗ System not ready${NC}"
    echo "  Fix critical issues before proceeding"
    exit 1
fi
