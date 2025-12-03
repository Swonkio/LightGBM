#!/bin/bash

# Hyperliquid Trading Bot Launcher
# Production-ready startup script with monitoring and auto-restart

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Hyperliquid Perpetuals Trading Bot${NC}"
echo -e "${GREEN}========================================${NC}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
CONFIG_FILE="./config/config.json"
PYTHON_BIN=${PYTHON_BIN:-python3}
LOG_DIR="./logs"
PID_FILE="./bot.pid"

# Pre-flight checks
echo -e "\n${YELLOW}Running pre-flight checks...${NC}"

# 1. Check Python version
echo -n "Checking Python version... "
PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo -e "${RED}FAIL${NC}"
    echo -e "${RED}Python 3.11+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}OK ($PYTHON_VERSION)${NC}"

# 2. Check config file exists
echo -n "Checking configuration file... "
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}FAIL${NC}"
    echo -e "${RED}Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}OK${NC}"

# 3. Check Ollama is running
echo -n "Checking Ollama service... "
if command -v systemctl &> /dev/null; then
    if systemctl is-active --quiet ollama; then
        echo -e "${GREEN}OK (running)${NC}"
    else
        echo -e "${YELLOW}WARNING${NC}"
        echo -e "${YELLOW}Ollama service not running. Starting...${NC}"
        sudo systemctl start ollama || true
    fi
else
    # Check if Ollama is accessible
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}OK (accessible)${NC}"
    else
        echo -e "${YELLOW}WARNING${NC}"
        echo -e "${YELLOW}Ollama API not accessible. LLM features may not work.${NC}"
    fi
fi

# 4. Check required Python packages
echo -n "Checking Python dependencies... "
if $PYTHON_BIN -c "import polars, lightgbm, websockets, requests" 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# 5. Check available RAM
echo -n "Checking available RAM... "
AVAILABLE_RAM=$(free -g | awk '/^Mem:/{print $7}')
if [ "$AVAILABLE_RAM" -lt 50 ]; then
    echo -e "${YELLOW}WARNING${NC}"
    echo -e "${YELLOW}Available RAM: ${AVAILABLE_RAM}GB (< 50GB). May impact performance.${NC}"
else
    echo -e "${GREEN}OK (${AVAILABLE_RAM}GB available)${NC}"
fi

# 6. Create necessary directories
echo -n "Creating directories... "
mkdir -p "$LOG_DIR"
mkdir -p ./data/{live,historical,models,backtests}
echo -e "${GREEN}OK${NC}"

# 7. Check for existing instance
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Bot already running (PID: $OLD_PID)${NC}"
        read -p "Stop existing instance and restart? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping existing instance..."
            kill "$OLD_PID"
            sleep 2
        else
            echo "Exiting..."
            exit 1
        fi
    fi
fi

echo -e "\n${GREEN}✓ All pre-flight checks passed${NC}\n"

# Parse command line arguments
MODE=${1:-live}  # live, backtest, or dry-run

case "$MODE" in
    live)
        echo -e "${GREEN}Starting LIVE trading bot...${NC}"
        echo -e "${YELLOW}WARNING: This will execute REAL trades!${NC}"
        read -p "Continue? (yes/no) " -r
        if [[ ! $REPLY =~ ^yes$ ]]; then
            echo "Exiting..."
            exit 0
        fi
        ;;

    backtest)
        echo -e "${GREEN}Starting backtest mode...${NC}"
        $PYTHON_BIN -m src.scripts.run_backtest
        exit 0
        ;;

    dry-run)
        echo -e "${GREEN}Starting dry-run mode (no execution)...${NC}"
        # Temporarily disable execution
        jq '.execution.enabled = false' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
        ;;

    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: $0 {live|backtest|dry-run}"
        exit 1
        ;;
esac

# Launch the bot
echo -e "\n${GREEN}Launching trading bot...${NC}"
echo "Logs: $LOG_DIR/trading_bot.log"
echo "Process ID will be saved to: $PID_FILE"

# Start the bot
nohup $PYTHON_BIN -u src/main.py > "$LOG_DIR/stdout.log" 2>&1 &
BOT_PID=$!

# Save PID
echo $BOT_PID > "$PID_FILE"

# Wait a moment and check if process is running
sleep 2
if ps -p "$BOT_PID" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Bot started successfully (PID: $BOT_PID)${NC}"
    echo ""
    echo "Monitor logs with:"
    echo "  tail -f $LOG_DIR/trading_bot.log"
    echo ""
    echo "Stop the bot with:"
    echo "  ./stop.sh"
    echo ""
    echo "View metrics:"
    echo "  cat $LOG_DIR/trading_bot.log | grep 'System Metrics' -A 10"
else
    echo -e "${RED}✗ Bot failed to start${NC}"
    echo "Check logs: $LOG_DIR/stdout.log"
    rm -f "$PID_FILE"
    exit 1
fi

# Optional: tail logs
read -p "Tail logs now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    tail -f "$LOG_DIR/trading_bot.log"
fi
