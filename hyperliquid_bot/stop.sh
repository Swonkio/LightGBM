#!/bin/bash

# Stop the Hyperliquid trading bot gracefully

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PID_FILE="./bot.pid"

echo -e "${YELLOW}Stopping Hyperliquid Trading Bot...${NC}"

if [ ! -f "$PID_FILE" ]; then
    echo -e "${RED}No PID file found. Bot may not be running.${NC}"
    exit 1
fi

BOT_PID=$(cat "$PID_FILE")

if ! ps -p "$BOT_PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}Bot process (PID: $BOT_PID) not running.${NC}"
    rm -f "$PID_FILE"
    exit 0
fi

echo "Sending SIGTERM to process $BOT_PID..."
kill -TERM "$BOT_PID"

# Wait for graceful shutdown (max 30 seconds)
for i in {1..30}; do
    if ! ps -p "$BOT_PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Bot stopped gracefully${NC}"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo -e "${YELLOW}Bot did not stop gracefully. Forcing shutdown...${NC}"
kill -9 "$BOT_PID"
rm -f "$PID_FILE"
echo -e "${GREEN}✓ Bot stopped (forced)${NC}"
