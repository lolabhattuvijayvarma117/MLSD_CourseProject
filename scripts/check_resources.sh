#!/bin/bash

# check_resources.sh: Report macOS and container execution metrics
echo "================================================================================"
echo "                   macOS MLOPS SYSTEM RESOURCE MONITOR"
echo "================================================================================"
echo "Date: $(date)"
echo "Host OS Version: $(sw_vers -productName) $(sw_vers -productVersion) (Build $(sw_vers -buildVersion))"
echo "Architecture: $(uname -m)"
echo "================================================================================"

echo ""
echo "[1] CPU & Core Status"
echo "--------------------------------------------------------------------------------"
echo "Hardware Cores: $(sysctl -n hw.physicalcpu)"
echo "Logical Threads: $(sysctl -n hw.logicalcpu)"
echo "Top CPU consuming processes (Python/Docker):"
ps aux | grep -E "python|docker|uvicorn|mlflow" | grep -v grep | awk '{printf "  PID: %-6s CPU: %-5s Command: %-40s\n", $2, $3, $11}' | head -5

echo ""
echo "[2] Memory Statistics"
echo "--------------------------------------------------------------------------------"
# Process vm_stat page details to calculate free memory in GB
PAGESIZE=$(vm_stat | grep "page size of" | awk '{print $8}')
PAGES_FREE=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
PAGES_ACTIVE=$(vm_stat | grep "Pages active" | awk '{print $3}' | sed 's/\.//')
PAGES_INACTIVE=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')

FREE_GB=$(echo "scale=2; ($PAGES_FREE * $PAGESIZE) / 1024 / 1024 / 1024" | bc)
ACTIVE_GB=$(echo "scale=2; ($PAGES_ACTIVE * $PAGESIZE) / 1024 / 1024 / 1024" | bc)

echo "  Free Memory:   ${FREE_GB} GB"
echo "  Active Memory: ${ACTIVE_GB} GB"

echo ""
echo "[3] Storage & Disk Usage"
echo "--------------------------------------------------------------------------------"
df -h | grep -E "Filesystem|/dev/disk|/Volumes"

echo ""
echo "[4] Docker Containers Status"
echo "--------------------------------------------------------------------------------"
if command -v docker &> /dev/null && docker ps &> /dev/null; then
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
else
    echo "  ⚠️ Docker daemon is not active or Docker CLI is not installed"
fi

echo ""
echo "[5] MLOps Ports Allocations"
echo "--------------------------------------------------------------------------------"
echo "Checking Local Ports 8000 (FastAPI) and 5001 (MLflow):"
PORT_8000=$(lsof -i :8000 -t)
PORT_5001=$(lsof -i :5001 -t)

if [ -n "$PORT_8000" ]; then
    echo "  - Port 8000: occupied by PID $PORT_8000"
    ps -p $PORT_8000 -o command= | sed 's/^/    /'
else
    echo "  - Port 8000: FREE"
fi

if [ -n "$PORT_5001" ]; then
    echo "  - Port 5001: occupied by PID $PORT_5001"
    ps -p $PORT_5001 -o command= | sed 's/^/    /'
else
    echo "  - Port 5001: FREE"
fi
echo "================================================================================"
