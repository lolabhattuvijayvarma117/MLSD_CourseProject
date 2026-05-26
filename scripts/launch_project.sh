#!/bin/bash

# launch_project.sh: Boots and coordinates FastAPI serving and MLflow tracking on macOS
# Compatible with both Apple Silicon and Intel Terminal environments

PROJECT_DIR="/Users/vijayvarmalolabhattu/L.VijayVarma_MLSD_CourseProject"
cd "$PROJECT_DIR" || exit 1

# Colors for output decoration
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                    LAUNCHING FRAUD DETECTION MLOPS SERVICES                    ${NC}"
echo -e "${BLUE}================================================================================${NC}"

# Check virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment 'venv' not found. Run setup/pip install first.${NC}"
    exit 1
fi

source venv/bin/activate

# Check if model binary is available
if [ ! -f "models/champion_model.joblib" ]; then
    echo -e "${YELLOW}⚠️ Warning: Model binary not found at models/champion_model.joblib${NC}"
    echo -e "${YELLOW}Training model first to ensure prediction service starts correctly...${NC}"
    python3 model_training/train_model.py
fi

# Kill any existing processes on port 8000 or 5001
echo -e "${BLUE}Cleaning up ports 8000 and 5001...${NC}"
pkill -f "uvicorn fraud_detection_service.main:app" 2>/dev/null
pkill -f "mlflow ui" 2>/dev/null
sleep 1

# Check if AppleScript can be used (if running interactively in macOS window system)
if [[ "$OSTYPE" == "darwin"* ]] && [ -n "$DISPLAY" -o -d "/Applications" ]; then
    echo -e "${GREEN}✓ Launching services in native macOS Terminal tabs...${NC}"
    
    # Launch MLflow UI in Terminal Tab 1
    osascript -e "tell application \"Terminal\"
        activate
        do script \"cd $PROJECT_DIR && source venv/bin/activate && mlflow ui --port 5001\"
    end tell"
    
    # Launch FastAPI Server in Terminal Tab 2
    osascript -e "tell application \"Terminal\"
        activate
        do script \"cd $PROJECT_DIR && source venv/bin/activate && uvicorn fraud_detection_service.main:app --port 8000 --reload\"
    end tell"
    
else
    # Fallback to background processes if non-interactive / SSH session
    echo -e "${YELLOW}⚠️ Non-interactive session detected. Starting processes in background...${NC}"
    
    # Start MLflow
    echo -e "${BLUE}Starting MLflow server...${NC}"
    mlflow ui --port 5001 > mlflow.log 2>&1 &
    MLFLOW_PID=$!
    
    # Start FastAPI
    echo -e "${BLUE}Starting FastAPI Uvicorn engine...${NC}"
    uvicorn fraud_detection_service.main:app --port 8000 --reload > fastapi.log 2>&1 &
    FASTAPI_PID=$!
    
    echo -e "${GREEN}✓ MLflow running in background (PID: $MLFLOW_PID, logging to mlflow.log)${NC}"
    echo -e "${GREEN}✓ FastAPI running in background (PID: $FASTAPI_PID, logging to fastapi.log)${NC}"
fi

# Verification
sleep 3
echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}🚀 MLOps Environment successfully initialized!${NC}"
echo -e "  - ${GREEN}FastAPI Swagger documentation: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  - ${GREEN}FastAPI Health check:          ${BLUE}http://localhost:8000/health${NC}"
echo -e "  - ${GREEN}MLflow tracking dashboard:     ${BLUE}http://localhost:5001${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo -e "${YELLOW}Use 'pkill -f mlflow' and 'pkill -f uvicorn' to shut down the processes.${NC}"
echo -e "${BLUE}================================================================================${NC}"
