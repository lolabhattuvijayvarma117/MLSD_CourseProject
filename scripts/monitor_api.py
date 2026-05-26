"""
Continuous API health monitoring and performance tracking
"""
import time
import requests
from datetime import datetime

API_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{API_URL}/health"

def check_api_health():
    """Check API health status"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=3)
        if response.status_code == 200:
            data = response.json()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ API Status: HEALTHY")
            print(f"  - Model Loaded: {data.get('model_loaded')}")
            print(f"  - Pipeline Features: {data.get('features_count')}")
            print(f"  - Model Performance (AUC-ROC): {data.get('auc_roc', 0.0):.4f}")
            print(f"  - serving Platform: {data.get('system')}")
            return True
        elif response.status_code == 503:
            data = response.json()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ API Status: DEGRADED")
            print(f"  - Warning: {data.get('detail')}")
            return False
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ API Status: UNEXPECTED (Code {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ API Status: UNREACHABLE (Is Uvicorn running on port 8000?)")
        return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ Monitor Exception: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("FastAPI serving Engine Continuous Health Monitor Initialized")
    print("Polling target: http://localhost:8000/health (Interval: 10s)")
    print("=" * 80)
    try:
        while True:
            check_api_health()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nHealth monitor terminated by user.")
