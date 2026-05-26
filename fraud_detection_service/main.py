"""
FastAPI application for fraud detection model serving
Endpoints for health checks and transaction fraud prediction
"""

import os
import json
import logging
import joblib
import numpy as np
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fraud_detection_service")

app = FastAPI(
    title="Fraud Detection Serving API",
    description="XGBoost model serving engine for IEEE-CIS transaction fraud detection on macOS",
    version="1.0.0"
)

# Global variables for state
model = None
feature_names = None
metadata = None

MODEL_PATH = "models/champion_model.joblib"
FEATURES_PATH = "models/feature_names.json"
METADATA_PATH = "models/champion_metadata.json"

@app.on_event("startup")
async def load_model_on_startup():
    """Load the trained model and associated feature/metric files on service start"""
    global model, feature_names, metadata
    
    logger.info("Initializing Fraud Detection Service...")
    
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"⚠️ Model binary not found at {MODEL_PATH}. Prediction service will be unavailable until trained.")
        return
        
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✓ Model binary successfully loaded from {MODEL_PATH}")
        
        if os.path.exists(FEATURES_PATH):
            with open(FEATURES_PATH, 'r') as f:
                feature_names = json.load(f)
            logger.info(f"✓ Feature names loaded ({len(feature_names)} features)")
        else:
            logger.error(f"❌ Feature names file missing at {FEATURES_PATH}")
            
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            logger.info(f"✓ Model metadata loaded (AUC-ROC: {metadata.get('auc_roc', 0.0):.4f})")
        else:
            logger.warning(f"⚠️ Metadata file missing at {METADATA_PATH}")
            
    except Exception as e:
        logger.error(f"❌ Error loading model artifacts: {str(e)}")

# Pydantic models for request/response structures
class TransactionFeatures(BaseModel):
    TransactionAmt: float = Field(..., description="Transaction amount in USD", example=250.0)
    card1: int = Field(..., description="Payment card 1 (categorical integer code)", example=9999)
    card2: float = Field(-999.0, description="Payment card 2", example=100.0)
    card3: float = Field(-999.0, description="Payment card 3", example=150.0)
    card5: float = Field(-999.0, description="Payment card 5", example=226.0)
    addr1: float = Field(-999.0, description="Billing region / zip area 1", example=299.0)
    addr2: float = Field(-999.0, description="Billing country code 2", example=87.0)
    dist1: float = Field(-999.0, description="Distance between billing address and transaction address", example=500.0)
    dist2: float = Field(-999.0, description="Second distance measure", example=-999.0)
    
    # Optional fields mapping to transaction and identity datasets to allow full flexibility
    C1: float = Field(-999.0, description="Mock count features", example=1.0)
    C2: float = Field(-999.0, description="Mock count features", example=1.0)
    DeviceType: str = Field("-999", description="Type of device (e.g. mobile, desktop)", example="desktop")
    DeviceInfo: str = Field("-999", description="System information of device", example="MacOS")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Legitimate, 1 = Fraud", example=0)
    label: str = Field(..., description="Human readable description", example="Legitimate Transaction")
    probability: float = Field(..., description="Calculated probability of fraud (0 to 1)", example=0.1245)
    risk_level: str = Field(..., description="Assessed risk level: Low, Medium, High", example="Low")
    optimal_threshold: float = Field(..., description="Decision boundary threshold utilized", example=0.8341)
    features_used: int = Field(..., description="Number of model features aligned", example=15)

# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "service": "Fraud Detection serving API",
        "version": "1.0.0",
        "model_loaded": model is not None
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Returns application status, system metrics, and model performance specifications"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "model_loaded": False,
                "detail": "Model is not loaded. Run model_training/train_model.py first."
            }
        )
        
    return {
        "status": "healthy",
        "model_loaded": True,
        "features_count": len(feature_names) if feature_names else 0,
        "auc_roc": metadata.get("auc_roc", 0.0) if metadata else 0.0,
        "optimal_threshold": metadata.get("optimal_threshold", 0.5) if metadata else 0.5,
        "system": "macOS"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_fraud(transaction: TransactionFeatures):
    """Processes transaction metrics and scores fraud risk classification"""
    global model, feature_names, metadata
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Serving engine model not loaded. Run training script."
        )
        
    try:
        # Export request values
        input_dict = transaction.dict()
        
        # XGBoost models expect categorical features to be preprocessed. 
        # Map string inputs in input_dict to categorical index if needed.
        if "DeviceType" in input_dict and isinstance(input_dict["DeviceType"], str):
            device_type_map = {"mobile": 0, "desktop": 1, "tablet": 2}
            input_dict["DeviceType"] = device_type_map.get(input_dict["DeviceType"].lower(), -1)
            
        if "DeviceInfo" in input_dict and isinstance(input_dict["DeviceInfo"], str):
            device_info_map = {"windows": 0, "ios device": 1, "macos": 2, "android": 3, "trident/7.0": 4}
            input_dict["DeviceInfo"] = device_info_map.get(input_dict["DeviceInfo"].lower(), -1)
            
        # Match input values with training feature structure
        feature_values = []
        for name in feature_names:
            if name in input_dict:
                val = input_dict[name]
                if val is None or val == np.nan:
                    feature_values.append(-999)
                else:
                    feature_values.append(val)
            else:
                feature_values.append(-999) # Fill omitted columns with imputation sentinel
                
        # Format array shape (1, n_features)
        features_array = np.array(feature_values).reshape(1, -1)
        
        # Predict probability
        proba = float(model.predict_proba(features_array)[0][1])
        
        # Determine classification by optimal threshold
        threshold = metadata.get("optimal_threshold", 0.5) if metadata else 0.5
        is_fraud = int(proba >= threshold)
        
        # Risk assessment
        if proba >= 0.75:
            risk_level = "High"
        elif proba >= 0.35:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        label = "Fraud Detected" if is_fraud == 1 else "Legitimate Transaction"
        
        return PredictionResponse(
            prediction=is_fraud,
            label=label,
            probability=proba,
            risk_level=risk_level,
            optimal_threshold=threshold,
            features_used=len(feature_names)
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )
