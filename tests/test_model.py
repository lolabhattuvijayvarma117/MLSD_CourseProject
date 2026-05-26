"""
Unit tests for data preparation and XGBoost training pipeline
"""

import os
import json
import pytest
import pandas as pd
import joblib
from model_training.train_model import train

def test_dataset_generation():
    """Verify that train_transaction.csv and train_identity.csv exist and are valid"""
    tx_path = 'data/train_transaction.csv'
    id_path = 'data/train_identity.csv'
    
    assert os.path.exists(tx_path), "Transaction data file missing"
    assert os.path.exists(id_path), "Identity data file missing"
    
    tx_df = pd.read_csv(tx_path)
    id_df = pd.read_csv(id_path)
    
    assert len(tx_df) > 0, "Transaction dataset is empty"
    assert len(id_df) > 0, "Identity dataset is empty"
    assert 'TransactionID' in tx_df.columns, "Transaction ID column missing"
    assert 'isFraud' in tx_df.columns, "Target column isFraud missing"
    assert 'TransactionID' in id_df.columns, "Identity ID column missing"

def test_model_training_outputs():
    """Verify that running the training script exports expected model binaries and metrics"""
    # Execute training
    train()
    
    # Check that output files exist
    assert os.path.exists('models/champion_model.joblib'), "Model binary was not saved"
    assert os.path.exists('models/feature_names.json'), "Feature list file was not saved"
    assert os.path.exists('models/champion_metadata.json'), "Metadata file was not saved"
    
    # Load model and verify
    model = joblib.load('models/champion_model.joblib')
    assert model is not None, "Model failed to load"
    
    # Load feature list and verify
    with open('models/feature_names.json', 'r') as f:
        features = json.load(f)
    assert len(features) > 0, "Feature list is empty"
    
    # Load metadata and verify values
    with open('models/champion_metadata.json', 'r') as f:
        meta = json.load(f)
        
    assert 'auc_roc' in meta, "AUC-ROC missing in metadata"
    assert 'pr_auc' in meta, "PR-AUC missing in metadata"
    assert 'optimal_threshold' in meta, "Optimal threshold missing in metadata"
    
    assert 0.0 <= meta['auc_roc'] <= 1.0, f"Invalid AUC-ROC: {meta['auc_roc']}"
    assert 0.0 <= meta['pr_auc'] <= 1.0, f"Invalid PR-AUC: {meta['pr_auc']}"
    assert 0.0 <= meta['optimal_threshold'] <= 1.0, f"Invalid threshold: {meta['optimal_threshold']}"
