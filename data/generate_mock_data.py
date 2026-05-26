import os
import numpy as np
import pandas as pd

def generate_mock_datasets():
    print("Generating synthetic IEEE-CIS Fraud Detection datasets...")
    np.random.seed(42)
    
    n_samples = 2500
    n_identity = 800
    
    # 1. Generate Transactions
    transaction_ids = np.arange(2987000, 2987000 + n_samples)
    
    # Fraud rate ~ 3.5%
    is_fraud = np.random.choice([0, 1], size=n_samples, p=[0.965, 0.035])
    
    transaction_amt = np.random.exponential(scale=120.0, size=n_samples) + 1.0
    transaction_dt = np.arange(100000, 100000 + n_samples * 10, 10)
    
    product_cd = np.random.choice(['W', 'H', 'C', 'S', 'R'], size=n_samples)
    
    card1 = np.random.randint(1000, 18000, size=n_samples)
    card2 = np.random.choice([100.0, 200.0, 300.0, 400.0, 500.0, np.nan], size=n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    card3 = np.random.choice([150.0, 185.0, 100.0, np.nan], size=n_samples, p=[0.7, 0.1, 0.1, 0.1])
    card4 = np.random.choice(['visa', 'mastercard', 'american express', 'discover', np.nan], size=n_samples, p=[0.6, 0.3, 0.05, 0.03, 0.02])
    card5 = np.random.choice([117.0, 226.0, 166.0, 102.0, np.nan], size=n_samples, p=[0.3, 0.4, 0.1, 0.1, 0.1])
    card6 = np.random.choice(['debit', 'credit', np.nan], size=n_samples, p=[0.7, 0.28, 0.02])
    
    addr1 = np.random.choice([299.0, 325.0, 204.0, 126.0, np.nan], size=n_samples, p=[0.4, 0.3, 0.1, 0.1, 0.1])
    addr2 = np.random.choice([87.0, np.nan], size=n_samples, p=[0.95, 0.05])
    
    dist1 = np.random.exponential(scale=50.0, size=n_samples)
    # Inject missing values in dist1 (about 50%)
    dist1[np.random.choice([True, False], size=n_samples, p=[0.5, 0.5])] = np.nan
    
    dist2 = np.random.exponential(scale=100.0, size=n_samples)
    # Inject missing values in dist2 (about 90% missing, which will be filtered out by threshold)
    dist2[np.random.choice([True, False], size=n_samples, p=[0.92, 0.08])] = np.nan
    
    # C1 - C14 features (typically transaction counts)
    c_features = {}
    for i in range(1, 15):
        c_features[f'C{i}'] = np.random.poisson(lam=1.5, size=n_samples)
        
    # D1 - D15 features (time deltas)
    d_features = {}
    for i in range(1, 16):
        d_val = np.random.exponential(scale=30.0, size=n_samples)
        # Inject ~30% missing values
        d_val[np.random.choice([True, False], size=n_samples, p=[0.3, 0.7])] = np.nan
        d_features[f'D{i}'] = d_val
        
    # V1 - V50 features (engineered features, float)
    v_features = {}
    for i in range(1, 51):
        v_val = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
        # Inject varying missingness
        missing_p = np.random.choice([0.1, 0.2, 0.5, 0.85, 0.95])
        v_val[np.random.choice([True, False], size=n_samples, p=[missing_p, 1-missing_p])] = np.nan
        v_features[f'V{i}'] = v_val
        
    tx_dict = {
        'TransactionID': transaction_ids,
        'isFraud': is_fraud,
        'TransactionDT': transaction_dt,
        'TransactionAmt': transaction_amt,
        'ProductCD': product_cd,
        'card1': card1,
        'card2': card2,
        'card3': card3,
        'card4': card4,
        'card5': card5,
        'card6': card6,
        'addr1': addr1,
        'addr2': addr2,
        'dist1': dist1,
        'dist2': dist2,
    }
    tx_dict.update(c_features)
    tx_dict.update(d_features)
    tx_dict.update(v_features)
    
    tx_df = pd.DataFrame(tx_dict)
    
    # 2. Generate Identity table (only for a subset of transactions)
    identity_ids = np.random.choice(transaction_ids, size=n_identity, replace=False)
    
    id_features = {}
    # id_01 to id_10 (numeric)
    for i in range(1, 11):
        id_val = np.random.normal(loc=0.0, scale=5.0, size=n_identity)
        id_val[np.random.choice([True, False], size=n_identity, p=[0.2, 0.8])] = np.nan
        id_features[f'id_{i:02d}'] = id_val
        
    # id_11 to id_38 (high-missingness identifiers)
    for i in range(11, 39):
        id_val = np.random.normal(loc=0.0, scale=1.0, size=n_identity)
        # 95% missing values
        id_val[np.random.choice([True, False], size=n_identity, p=[0.95, 0.05])] = np.nan
        id_features[f'id_{i}'] = id_val
        
    device_type = np.random.choice(['mobile', 'desktop', 'tablet', np.nan], size=n_identity, p=[0.5, 0.4, 0.05, 0.05])
    device_info = np.random.choice(['Windows', 'iOS Device', 'MacOS', 'Android', 'Trident/7.0', np.nan], size=n_identity, p=[0.4, 0.2, 0.15, 0.15, 0.05, 0.05])
    
    id_dict = {
        'TransactionID': identity_ids,
        'DeviceType': device_type,
        'DeviceInfo': device_info,
    }
    id_dict.update(id_features)
    
    id_df = pd.DataFrame(id_dict)
    
    # Write to CSV
    os.makedirs('data', exist_ok=True)
    tx_df.to_csv('data/train_transaction.csv', index=False)
    id_df.to_csv('data/train_identity.csv', index=False)
    
    print(f"✓ Synthetic Transaction dataset generated: {tx_df.shape} saved to data/train_transaction.csv")
    print(f"✓ Synthetic Identity dataset generated: {id_df.shape} saved to data/train_identity.csv")

if __name__ == '__main__':
    generate_mock_datasets()
