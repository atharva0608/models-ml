"""
AWS Spot Instance Optimizer - PRODUCTION MODEL TRAINER
=======================================================

Single production model using proven v8.0 EVENT-DRIVEN architecture.
Trains on complete 2023-2025 Mumbai data for production deployment.

This replaces the two-model approach (base + regional) with a single
optimized model ready for EC2 deployment.

Version: v8.0 PRODUCTION
Date: 2025-11-11
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from tqdm.auto import tqdm
from pathlib import Path
import shutil

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

print("="*80)
print("AWS SPOT OPTIMIZER - PRODUCTION MODEL TRAINER v8.0")
print("Training on complete 2023-2025 Mumbai data")
print("="*80)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class ProductionConfig:
    """Production model configuration"""
    
    # Data paths - Mumbai 2023-2025
    TRAINING_DATA_2023_2024 = '/Users/atharvapudale/Downloads/aws_2023_2024_complete_24months.csv'
    TEST_Q1_2025 = '/Users/atharvapudale/Downloads/mumbai_spot_data_sorted_asc(1-2-3-25).csv'
    TEST_Q2_2025 = '/Users/atharvapudale/Downloads/mumbai_spot_data_sorted_asc(4-5-6-25).csv'
    TEST_Q3_2025 = '/Users/atharvapudale/Downloads/mumbai_spot_data_sorted_asc(7-8-9-25).csv'
    
    # Model directories
    SCRIPT_DIR = Path(__file__).parent.absolute() if '__file__' in globals() else Path.cwd()
    MODEL_DIR = SCRIPT_DIR / 'models' / 'mumbai_production'
    OUTPUT_DIR = SCRIPT_DIR / 'outputs' / 'mumbai_production'
    PRODUCTION_PACKAGE_DIR = SCRIPT_DIR / 'production_models'
    
    # Region info
    TARGET_REGION = 'ap-south-1'
    REGION_NAME = 'Mumbai'
    INSTANCE_TYPES = ['t3.medium', 't4g.small', 't4g.medium', 'c5.large']
    
    # EVENT DETECTION THRESHOLDS (v8.0 proven values)
    RATIO_SPIKE_THRESHOLD = 0.15
    RATIO_ABSOLUTE_HIGH = 0.70
    RATIO_EVENT_PERCENTILE = 92
    
    PRICE_EVENT_PERCENTILE = 93
    DISCOUNT_EVENT_PERCENTILE = 8
    
    LOOKBACK_HOURS = 6
    MIN_EVENT_DURATION_HOURS = 12
    MAX_EVENT_DURATION_HOURS = 72
    EVENT_COOLDOWN_HOURS = 6
    
    RATIO_SAFE_RETURN = 0.50
    RATIO_SAFE_PERCENTILE = 70
    PRICE_SAFE_PERCENTILE = 75
    
    # POOL SWITCHING (v8.1 efficient values)
    MIN_COST_SAVINGS_NORMAL = 10.0
    MIN_COST_SAVINGS_HIGH_RISK = 25.0
    MIN_COST_SAVINGS_POST_EVENT = 15.0
    MIN_POOL_DURATION_HOURS = 24
    MIN_HOURS_AFTER_OD_RETURN = 48
    MAX_SWITCHES_PER_WEEK = 3
    
    # Target performance
    TARGET_SPOT_MIN = 85
    TARGET_SPOT_MAX = 92
    
    # Training
    MIN_TRAINING_SAMPLES = 200
    LAG_FEATURES = [1, 6, 12, 24]
    ROLLING_WINDOWS = [6, 12, 24]

config = ProductionConfig()
config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config.PRODUCTION_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_standardize_mumbai(csv_path, name):
    """Load and standardize Mumbai data"""
    print(f"\nLoading {name}...")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()
    
    col_map = {}
    for col in df.columns:
        if any(x in col for x in ['time', 'date', 'timestamp']):
            col_map[col] = 'timestamp'
        elif 'spot' in col and 'price' in col:
            col_map[col] = 'SpotPrice'
        elif any(x in col for x in ['ondemand', 'on_demand', 'on-demand']):
            col_map[col] = 'OnDemandPrice'
        elif 'instance' in col and 'type' in col:
            col_map[col] = 'InstanceType'
        elif col in ['az', 'availability_zone', 'availabilityzone']:
            col_map[col] = 'AZ'
        elif col in ['region']:
            col_map[col] = 'Region'
    
    df = df.rename(columns=col_map)
    
    if 'Region' not in df.columns or df['Region'].isna().all():
        if 'AZ' in df.columns:
            df['Region'] = df['AZ'].str.extract(r'^([a-z]+-[a-z]+-\d+)')[0]
    
    df['SpotPrice'] = pd.to_numeric(df['SpotPrice'], errors='coerce')
    df['OnDemandPrice'] = pd.to_numeric(df['OnDemandPrice'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    df = df.dropna(subset=['timestamp', 'SpotPrice', 'OnDemandPrice'])
    df = df[(df['SpotPrice'] > 0) & (df['SpotPrice'] < 10) & (df['OnDemandPrice'] > 0)].copy()
    
    df = df[df['Region'] == config.TARGET_REGION].copy()
    df = df[df['InstanceType'].isin(config.INSTANCE_TYPES)].copy()
    
    df['Pool_ID'] = df['InstanceType'] + '_' + df['AZ']
    df['discount'] = (1 - df['SpotPrice'] / df['OnDemandPrice']).clip(0, 1)
    df['price_ratio'] = (df['SpotPrice'] / df['OnDemandPrice']).clip(0, 10)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"  ✓ {len(df):,} records, {df['Pool_ID'].nunique()} pools")
    print(f"    Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    return df


# ==============================================================================
# CAPACITY EVENT DETECTOR (v8.0)
# ==============================================================================

class ProductionCapacityDetector:
    """Production capacity detector using v8.0 proven logic"""
    
    def __init__(self):
        self.pool_context = {}
        self.pool_history = {}
        
    def calculate_pool_context(self, df_train, pool_id):
        """Calculate historical baselines"""
        pool_data = df_train[df_train['Pool_ID'] == pool_id]
        
        if len(pool_data) < 100:
            return None
        
        return {
            'ratio_p50': pool_data['price_ratio'].quantile(0.50),
            'ratio_p70': pool_data['price_ratio'].quantile(0.70),
            'ratio_p75': pool_data['price_ratio'].quantile(0.75),
            'ratio_p80': pool_data['price_ratio'].quantile(0.80),
            'ratio_p85': pool_data['price_ratio'].quantile(0.85),
            'ratio_p90': pool_data['price_ratio'].quantile(0.90),
            'ratio_p92': pool_data['price_ratio'].quantile(0.92),
            'ratio_p95': pool_data['price_ratio'].quantile(0.95),
            'ratio_mean': pool_data['price_ratio'].mean(),
            'ratio_std': pool_data['price_ratio'].std(),
            
            'price_p50': pool_data['SpotPrice'].quantile(0.50),
            'price_p75': pool_data['SpotPrice'].quantile(0.75),
            'price_p85': pool_data['SpotPrice'].quantile(0.85),
            'price_p90': pool_data['SpotPrice'].quantile(0.90),
            'price_p93': pool_data['SpotPrice'].quantile(0.93),
            'price_p95': pool_data['SpotPrice'].quantile(0.95),
            'price_mean': pool_data['SpotPrice'].mean(),
            'price_std': pool_data['SpotPrice'].std(),
            
            'discount_p05': pool_data['discount'].quantile(0.05),
            'discount_p08': pool_data['discount'].quantile(0.08),
            'discount_p10': pool_data['discount'].quantile(0.10),
            'discount_p25': pool_data['discount'].quantile(0.25),
            'discount_p50': pool_data['discount'].quantile(0.50),
            'discount_mean': pool_data['discount'].mean(),
        }
    
    def train_all(self, df_train):
        """Train on all pools"""
        print(f"\nCapacity Event Detector (v8.0)")
        pools = sorted(df_train['Pool_ID'].unique())
        successful = 0
        
        for pool_id in tqdm(pools, desc="Training pools"):
            context = self.calculate_pool_context(df_train, pool_id)
            if context:
                self.pool_context[pool_id] = context
                self.pool_history[pool_id] = []
                successful += 1
        
        print(f"  ✓ Trained {successful}/{len(pools)} pools")
        
        if successful > 0:
            sample_pool = list(self.pool_context.keys())[0]
            ctx = self.pool_context[sample_pool]
            print(f"\n  Sample: {sample_pool}")
            print(f"    Ratio p50: {ctx['ratio_p50']:.4f} → p92: {ctx['ratio_p92']:.4f} (event)")
            print(f"    Price p50: ${ctx['price_p50']:.6f} → p93: ${ctx['price_p93']:.6f} (event)")
    
    def save(self):
        """Save detector"""
        with open(config.MODEL_DIR / 'capacity_detector.pkl', 'wb') as f:
            pickle.dump({
                'pool_context': self.pool_context,
                'pool_history': self.pool_history,
                'config': {
                    'ratio_absolute_high': config.RATIO_ABSOLUTE_HIGH,
                    'ratio_event_percentile': config.RATIO_EVENT_PERCENTILE,
                    'ratio_safe_return': config.RATIO_SAFE_RETURN,
                    'ratio_spike_threshold': config.RATIO_SPIKE_THRESHOLD,
                    'lookback_hours': config.LOOKBACK_HOURS,
                }
            }, f)
        print(f"  ✓ Saved capacity detector")


# ==============================================================================
# PRICE PREDICTOR (v8.0)
# ==============================================================================

class ProductionPricePredictor:
    """Production price predictor using v8.0 proven logic"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        
    def create_features(self, df, pool_id):
        """Create features for a pool"""
        pool_data = df[df['Pool_ID'] == pool_id].copy().sort_values('timestamp')
        
        pool_data['hour'] = pool_data['timestamp'].dt.hour
        pool_data['day_of_week'] = pool_data['timestamp'].dt.dayofweek
        
        for lag in config.LAG_FEATURES:
            pool_data[f'spot_lag_{lag}'] = pool_data['SpotPrice'].shift(lag)
            pool_data[f'discount_lag_{lag}'] = pool_data['discount'].shift(lag)
        
        for window in config.ROLLING_WINDOWS:
            pool_data[f'spot_roll_mean_{window}'] = pool_data['SpotPrice'].rolling(window).mean()
            pool_data[f'discount_roll_mean_{window}'] = pool_data['discount'].rolling(window).mean()
        
        return pool_data.dropna()
    
    def train_pool(self, df, pool_id):
        """Train model for a pool"""
        try:
            pool_data = self.create_features(df, pool_id)
            if len(pool_data) < config.MIN_TRAINING_SAMPLES:
                return None, "Insufficient samples"
            
            # Get numeric columns only
            numeric_cols = pool_data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in 
                          ['SpotPrice', 'OnDemandPrice', 'discount', 'price_ratio']]
            
            X = pool_data[feature_cols].values.astype(np.float64)
            y = pool_data['SpotPrice'].values.astype(np.float64)
            
            # Handle NaN/Inf
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                X = np.nan_to_num(X, nan=0.0, posinf=999.0, neginf=-999.0)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            gb = GradientBoostingRegressor(
                n_estimators=50, 
                learning_rate=0.1, 
                max_depth=4, 
                random_state=42
            )
            gb.fit(X_scaled, y)
            
            self.models[pool_id] = {'gb': gb, 'feature_cols': feature_cols}
            self.scalers[pool_id] = scaler
            
            mae = mean_absolute_error(y, gb.predict(X_scaled))
            self.metrics[pool_id] = {'mae': mae, 'samples': len(pool_data)}
            return self.metrics[pool_id], None
            
        except Exception as e:
            return None, str(e)
    
    def train_all(self, df_train):
        """Train all pools"""
        print(f"\nPrice Predictor (v8.0)")
        pools = sorted(df_train['Pool_ID'].unique())
        successful = 0
        failed = []
        
        for pool_id in tqdm(pools, desc="Training pools"):
            metrics, error = self.train_pool(df_train, pool_id)
            if metrics:
                successful += 1
            else:
                failed.append((pool_id, error))
        
        if successful > 0:
            avg_mae = np.mean([m['mae'] for m in self.metrics.values()])
            print(f"  ✓ Trained {successful}/{len(pools)} pools, Avg MAE: ${avg_mae:.6f}")
        
        if failed:
            print(f"\n  ⚠ Failed pools:")
            for pool_id, error in failed[:3]:  # Show first 3
                print(f"    - {pool_id}: {error}")
        
        if successful == 0:
            raise RuntimeError("No pools trained successfully!")
    
    def save(self):
        """Save models"""
        if len(self.models) > 0:
            with open(config.MODEL_DIR / 'price_predictor.pkl', 'wb') as f:
                pickle.dump({
                    'models': self.models,
                    'scalers': self.scalers,
                    'metrics': self.metrics
                }, f)
            print(f"  ✓ Saved {len(self.models)} price models")


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def main():
    """Main production training pipeline"""
    
    print("\n" + "="*80)
    print("PRODUCTION MODEL TRAINING - MUMBAI v8.0")
    print("="*80)
    
    # STEP 1: Load all Mumbai data (2023-2025)
    print("\n" + "="*80)
    print("STEP 1: LOADING COMPLETE MUMBAI DATA (2023-2025)")
    print("="*80)
    
    df_2023_2024 = load_and_standardize_mumbai(config.TRAINING_DATA_2023_2024, "2023-2024")
    df_q1_2025 = load_and_standardize_mumbai(config.TEST_Q1_2025, "Q1 2025")
    df_q2_2025 = load_and_standardize_mumbai(config.TEST_Q2_2025, "Q2 2025")
    df_q3_2025 = load_and_standardize_mumbai(config.TEST_Q3_2025, "Q3 2025")
    
    # Combine all data for production training
    df_all = pd.concat([df_2023_2024, df_q1_2025, df_q2_2025, df_q3_2025])
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n✓ Combined training set:")
    print(f"  Total Records: {len(df_all):,}")
    print(f"  Date Range: {df_all['timestamp'].min().date()} to {df_all['timestamp'].max().date()}")
    print(f"  Pools: {df_all['Pool_ID'].nunique()}")
    print(f"  Duration: {(df_all['timestamp'].max() - df_all['timestamp'].min()).days} days")
    
    # STEP 2: Train capacity detector
    print("\n" + "="*80)
    print("STEP 2: TRAINING CAPACITY EVENT DETECTOR")
    print("="*80)
    
    capacity_detector = ProductionCapacityDetector()
    capacity_detector.train_all(df_all)
    capacity_detector.save()
    
    # STEP 3: Train price predictor
    print("\n" + "="*80)
    print("STEP 3: TRAINING PRICE PREDICTOR")
    print("="*80)
    
    price_predictor = ProductionPricePredictor()
    price_predictor.train_all(df_all)
    price_predictor.save()
    
    # STEP 4: Save configuration
    print("\n" + "="*80)
    print("STEP 4: SAVING CONFIGURATION")
    print("="*80)
    
    production_config = {
        'version': 'v8.0 PRODUCTION',
        'region': config.TARGET_REGION,
        'region_name': config.REGION_NAME,
        'training_date': datetime.now().isoformat(),
        'instance_types': config.INSTANCE_TYPES,
        'training_records': len(df_all),
        'pools_trained': len(capacity_detector.pool_context),
        'date_range': {
            'start': df_all['timestamp'].min().isoformat(),
            'end': df_all['timestamp'].max().isoformat(),
            'days': (df_all['timestamp'].max() - df_all['timestamp'].min()).days
        },
        'event_thresholds': {
            'ratio_absolute_high': config.RATIO_ABSOLUTE_HIGH,
            'ratio_event_percentile': config.RATIO_EVENT_PERCENTILE,
            'ratio_safe_return': config.RATIO_SAFE_RETURN,
            'ratio_spike_threshold': config.RATIO_SPIKE_THRESHOLD,
            'lookback_hours': config.LOOKBACK_HOURS,
            'min_event_duration_hours': config.MIN_EVENT_DURATION_HOURS,
            'max_event_duration_hours': config.MAX_EVENT_DURATION_HOURS,
        },
        'pool_switching': {
            'min_cost_savings_normal': config.MIN_COST_SAVINGS_NORMAL,
            'min_cost_savings_high_risk': config.MIN_COST_SAVINGS_HIGH_RISK,
            'min_pool_duration_hours': config.MIN_POOL_DURATION_HOURS,
            'max_switches_per_week': config.MAX_SWITCHES_PER_WEEK,
        },
        'performance_targets': {
            'spot_usage_min': config.TARGET_SPOT_MIN,
            'spot_usage_max': config.TARGET_SPOT_MAX,
        }
    }
    
    with open(config.MODEL_DIR / 'config.json', 'w') as f:
        json.dump(production_config, f, indent=2, default=str)
    
    print(f"  ✓ Saved configuration")
    
    # STEP 5: Create production package
    print("\n" + "="*80)
    print("STEP 5: CREATING PRODUCTION PACKAGE")
    print("="*80)
    
    # Create package directory
    (config.PRODUCTION_PACKAGE_DIR / 'mumbai').mkdir(parents=True, exist_ok=True)
    
    # Copy models
    files_to_copy = [
        ('capacity_detector.pkl', 'Capacity Detector'),
        ('price_predictor.pkl', 'Price Predictor'),
        ('config.json', 'Configuration')
    ]
    
    print("  Copying models to production package...")
    for filename, description in files_to_copy:
        src = config.MODEL_DIR / filename
        dst = config.PRODUCTION_PACKAGE_DIR / 'mumbai' / filename
        if src.exists():
            shutil.copy(src, dst)
            size_kb = src.stat().st_size / 1024
            print(f"    ✓ {description}: {size_kb:.1f} KB")
        else:
            print(f"    ✗ {description}: NOT FOUND")
    
    # Create deployment manifest
    manifest = {
        'package_version': 'v8.0 PRODUCTION',
        'created': datetime.now().isoformat(),
        'region': config.TARGET_REGION,
        'models': {
            'mumbai': {
                'capacity_detector': 'mumbai/capacity_detector.pkl',
                'price_predictor': 'mumbai/price_predictor.pkl',
                'config': 'mumbai/config.json'
            }
        },
        'deployment': {
            'ec2_path': '/home/ubuntu/production_models',
            'backend_config': 'Update MODEL_DIR in backend.py',
            'requires': ['Flask', 'pandas', 'numpy', 'scikit-learn']
        }
    }
    
    with open(config.PRODUCTION_PACKAGE_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  ✓ Created deployment manifest")
    print(f"\n  Production package location:")
    print(f"    {config.PRODUCTION_PACKAGE_DIR.absolute()}")
    
    # STEP 6: Summary
    print("\n" + "="*80)
    print("PRODUCTION MODEL TRAINING COMPLETE!")
    print("="*80)
    
    print(f"\n✅ Training Summary:")
    print(f"  Region: {config.REGION_NAME} ({config.TARGET_REGION})")
    print(f"  Training Period: 2023-2025 ({(df_all['timestamp'].max() - df_all['timestamp'].min()).days} days)")
    print(f"  Total Records: {len(df_all):,}")
    print(f"  Pools Trained: {len(capacity_detector.pool_context)}")
    print(f"  Model Version: v8.0 EVENT-DRIVEN")
    
    print(f"\n✅ Production Package:")
    print(f"  Location: {config.PRODUCTION_PACKAGE_DIR.absolute()}")
    print(f"  Files:")
    print(f"    - mumbai/capacity_detector.pkl")
    print(f"    - mumbai/price_predictor.pkl")
    print(f"    - mumbai/config.json")
    print(f"    - manifest.json")
    
    print(f"\n✅ Model Capabilities:")
    print(f"  - Event detection (ratio spike, absolute, percentile)")
    print(f"  - Price prediction (per-pool gradient boosting)")
    print(f"  - Safe return detection (aggressive)")
    print(f"  - Pool optimization (efficient switching)")
    
    print(f"\n📦 Next Steps:")
    print(f"  1. Upload to EC2:")
    print(f"     scp -r production_models/ ubuntu@YOUR_EC2_IP:/home/ubuntu/")
    print(f"  2. Deploy backend (see backend_setup.md)")
    print(f"  3. Test API endpoints")
    print(f"  4. Build agent for execution")
    
    print("\n" + "="*80)
    print("READY FOR PRODUCTION DEPLOYMENT! 🚀")
    print("="*80)


if __name__ == "__main__":
    main()
