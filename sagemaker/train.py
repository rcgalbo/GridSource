"""
GridSource Bank Liquidity Forecasting Model Training Script

This script trains a machine learning model to predict energy banking liquidity needs
based on electricity generation, weather, and economic factors.

The model is designed to run in Amazon SageMaker and follows SageMaker conventions:
- Training data is read from /opt/ml/input/data/training/
- Model artifacts are saved to /opt/ml/model/
- Hyperparameters are passed via environment variables

Model Features:
- total_generation_mwh: Total electricity generation
- avg_temperature_f: Average temperature 
- oil_price_usd: Crude oil price
- industrial_production_index: Economic activity indicator
- avg_electricity_price: Average electricity price

Target Variable:
- liquidity_need_millions: Liquidity requirement in millions USD
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiquidityForecastingModel:
    """
    Energy Banking Liquidity Forecasting Model
    
    This class encapsulates the entire ML pipeline for predicting
    liquidity needs based on energy and economic factors.
    """
    
    def __init__(self, model_type='linear_regression', random_state=42):
        """
        Initialize the forecasting model
        
        Args:
            model_type (str): Type of model to use ('linear_regression', 'ridge', 'random_forest')
            random_state (int): Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.training_metrics = {}
        
        # Initialize the appropriate model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the machine learning model based on model_type"""
        if self.model_type == 'linear_regression':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=self.random_state)
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def load_and_prepare_data(self, data_path):
        """
        Load training data and prepare features
        
        Args:
            data_path (str): Path to the training CSV file
            
        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        logger.info(f"Loading training data from {data_path}")
        
        # Load the CSV file
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} training samples")
        
        # Log basic data statistics
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Define feature columns (all except target)
        feature_columns = [
            'total_generation_mwh',
            'avg_temperature_f', 
            'oil_price_usd',
            'industrial_production_index',
            'avg_electricity_price'
        ]
        
        target_column = 'liquidity_need_millions'
        
        # Check for required columns
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")
        
        if target_column not in df.columns:
            raise ValueError(f"Missing target column: {target_column}")
        
        # Extract features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        logger.info("Handling missing values...")
        
        # Log missing value counts
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values found: {missing_counts.to_dict()}")
            
            # Simple forward fill for missing values
            X = X.fillna(method='ffill').fillna(method='bfill')
            
            # If still missing, fill with median
            X = X.fillna(X.median())
        
        # Remove any rows with missing targets
        valid_rows = ~y.isnull()
        X = X[valid_rows]
        y = y[valid_rows]
        
        logger.info(f"Final dataset size: {len(X)} samples with {len(feature_columns)} features")
        
        # Store feature names for later use
        self.feature_names = feature_columns
        
        # Log basic statistics
        logger.info("Target variable statistics:")
        logger.info(f"  Mean: ${y.mean():.2f}M")
        logger.info(f"  Std:  ${y.std():.2f}M")
        logger.info(f"  Min:  ${y.min():.2f}M")
        logger.info(f"  Max:  ${y.max():.2f}M")
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """
        Train the liquidity forecasting model
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            test_size (float): Fraction of data to use for testing
        """
        logger.info("Starting model training...")
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train the model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Generate predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred, "Training")
        val_metrics = self._calculate_metrics(y_val, y_val_pred, "Validation")
        
        # Store metrics
        self.training_metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val)
        }
        
        # Log feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Feature Importance:")
            for _, row in importance_df.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        logger.info("Model training completed successfully!")
    
    def _calculate_metrics(self, y_true, y_pred, set_name):
        """Calculate and log model performance metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"{set_name} Metrics:")
        logger.info(f"  MAE:  ${mae:.2f}M")
        logger.info(f"  RMSE: ${rmse:.2f}M") 
        logger.info(f"  R²:   {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def save_model(self, model_dir):
        """
        Save the trained model and metadata
        
        Args:
            model_dir (str): Directory to save model artifacts
        """
        logger.info(f"Saving model to {model_dir}")
        
        # Ensure directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model and scaler
        joblib.dump(self.model, os.path.join(model_dir, 'model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        # Save feature names
        with open(os.path.join(model_dir, 'feature_names.json'), 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save training metrics and metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'sklearn_version': str(pd.__version__),  # Using pandas version as proxy
            'training_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Model saved successfully!")
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train GridSource Liquidity Forecasting Model')
    
    # SageMaker provides these paths
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model',
                       help='Directory to save model artifacts')
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/training',
                       help='Directory containing training data')
    
    # Model hyperparameters
    parser.add_argument('--model-type', type=str, default='linear_regression',
                       choices=['linear_regression', 'ridge', 'random_forest'],
                       help='Type of model to train')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for validation')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main training function"""
    logger.info("Starting GridSource Liquidity Forecasting Model Training")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Model type: {args.model_type}")
    logger.info(f"  Test size: {args.test_size}")
    logger.info(f"  Random state: {args.random_state}")
    logger.info(f"  Training data path: {args.train}")
    logger.info(f"  Model output path: {args.model_dir}")
    
    try:
        # Initialize model
        model = LiquidityForecastingModel(
            model_type=args.model_type,
            random_state=args.random_state
        )
        
        # Find training data file
        train_file = None
        for file in os.listdir(args.train):
            if file.endswith('.csv'):
                train_file = os.path.join(args.train, file)
                break
        
        if train_file is None:
            raise FileNotFoundError(f"No CSV file found in {args.train}")
        
        # Load and prepare data
        X, y = model.load_and_prepare_data(train_file)
        
        # Train model
        model.train(X, y, test_size=args.test_size)
        
        # Save model
        model.save_model(args.model_dir)
        
        logger.info("Training completed successfully!")
        
        # Print final metrics for SageMaker logs
        val_metrics = model.training_metrics['validation']
        print(f"Final Validation MAE: ${val_metrics['mae']:.2f}M")
        print(f"Final Validation R²: {val_metrics['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()