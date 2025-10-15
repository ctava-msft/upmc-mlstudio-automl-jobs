"""
Feature Engineering Module
Provides configurable data preprocessing and feature engineering capabilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles all feature engineering transformations based on configuration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature engineer with configuration
        
        Args:
            config: Feature engineering configuration dictionary
        """
        self.config = config
        self.scalers = {}
        
    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all configured transformations to the dataframe
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        if not self.config.get('enabled', True):
            logger.info("Feature engineering is disabled")
            return df
        
        df = df.copy()
        
        # Apply binning
        if self.config.get('binning', {}).get('enabled', False):
            df = self._apply_binning(df)
        
        # Apply datetime features
        if self.config.get('datetime_features', {}).get('enabled', False):
            df = self._apply_datetime_features(df)
        
        # Apply custom transformations
        if self.config.get('custom', {}).get('enabled', False):
            df = self._apply_custom_transformations(df)
        
        # Apply scaling
        if self.config.get('scaling', {}).get('enabled', False):
            df = self._apply_scaling(df)
        
        # Apply encoding
        if self.config.get('encoding', {}).get('enabled', False):
            df = self._apply_encoding(df)
        
        # Apply interactions
        if self.config.get('interactions', {}).get('enabled', False):
            df = self._apply_interactions(df)
        
        # Apply polynomial features
        if self.config.get('polynomial', {}).get('enabled', False):
            df = self._apply_polynomial(df)
        
        return df
    
    def _apply_binning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply binning transformations"""
        binning_config = self.config.get('binning', {})
        features = binning_config.get('features', [])
        
        for feature in features:
            column = feature.get('column')
            bins = feature.get('bins')
            labels = feature.get('labels')
            new_column = feature.get('new_column', f"{column}_binned")
            
            if column not in df.columns:
                logger.warning(f"Column {column} not found for binning")
                continue
            
            try:
                df[new_column] = pd.cut(df[column], bins=bins, labels=labels)
                logger.info(f"Created binned feature: {new_column}")
            except Exception as e:
                logger.error(f"Error binning column {column}: {str(e)}")
        
        return df
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling transformations"""
        scaling_config = self.config.get('scaling', {})
        method = scaling_config.get('method', 'standard')
        columns = scaling_config.get('columns', [])
        
        if not columns:
            return df
        
        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}")
            return df
        
        # Apply scaling to specified columns
        existing_columns = [col for col in columns if col in df.columns]
        if existing_columns:
            df[existing_columns] = scaler.fit_transform(df[existing_columns])
            self.scalers[method] = scaler
            logger.info(f"Applied {method} scaling to {len(existing_columns)} columns")
        
        return df
    
    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding"""
        encoding_config = self.config.get('encoding', {})
        method = encoding_config.get('method', 'onehot')
        columns = encoding_config.get('columns', [])
        
        if not columns:
            return df
        
        existing_columns = [col for col in columns if col in df.columns]
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=existing_columns, drop_first=True)
            logger.info(f"Applied one-hot encoding to {len(existing_columns)} columns")
        elif method == 'label':
            for col in existing_columns:
                df[col] = df[col].astype('category').cat.codes
            logger.info(f"Applied label encoding to {len(existing_columns)} columns")
        
        return df
    
    def _apply_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract datetime features"""
        datetime_config = self.config.get('datetime_features', {})
        columns = datetime_config.get('columns', [])
        
        for col_config in columns:
            column = col_config.get('column')
            extract = col_config.get('extract', [])
            
            if column not in df.columns:
                logger.warning(f"Column {column} not found for datetime extraction")
                continue
            
            try:
                df[column] = pd.to_datetime(df[column])
                
                if 'year' in extract:
                    df[f"{column}_year"] = df[column].dt.year
                if 'month' in extract:
                    df[f"{column}_month"] = df[column].dt.month
                if 'day' in extract:
                    df[f"{column}_day"] = df[column].dt.day
                if 'dayofweek' in extract:
                    df[f"{column}_dayofweek"] = df[column].dt.dayofweek
                if 'quarter' in extract:
                    df[f"{column}_quarter"] = df[column].dt.quarter
                if 'hour' in extract:
                    df[f"{column}_hour"] = df[column].dt.hour
                
                logger.info(f"Extracted datetime features from {column}")
            except Exception as e:
                logger.error(f"Error extracting datetime features from {column}: {str(e)}")
        
        return df
    
    def _apply_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        interactions_config = self.config.get('interactions', {})
        feature_pairs = interactions_config.get('feature_pairs', [])
        
        for pair in feature_pairs:
            if len(pair) != 2:
                continue
            
            col1, col2 = pair
            if col1 in df.columns and col2 in df.columns:
                interaction_name = f"{col1}_x_{col2}"
                df[interaction_name] = df[col1] * df[col2]
                logger.info(f"Created interaction feature: {interaction_name}")
        
        return df
    
    def _apply_polynomial(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features"""
        poly_config = self.config.get('polynomial', {})
        degree = poly_config.get('degree', 2)
        columns = poly_config.get('columns', [])
        
        existing_columns = [col for col in columns if col in df.columns]
        
        for col in existing_columns:
            for d in range(2, degree + 1):
                new_col = f"{col}_power_{d}"
                df[new_col] = df[col] ** d
                logger.info(f"Created polynomial feature: {new_col}")
        
        return df
    
    def _apply_custom_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply custom Python expressions"""
        custom_config = self.config.get('custom', {})
        transformations = custom_config.get('transformations', [])
        
        for transform in transformations:
            name = transform.get('name')
            expression = transform.get('expression')
            
            if not name or not expression:
                continue
            
            try:
                # Evaluate the expression with df in scope
                result = eval(expression, {'pd': pd, 'np': np, 'df': df})
                df[name] = result
                logger.info(f"Created custom feature: {name}")
            except Exception as e:
                logger.error(f"Error creating custom feature {name}: {str(e)}")
        
        return df


def validate_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Validate data according to configuration
    
    Args:
        df: Input dataframe
        config: Validation configuration
        
    Returns:
        Validated dataframe
    """
    if not config.get('enabled', True):
        return df
    
    # Check for missing values
    if config.get('check_missing_values', True):
        max_missing_pct = config.get('max_missing_percentage', 50)
        missing_pct = (df.isnull().sum() / len(df)) * 100
        cols_to_drop = missing_pct[missing_pct > max_missing_pct].index.tolist()
        
        if cols_to_drop:
            logger.warning(f"Dropping {len(cols_to_drop)} columns with >{max_missing_pct}% missing values")
            df = df.drop(columns=cols_to_drop)
    
    # Check for duplicates
    if config.get('check_duplicates', True):
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            logger.warning(f"Found {dup_count} duplicate rows, removing them")
            df = df.drop_duplicates()
    
    return df
