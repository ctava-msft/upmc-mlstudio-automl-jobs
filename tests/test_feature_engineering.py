"""
Unit tests for feature_engineering.py
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engineering import FeatureEngineer, validate_data


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class"""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization"""
        config = {'enabled': True}
        fe = FeatureEngineer(config)
        assert fe.config == config
        assert fe.scalers == {}
    
    def test_apply_transformations_disabled(self):
        """Test that transformations are skipped when disabled"""
        config = {'enabled': False}
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = fe.apply_transformations(df)
        pd.testing.assert_frame_equal(result, df)
    
    def test_binning_basic(self):
        """Test basic binning transformation"""
        config = {
            'enabled': True,
            'binning': {
                'enabled': True,
                'features': [
                    {
                        'column': 'age',
                        'bins': [0, 18, 35, 50, 100],
                        'labels': ['Child', 'Young', 'Adult', 'Senior'],
                        'new_column': 'age_group'
                    }
                ]
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'age': [10, 25, 40, 60]})
        result = fe.apply_transformations(df)
        
        assert 'age_group' in result.columns
        assert list(result['age_group']) == ['Child', 'Young', 'Adult', 'Senior']
    
    def test_binning_missing_column(self):
        """Test binning with missing column"""
        config = {
            'enabled': True,
            'binning': {
                'enabled': True,
                'features': [
                    {
                        'column': 'nonexistent',
                        'bins': [0, 10, 20],
                        'labels': ['Low', 'High'],
                        'new_column': 'binned'
                    }
                ]
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'age': [10, 25, 40]})
        result = fe.apply_transformations(df)
        
        # Should not add the binned column if source column doesn't exist
        assert 'binned' not in result.columns
    
    def test_datetime_features_extraction(self):
        """Test datetime feature extraction"""
        config = {
            'enabled': True,
            'datetime_features': {
                'enabled': True,
                'columns': [
                    {
                        'column': 'date',
                        'extract': ['year', 'month', 'day', 'dayofweek', 'quarter']
                    }
                ]
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({
            'date': ['2023-01-15', '2023-06-20', '2023-12-31']
        })
        result = fe.apply_transformations(df)
        
        assert 'date_year' in result.columns
        assert 'date_month' in result.columns
        assert 'date_day' in result.columns
        assert 'date_dayofweek' in result.columns
        assert 'date_quarter' in result.columns
        
        assert list(result['date_year']) == [2023, 2023, 2023]
        assert list(result['date_month']) == [1, 6, 12]
        assert list(result['date_day']) == [15, 20, 31]
        assert list(result['date_quarter']) == [1, 2, 4]
    
    def test_datetime_features_missing_column(self):
        """Test datetime feature extraction with missing column"""
        config = {
            'enabled': True,
            'datetime_features': {
                'enabled': True,
                'columns': [
                    {
                        'column': 'nonexistent_date',
                        'extract': ['year', 'month']
                    }
                ]
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'value': [1, 2, 3]})
        result = fe.apply_transformations(df)
        
        # Should not add datetime columns if source column doesn't exist
        assert 'nonexistent_date_year' not in result.columns
        assert 'nonexistent_date_month' not in result.columns
    
    def test_scaling_standard(self):
        """Test standard scaling"""
        config = {
            'enabled': True,
            'scaling': {
                'enabled': True,
                'method': 'standard',
                'columns': ['value1', 'value2']
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({
            'value1': [10, 20, 30, 40, 50],
            'value2': [100, 200, 300, 400, 500]
        })
        result = fe.apply_transformations(df)
        
        # Check that scaling was applied (values should have mean ~0 and std close to 1)
        assert abs(result['value1'].mean()) < 1e-10
        # StandardScaler uses ddof=0 by default, pandas uses ddof=1
        assert abs(result['value1'].std(ddof=0) - 1.0) < 0.01
    
    def test_scaling_minmax(self):
        """Test MinMax scaling"""
        config = {
            'enabled': True,
            'scaling': {
                'enabled': True,
                'method': 'minmax',
                'columns': ['value']
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        result = fe.apply_transformations(df)
        
        # MinMax scaling should result in values between 0 and 1
        assert result['value'].min() == 0.0
        assert result['value'].max() == 1.0
    
    def test_scaling_robust(self):
        """Test Robust scaling"""
        config = {
            'enabled': True,
            'scaling': {
                'enabled': True,
                'method': 'robust',
                'columns': ['value']
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        result = fe.apply_transformations(df)
        
        # Result should be scaled (different from original)
        assert not result['value'].equals(df['value'])
    
    def test_encoding_onehot(self):
        """Test one-hot encoding"""
        config = {
            'enabled': True,
            'encoding': {
                'enabled': True,
                'method': 'onehot',
                'columns': ['category']
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'category': ['A', 'B', 'C', 'A', 'B']})
        result = fe.apply_transformations(df)
        
        # Original column should be replaced with dummy columns
        assert 'category' not in result.columns
        # Should have n-1 dummy columns (drop_first=True)
        assert 'category_B' in result.columns or 'category_C' in result.columns
    
    def test_encoding_label(self):
        """Test label encoding"""
        config = {
            'enabled': True,
            'encoding': {
                'enabled': True,
                'method': 'label',
                'columns': ['category']
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'category': ['A', 'B', 'C', 'A', 'B']})
        result = fe.apply_transformations(df)
        
        # Column should still exist but with integer values
        assert 'category' in result.columns
        assert result['category'].dtype in [np.int8, np.int16, np.int32, np.int64]
    
    def test_interactions(self):
        """Test interaction features"""
        config = {
            'enabled': True,
            'interactions': {
                'enabled': True,
                'feature_pairs': [
                    ['feature1', 'feature2'],
                    ['feature1', 'feature3']
                ]
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({
            'feature1': [2, 3, 4],
            'feature2': [5, 6, 7],
            'feature3': [10, 20, 30]
        })
        result = fe.apply_transformations(df)
        
        assert 'feature1_x_feature2' in result.columns
        assert 'feature1_x_feature3' in result.columns
        assert list(result['feature1_x_feature2']) == [10, 18, 28]
        assert list(result['feature1_x_feature3']) == [20, 60, 120]
    
    def test_interactions_missing_columns(self):
        """Test interaction features with missing columns"""
        config = {
            'enabled': True,
            'interactions': {
                'enabled': True,
                'feature_pairs': [
                    ['feature1', 'nonexistent']
                ]
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'feature1': [2, 3, 4]})
        result = fe.apply_transformations(df)
        
        # Interaction should not be created if a column is missing
        assert 'feature1_x_nonexistent' not in result.columns
    
    def test_polynomial_features(self):
        """Test polynomial features"""
        config = {
            'enabled': True,
            'polynomial': {
                'enabled': True,
                'degree': 3,
                'columns': ['value']
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'value': [2, 3, 4]})
        result = fe.apply_transformations(df)
        
        assert 'value_power_2' in result.columns
        assert 'value_power_3' in result.columns
        assert list(result['value_power_2']) == [4, 9, 16]
        assert list(result['value_power_3']) == [8, 27, 64]
    
    def test_custom_transformations(self):
        """Test custom transformations"""
        config = {
            'enabled': True,
            'custom': {
                'enabled': True,
                'transformations': [
                    {
                        'name': 'sum_feature',
                        'expression': "df['a'] + df['b']"
                    },
                    {
                        'name': 'product_feature',
                        'expression': "df['a'] * df['b']"
                    }
                ]
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = fe.apply_transformations(df)
        
        assert 'sum_feature' in result.columns
        assert 'product_feature' in result.columns
        assert list(result['sum_feature']) == [5, 7, 9]
        assert list(result['product_feature']) == [4, 10, 18]
    
    def test_custom_transformations_with_numpy(self):
        """Test custom transformations using numpy"""
        config = {
            'enabled': True,
            'custom': {
                'enabled': True,
                'transformations': [
                    {
                        'name': 'sqrt_feature',
                        'expression': "np.sqrt(df['value'])"
                    }
                ]
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'value': [4, 9, 16]})
        result = fe.apply_transformations(df)
        
        assert 'sqrt_feature' in result.columns
        assert list(result['sqrt_feature']) == [2.0, 3.0, 4.0]
    
    def test_custom_transformations_invalid_expression(self):
        """Test custom transformations with invalid expression"""
        config = {
            'enabled': True,
            'custom': {
                'enabled': True,
                'transformations': [
                    {
                        'name': 'invalid_feature',
                        'expression': "df['nonexistent_column']"
                    }
                ]
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({'value': [1, 2, 3]})
        result = fe.apply_transformations(df)
        
        # Should not create the feature if expression fails
        assert 'invalid_feature' not in result.columns
    
    def test_multiple_transformations(self):
        """Test applying multiple transformations together"""
        config = {
            'enabled': True,
            'binning': {
                'enabled': True,
                'features': [
                    {
                        'column': 'age',
                        'bins': [0, 30, 60, 100],
                        'labels': ['Young', 'Middle', 'Senior'],
                        'new_column': 'age_group'
                    }
                ]
            },
            'interactions': {
                'enabled': True,
                'feature_pairs': [['value1', 'value2']]
            },
            'custom': {
                'enabled': True,
                'transformations': [
                    {
                        'name': 'total',
                        'expression': "df['value1'] + df['value2']"
                    }
                ]
            }
        }
        fe = FeatureEngineer(config)
        df = pd.DataFrame({
            'age': [25, 45, 70],
            'value1': [10, 20, 30],
            'value2': [5, 10, 15]
        })
        result = fe.apply_transformations(df)
        
        assert 'age_group' in result.columns
        assert 'value1_x_value2' in result.columns
        assert 'total' in result.columns


class TestValidateData:
    """Test cases for validate_data function"""
    
    def test_validate_data_disabled(self):
        """Test that validation is skipped when disabled"""
        config = {'enabled': False}
        df = pd.DataFrame({'a': [1, 2, None], 'b': [4, 5, 6]})
        result = validate_data(df, config)
        pd.testing.assert_frame_equal(result, df)
    
    def test_check_missing_values(self):
        """Test missing values check"""
        config = {
            'enabled': True,
            'check_missing_values': True,
            'max_missing_percentage': 50
        }
        df = pd.DataFrame({
            'good_col': [1, 2, 3, 4, 5],
            'bad_col': [1, None, None, None, None]  # 80% missing
        })
        result = validate_data(df, config)
        
        assert 'good_col' in result.columns
        assert 'bad_col' not in result.columns
    
    def test_check_missing_values_threshold(self):
        """Test missing values with different threshold"""
        config = {
            'enabled': True,
            'check_missing_values': True,
            'max_missing_percentage': 30
        }
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [1, 2, None, None, 5]  # 40% missing
        })
        result = validate_data(df, config)
        
        assert 'col1' in result.columns
        assert 'col2' not in result.columns
    
    def test_check_duplicates(self):
        """Test duplicate rows check"""
        config = {
            'enabled': True,
            'check_duplicates': True
        }
        df = pd.DataFrame({
            'a': [1, 2, 3, 2],
            'b': [4, 5, 6, 5]
        })
        result = validate_data(df, config)
        
        # Should have 3 rows after removing 1 duplicate
        assert len(result) == 3
        assert not result.duplicated().any()
    
    def test_check_duplicates_disabled(self):
        """Test that duplicates are kept when check is disabled"""
        config = {
            'enabled': True,
            'check_duplicates': False
        }
        df = pd.DataFrame({
            'a': [1, 2, 2],
            'b': [4, 5, 5]
        })
        result = validate_data(df, config)
        
        # Should still have all rows including duplicates
        assert len(result) == 3
    
    def test_validation_combined(self):
        """Test combined validation checks"""
        config = {
            'enabled': True,
            'check_missing_values': True,
            'max_missing_percentage': 50,
            'check_duplicates': True
        }
        df = pd.DataFrame({
            'good_col': [1, 2, 3, 4, 2],
            'bad_col': [None, None, None, None, 5],  # 80% missing
            'other_col': [10, 20, 30, 40, 20]
        })
        result = validate_data(df, config)
        
        # bad_col should be removed (too many missing values)
        assert 'bad_col' not in result.columns
        # Duplicate row should be removed
        assert len(result) == 4
        # Other columns should remain
        assert 'good_col' in result.columns
        assert 'other_col' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
