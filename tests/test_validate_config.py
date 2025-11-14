"""
Unit tests for validate_config.py
"""

import pytest
import yaml
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import os

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from validate_config import ConfigValidator


class TestConfigValidator:
    """Test cases for ConfigValidator class"""
    
    def test_initialization(self):
        """Test ConfigValidator initialization"""
        validator = ConfigValidator('test_config.yaml')
        assert validator.config_path == 'test_config.yaml'
        assert validator.errors == []
        assert validator.warnings == []
        assert validator.config is None
    
    def test_file_not_found(self):
        """Test validation with non-existent file"""
        validator = ConfigValidator('nonexistent_file.yaml')
        is_valid, errors, warnings = validator.validate()
        
        assert not is_valid
        assert len(errors) > 0
        assert 'not found' in errors[0].lower()
    
    def test_invalid_yaml_syntax(self):
        """Test validation with invalid YAML syntax"""
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: syntax:')
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert len(errors) > 0
            assert 'yaml' in errors[0].lower()
        finally:
            os.unlink(temp_file)
    
    def test_valid_minimal_config(self):
        """Test validation with minimal valid configuration"""
        config = {
            'experiment': {
                'name': 'test-experiment',
                'description': 'Test experiment'
            },
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test_dataset',
                'label_column': 'target'
            },
            'compute': {
                'cluster_name': 'test-cluster'
            },
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {
                    'experiment_timeout_minutes': 60
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            # May have warnings (e.g., file not found) but should be valid structure
            assert len(errors) == 0
        finally:
            os.unlink(temp_file)
    
    def test_missing_required_sections(self):
        """Test validation with missing required sections"""
        config = {
            'experiment': {
                'name': 'test'
            }
            # Missing data, compute, automl sections
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('data' in error.lower() for error in errors)
            assert any('compute' in error.lower() for error in errors)
            assert any('automl' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_experiment_validation_missing_name(self):
        """Test experiment validation with missing name"""
        config = {
            'experiment': {
                'description': 'Test'
            },
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {
                'cluster_name': 'test'
            },
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('experiment.name' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_experiment_validation_invalid_name_type(self):
        """Test experiment validation with invalid name type"""
        config = {
            'experiment': {
                'name': 123  # Should be string
            },
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {
                'cluster_name': 'test'
            },
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('string' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_data_validation_missing_fields(self):
        """Test data validation with missing required fields"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                # Missing input_path, dataset_name, label_column
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('input_path' in error.lower() for error in errors)
            assert any('dataset_name' in error.lower() for error in errors)
            assert any('label_column' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_data_validation_file_exists(self):
        """Test data validation with existing file"""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / 'test_data.csv'
            data_file.write_text('col1,col2\n1,2\n3,4\n')
            
            config = {
                'experiment': {'name': 'test'},
                'data': {
                    'input_path': str(data_file),
                    'dataset_name': 'test',
                    'label_column': 'target'
                },
                'compute': {'cluster_name': 'test'},
                'automl': {
                    'task': 'regression',
                    'primary_metric': 'r2_score',
                    'training': {}
                }
            }
            
            with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                temp_file = f.name
            
            try:
                validator = ConfigValidator(temp_file)
                is_valid, errors, warnings = validator.validate()
                
                # File exists, so no warning about missing file
                assert not any('not found' in warning.lower() for warning in warnings)
            finally:
                os.unlink(temp_file)
    
    def test_data_validation_invalid_file_extension(self):
        """Test data validation with unusual file extension"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'test.txt',  # Unusual extension
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            # Should have warning about unusual extension
            assert any('extension' in warning.lower() for warning in warnings)
        finally:
            os.unlink(temp_file)
    
    def test_binning_validation(self):
        """Test feature engineering binning validation"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            },
            'feature_engineering': {
                'binning': {
                    'enabled': True,
                    'features': [
                        {
                            'column': 'age',
                            'bins': [0, 18, 35, 50, 100],
                            'labels': ['Child', 'Young', 'Adult', 'Senior']
                        }
                    ]
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            # Should be valid - correct number of labels for bins
            assert len([e for e in errors if 'binning' in e.lower()]) == 0
        finally:
            os.unlink(temp_file)
    
    def test_binning_validation_invalid_labels(self):
        """Test binning validation with wrong number of labels"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            },
            'feature_engineering': {
                'binning': {
                    'enabled': True,
                    'features': [
                        {
                            'column': 'age',
                            'bins': [0, 18, 35, 50, 100],  # 5 bins = 4 intervals
                            'labels': ['Child', 'Young']  # Only 2 labels
                        }
                    ]
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('labels' in error.lower() and 'bins' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_binning_validation_missing_fields(self):
        """Test binning validation with missing required fields"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            },
            'feature_engineering': {
                'binning': {
                    'enabled': True,
                    'features': [
                        {
                            # Missing column, bins, labels
                        }
                    ]
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('column' in error.lower() and 'required' in error.lower() for error in errors)
            assert any('bins' in error.lower() and 'required' in error.lower() for error in errors)
            assert any('labels' in error.lower() and 'required' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_datetime_features_validation(self):
        """Test datetime features validation"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            },
            'feature_engineering': {
                'datetime_features': {
                    'enabled': True,
                    'columns': [
                        {
                            'column': 'date',
                            'extract': ['year', 'month', 'day']
                        }
                    ]
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            # Valid datetime configuration
            assert len([e for e in errors if 'datetime' in e.lower()]) == 0
        finally:
            os.unlink(temp_file)
    
    def test_datetime_features_validation_invalid_extract(self):
        """Test datetime features validation with invalid extract type"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            },
            'feature_engineering': {
                'datetime_features': {
                    'enabled': True,
                    'columns': [
                        {
                            'column': 'date',
                            'extract': ['year', 'invalid_extract_type']
                        }
                    ]
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            # Should have warning about unknown extract type
            assert any('unknown extract' in warning.lower() for warning in warnings)
        finally:
            os.unlink(temp_file)
    
    def test_interactions_validation(self):
        """Test interactions validation"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            },
            'feature_engineering': {
                'interactions': {
                    'enabled': True,
                    'feature_pairs': [
                        ['feature1', 'feature2'],
                        ['feature3', 'feature4']
                    ]
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            # Valid interactions configuration
            assert len([e for e in errors if 'interaction' in e.lower()]) == 0
        finally:
            os.unlink(temp_file)
    
    def test_interactions_validation_invalid_pairs(self):
        """Test interactions validation with invalid pairs"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            },
            'feature_engineering': {
                'interactions': {
                    'enabled': True,
                    'feature_pairs': [
                        ['feature1'],  # Only 1 item, should be 2
                        ['feature2', 'feature3', 'feature4']  # 3 items, should be 2
                    ]
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('list of 2 items' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_custom_transformations_validation(self):
        """Test custom transformations validation"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            },
            'feature_engineering': {
                'custom': {
                    'enabled': True,
                    'transformations': [
                        {
                            'name': 'bmi',
                            'expression': "df['weight'] / (df['height'] ** 2)"
                        }
                    ]
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            # Valid custom transformations
            assert len([e for e in errors if 'custom' in e.lower()]) == 0
        finally:
            os.unlink(temp_file)
    
    def test_custom_transformations_missing_fields(self):
        """Test custom transformations with missing fields"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            },
            'feature_engineering': {
                'custom': {
                    'enabled': True,
                    'transformations': [
                        {
                            'name': 'feature1'
                            # Missing expression
                        },
                        {
                            'expression': "df['col1'] * 2"
                            # Missing name
                        }
                    ]
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('expression' in error.lower() and 'required' in error.lower() for error in errors)
            assert any('name' in error.lower() and 'required' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_compute_validation(self):
        """Test compute configuration validation"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {
                'cluster_name': 'my-cluster'
            },
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            # Valid compute configuration
            assert not any('compute.cluster_name' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_compute_validation_missing_cluster(self):
        """Test compute validation with missing cluster name"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {
                # Missing cluster_name
            },
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('cluster_name' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_automl_validation_invalid_task(self):
        """Test automl validation with invalid task type"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'invalid_task',  # Invalid task type
                'primary_metric': 'r2_score',
                'training': {}
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('task' in error.lower() and 'regression' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_automl_validation_missing_primary_metric(self):
        """Test automl validation with missing primary metric"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                # Missing primary_metric
                'training': {}
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('primary_metric' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_automl_validation_missing_training_section(self):
        """Test automl validation with missing training section"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score'
                # Missing training section
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('training' in error.lower() and 'required' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_automl_featurization_validation(self):
        """Test automl featurization mode validation"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {},
                'featurization': {
                    'mode': 'auto'
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            # Valid featurization mode
            assert not any('featurization' in error.lower() and 'mode' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_automl_featurization_invalid_mode(self):
        """Test automl featurization with invalid mode"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {},
                'featurization': {
                    'mode': 'invalid_mode'
                }
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('featurization' in error.lower() and 'mode' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_output_validation_invalid_log_level(self):
        """Test output validation with invalid log level"""
        config = {
            'experiment': {'name': 'test'},
            'data': {
                'input_path': 'dummy.csv',
                'dataset_name': 'test',
                'label_column': 'target'
            },
            'compute': {'cluster_name': 'test'},
            'automl': {
                'task': 'regression',
                'primary_metric': 'r2_score',
                'training': {}
            },
            'output': {
                'log_level': 'INVALID_LEVEL'
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_file = f.name
        
        try:
            validator = ConfigValidator(temp_file)
            is_valid, errors, warnings = validator.validate()
            
            assert not is_valid
            assert any('log_level' in error.lower() for error in errors)
        finally:
            os.unlink(temp_file)
    
    def test_output_validation_valid_log_level(self):
        """Test output validation with valid log level"""
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            config = {
                'experiment': {'name': 'test'},
                'data': {
                    'input_path': 'dummy.csv',
                    'dataset_name': 'test',
                    'label_column': 'target'
                },
                'compute': {'cluster_name': 'test'},
                'automl': {
                    'task': 'regression',
                    'primary_metric': 'r2_score',
                    'training': {}
                },
                'output': {
                    'log_level': level
                }
            }
            
            with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                temp_file = f.name
            
            try:
                validator = ConfigValidator(temp_file)
                is_valid, errors, warnings = validator.validate()
                
                # Should not have errors about log level
                assert not any('log_level' in error.lower() for error in errors)
            finally:
                os.unlink(temp_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
