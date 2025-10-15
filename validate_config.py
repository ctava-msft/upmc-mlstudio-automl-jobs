"""
Configuration Validator
Validates config.yaml files before training to catch common errors
"""

import yaml
import argparse
import sys
from pathlib import Path
from typing import List, Tuple


class ConfigValidator:
    """Validates training configuration files"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.errors = []
        self.warnings = []
        self.config = None
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate the configuration file
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        # Check file exists
        if not Path(self.config_path).exists():
            self.errors.append(f"Configuration file not found: {self.config_path}")
            return False, self.errors, self.warnings
        
        # Load YAML
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML syntax: {str(e)}")
            return False, self.errors, self.warnings
        
        # Run validation checks
        self._validate_structure()
        self._validate_experiment()
        self._validate_data()
        self._validate_feature_engineering()
        self._validate_compute()
        self._validate_automl()
        self._validate_output()
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_structure(self):
        """Check required top-level sections"""
        required_sections = ['experiment', 'data', 'compute', 'automl']
        for section in required_sections:
            if section not in self.config:
                self.errors.append(f"Missing required section: '{section}'")
    
    def _validate_experiment(self):
        """Validate experiment configuration"""
        if 'experiment' not in self.config:
            return
        
        exp = self.config['experiment']
        if 'name' not in exp:
            self.errors.append("experiment.name is required")
        elif not isinstance(exp['name'], str):
            self.errors.append("experiment.name must be a string")
    
    def _validate_data(self):
        """Validate data configuration"""
        if 'data' not in self.config:
            return
        
        data = self.config['data']
        
        # Required fields
        if 'input_path' not in data:
            self.errors.append("data.input_path is required")
        else:
            input_path = Path(data['input_path'])
            if not input_path.exists():
                self.warnings.append(f"Data file not found: {data['input_path']}")
            
            # Check file extension
            valid_extensions = ['.csv', '.parquet', '.xlsx', '.xls']
            if input_path.suffix not in valid_extensions:
                self.warnings.append(
                    f"Unusual file extension: {input_path.suffix}. "
                    f"Supported: {', '.join(valid_extensions)}"
                )
        
        if 'dataset_name' not in data:
            self.errors.append("data.dataset_name is required")
        
        if 'label_column' not in data:
            self.errors.append("data.label_column is required")
        elif not isinstance(data['label_column'], str):
            self.errors.append("data.label_column must be a string")
    
    def _validate_feature_engineering(self):
        """Validate feature engineering configuration"""
        if 'feature_engineering' not in self.config:
            return
        
        fe = self.config['feature_engineering']
        
        # Validate binning
        if fe.get('binning', {}).get('enabled', False):
            features = fe['binning'].get('features', [])
            for i, feature in enumerate(features):
                if 'column' not in feature:
                    self.errors.append(f"binning.features[{i}]: 'column' is required")
                if 'bins' not in feature:
                    self.errors.append(f"binning.features[{i}]: 'bins' is required")
                if 'labels' not in feature:
                    self.errors.append(f"binning.features[{i}]: 'labels' is required")
                elif 'bins' in feature:
                    if len(feature['labels']) != len(feature['bins']) - 1:
                        self.errors.append(
                            f"binning.features[{i}]: number of labels must be "
                            f"len(bins) - 1"
                        )
        
        # Validate datetime features
        if fe.get('datetime_features', {}).get('enabled', False):
            columns = fe['datetime_features'].get('columns', [])
            valid_extracts = ['year', 'month', 'day', 'dayofweek', 'quarter', 'hour']
            for i, col_config in enumerate(columns):
                if 'column' not in col_config:
                    self.errors.append(
                        f"datetime_features.columns[{i}]: 'column' is required"
                    )
                if 'extract' not in col_config:
                    self.errors.append(
                        f"datetime_features.columns[{i}]: 'extract' is required"
                    )
                else:
                    for extract in col_config['extract']:
                        if extract not in valid_extracts:
                            self.warnings.append(
                                f"datetime_features.columns[{i}]: unknown extract "
                                f"type '{extract}'. Valid: {', '.join(valid_extracts)}"
                            )
        
        # Validate interactions
        if fe.get('interactions', {}).get('enabled', False):
            pairs = fe['interactions'].get('feature_pairs', [])
            for i, pair in enumerate(pairs):
                if not isinstance(pair, list) or len(pair) != 2:
                    self.errors.append(
                        f"interactions.feature_pairs[{i}]: must be a list of 2 items"
                    )
        
        # Validate custom transformations
        if fe.get('custom', {}).get('enabled', False):
            transforms = fe['custom'].get('transformations', [])
            for i, transform in enumerate(transforms):
                if 'name' not in transform:
                    self.errors.append(
                        f"custom.transformations[{i}]: 'name' is required"
                    )
                if 'expression' not in transform:
                    self.errors.append(
                        f"custom.transformations[{i}]: 'expression' is required"
                    )
    
    def _validate_compute(self):
        """Validate compute configuration"""
        if 'compute' not in self.config:
            return
        
        compute = self.config['compute']
        if 'cluster_name' not in compute:
            self.errors.append("compute.cluster_name is required")
    
    def _validate_automl(self):
        """Validate AutoML configuration"""
        if 'automl' not in self.config:
            return
        
        automl = self.config['automl']
        
        # Validate task
        valid_tasks = ['regression', 'classification', 'forecasting']
        if 'task' not in automl:
            self.errors.append("automl.task is required")
        elif automl['task'] not in valid_tasks:
            self.errors.append(
                f"automl.task must be one of: {', '.join(valid_tasks)}"
            )
        
        # Validate primary metric
        if 'primary_metric' not in automl:
            self.errors.append("automl.primary_metric is required")
        
        # Validate training section
        if 'training' not in automl:
            self.errors.append("automl.training section is required")
        else:
            training = automl['training']
            if 'experiment_timeout_minutes' not in training:
                self.warnings.append(
                    "automl.training.experiment_timeout_minutes not set"
                )
            
            # Check for cross-validation or validation size
            if 'n_cross_validations' not in training and 'validation_size' not in training:
                self.warnings.append(
                    "Neither n_cross_validations nor validation_size is set. "
                    "Recommend setting one."
                )
        
        # Validate featurization
        if 'featurization' in automl:
            feat = automl['featurization']
            if 'mode' in feat:
                valid_modes = ['auto', 'custom', 'off']
                if feat['mode'] not in valid_modes:
                    self.errors.append(
                        f"automl.featurization.mode must be one of: "
                        f"{', '.join(valid_modes)}"
                    )
    
    def _validate_output(self):
        """Validate output configuration"""
        if 'output' not in self.config:
            self.warnings.append("output section not configured, using defaults")
            return
        
        output = self.config['output']
        
        if output.get('log_level'):
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
            if output['log_level'] not in valid_levels:
                self.errors.append(
                    f"output.log_level must be one of: {', '.join(valid_levels)}"
                )


def main():
    parser = argparse.ArgumentParser(
        description='Validate AutoML training configuration file'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Configuration Validator")
    print("="*60)
    print(f"Validating: {args.config}")
    print()
    
    validator = ConfigValidator(args.config)
    is_valid, errors, warnings = validator.validate()
    
    # Print warnings
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    # Print errors
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print()
        print("Configuration is INVALID. Please fix the errors above.")
        sys.exit(1)
    
    # Success
    print("✅ Configuration is VALID!")
    print()
    print("Summary:")
    print(f"  Experiment: {validator.config['experiment']['name']}")
    print(f"  Task: {validator.config['automl']['task']}")
    print(f"  Data: {validator.config['data']['input_path']}")
    print(f"  Target: {validator.config['data']['label_column']}")
    print()
    print("You can now run: python train.py --config", args.config)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
