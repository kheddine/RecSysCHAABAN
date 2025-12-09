"""
EDA & Data Cleaning Module
Wrapper around the user's original data cleaning code
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

class DataCleaner:
    """Encapsulates EDA and cleaning logic from original code"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.df_clean = None
        self.feature_cols = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]
        self.eda_results = {}
        self.validation_results = {}
    
    def explore(self) -> dict:
        """Comprehensive data exploration"""
        results = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': {col: str(self.df[col].dtype) for col in self.df.columns},
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': {
                'total': int(self.df.duplicated().sum()),
                'by_track_name': int(self.df.duplicated(subset=['track_name']).sum()) if 'track_name' in self.df.columns else 0
            },
            'summary_stats': self.df.describe().to_dict()
        }
        self.eda_results = results
        return results
    
    def validate_audio_features(self) -> Tuple[bool, List[str]]:
        """Check for required audio feature columns"""
        required_features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'key', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]
        
        found_features = [f for f in required_features if f in self.df.columns]
        missing_features = [f for f in required_features if f not in self.df.columns]
        
        self.validation_results['audio_features'] = {
            'found': len(found_features),
            'required': len(required_features),
            'features': found_features,
            'missing': missing_features
        }
        
        return len(missing_features) == 0, found_features
    
    def check_feature_ranges(self, features: List[str]) -> dict:
        """Validate feature value ranges"""
        expected_ranges = {
            'acousticness': (0, 1),
            'danceability': (0, 1),
            'energy': (0, 1),
            'instrumentalness': (0, 1),
            'key': (0, 11),
            'liveness': (0, 1),
            'loudness': (-60, 0),
            'speechiness': (0, 1),
            'tempo': (0, 300),
            'valence': (0, 1)
        }
        
        issues = {}
        for feature in features:
            if feature not in self.df.columns:
                continue
            
            if self.df[feature].dtype == 'object':
                issues[feature] = {
                    'type': 'categorical',
                    'unique_values': self.df[feature].nunique()
                }
                continue
            
            min_val = float(self.df[feature].min())
            max_val = float(self.df[feature].max())
            
            if feature in expected_ranges:
                expected_min, expected_max = expected_ranges[feature]
                out_of_range = ((self.df[feature] < expected_min) | (self.df[feature] > expected_max)).sum()
                
                issues[feature] = {
                    'in_range': out_of_range == 0,
                    'expected_range': [expected_min, expected_max],
                    'actual_range': [min_val, max_val],
                    'out_of_range_count': int(out_of_range)
                }
            else:
                issues[feature] = {
                    'actual_range': [min_val, max_val]
                }
        
        self.validation_results['feature_ranges'] = issues
        return issues
    
    def clean(self) -> pd.DataFrame:
        """Clean and prepare data"""
        self.df_clean = self.df.copy()
        original_len = len(self.df_clean)
        
        # Find available features
        has_all, features = self.validate_audio_features()
        if not features:
            features = self.feature_cols
        
        # 1. Remove complete duplicates
        self.df_clean = self.df_clean.drop_duplicates()
        removed_dups = original_len - len(self.df_clean)
        
        # 2. Remove rows with missing audio features
        before_missing = len(self.df_clean)
        self.df_clean = self.df_clean.dropna(subset=[f for f in features if f in self.df_clean.columns])
        removed_missing = before_missing - len(self.df_clean)
        
        # 3. Fix out-of-range values
        # For 0-1 features, clip to [0, 1]
        features_0_1 = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                        'liveness', 'speechiness', 'valence']
        for feat in features_0_1:
            if feat in self.df_clean.columns:
                self.df_clean[feat] = self.df_clean[feat].clip(0, 1)
        
        # Tempo: clip to reasonable range
        if 'tempo' in self.df_clean.columns:
            self.df_clean['tempo'] = self.df_clean['tempo'].clip(0, 300)
        
        # 4. Remove track name duplicates
        name_col = None
        if 'track_name' in self.df_clean.columns:
            name_col = 'track_name'
        elif 'name' in self.df_clean.columns:
            name_col = 'name'
        
        removed_track_dups = 0
        if name_col:
            before = len(self.df_clean)
            self.df_clean = self.df_clean.drop_duplicates(subset=[name_col], keep='first')
            removed_track_dups = before - len(self.df_clean)
        
        # Store cleaning results
        self.validation_results['cleaning'] = {
            'original_rows': original_len,
            'final_rows': len(self.df_clean),
            'removed_duplicates': int(removed_dups),
            'removed_missing': int(removed_missing),
            'removed_track_duplicates': int(removed_track_dups),
            'total_removed': original_len - len(self.df_clean),
            'removal_percentage': round(100 * (original_len - len(self.df_clean)) / original_len, 2)
        }
        
        return self.df_clean
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics"""
        self.explore()
        self.validate_audio_features()
        
        return {
            'eda': self.eda_results,
            'validation': self.validation_results
        }
