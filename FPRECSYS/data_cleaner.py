import pandas as pd
import numpy as np
from typing import Tuple, List

class DataCleaner:
    """Clean and validate Spotify data"""
    
    def __init__(self):
        self.required_features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'key', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]
    
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check if dataset has required columns"""
        missing = []
        for col in self.required_features:
            if col not in df.columns:
                missing.append(col)
        
        return len(missing) == 0, missing
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of dataset"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'dtypes': dict(df.dtypes),
            'missing_values': dict(df.isnull().sum()),
            'duplicates': df.duplicated().sum()
        }
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Clean and prepare data for recommendations"""
        df_clean = df.copy()
        cleaning_log = {
            'original_rows': len(df),
            'removed_duplicates': 0,
            'removed_missing': 0,
            'removed_outliers': 0,
            'final_rows': 0,
            'removed_duplicate_names': 0
        }
        
        # 1. Remove complete duplicates
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        cleaning_log['removed_duplicates'] = before - len(df_clean)
        
        # 2. Remove rows with missing audio features
        numeric_features = [f for f in self.required_features if f in df_clean.columns and f != 'key']
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=numeric_features)
        cleaning_log['removed_missing'] = before - len(df_clean)
        
        # 3. Handle outliers and invalid values
        # Clip 0-1 features
        features_0_1 = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                        'liveness', 'speechiness', 'valence']
        for feat in features_0_1:
            if feat in df_clean.columns:
                before_invalid = ((df_clean[feat] < 0) | (df_clean[feat] > 1)).sum()
                df_clean[feat] = df_clean[feat].clip(0, 1)
                cleaning_log['removed_outliers'] += before_invalid
        
        # Clip tempo
        if 'tempo' in df_clean.columns:
            before_invalid = ((df_clean['tempo'] < 0) | (df_clean['tempo'] > 300)).sum()
            df_clean['tempo'] = df_clean['tempo'].clip(0, 300)
            cleaning_log['removed_outliers'] += before_invalid
        
        # 4. Remove duplicate track names (keep first)
        if 'track_name' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=['track_name'], keep='first')
            cleaning_log['removed_duplicate_names'] = before - len(df_clean)
        
        cleaning_log['final_rows'] = len(df_clean)
        
        return df_clean, cleaning_log
    
    def get_feature_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics for each audio feature"""
        stats = {}
        numeric_features = [f for f in self.required_features if f in df.columns and f != 'key']
        
        for feature in numeric_features:
            if feature in df.columns:
                data = df[feature].dropna()
                stats[feature] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    '25%': float(data.quantile(0.25)),
                    '50%': float(data.quantile(0.50)),
                    '75%': float(data.quantile(0.75))
                }
        
        return stats
    
    def detect_issues(self, df: pd.DataFrame) -> List[str]:
        """Detect potential data quality issues"""
        issues = []
        
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100)
        for col, pct in missing_pct[missing_pct > 5].items():
            issues.append(f"⚠️  Column '{col}' has {pct:.1f}% missing values")
        
        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append(f"⚠️  {dup_count} duplicate rows found")
        
        # Check for outliers in features
        numeric_features = [f for f in self.required_features if f in df.columns and f != 'key']
        for feat in numeric_features:
            if feat in df.columns:
                data = df[feat]
                # Check for values outside expected ranges
                if feat in ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                           'liveness', 'speechiness', 'valence']:
                    invalid = ((data < 0) | (data > 1)).sum()
                    if invalid > 0:
                        issues.append(f"⚠️  {feat} has {invalid} values outside [0, 1]")
                
                elif feat == 'tempo':
                    invalid = ((data < 0) | (data > 300)).sum()
                    if invalid > 0:
                        issues.append(f"⚠️  tempo has {invalid} values outside [0, 300]")
        
        return issues if issues else ["✅ No major issues detected"]
