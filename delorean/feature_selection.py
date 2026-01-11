from typing import List
import pandas as pd
from qlib.data.dataset import DatasetH

class FeatureSelector:
    """
    Handles feature selection and filtering, such as correlation-based removal.
    """
    @staticmethod
    def filter_by_correlation(dataset: DatasetH, candidate_features: List[str], threshold: float = 0.95) -> List[str]:
        """
        Filters out highly correlated features from the candidate list.
        Features are assumed to be sorted by importance (descending).
        If two features are correlated > threshold, the one appearing later in the list (less important) is dropped.

        Args:
            dataset (DatasetH): Qlib dataset containing the features.
            candidate_features (List[str]): List of feature names sorted by importance.
            threshold (float): Correlation threshold to drop features.

        Returns:
            List[str]: Filtered list of feature names.
        """
        print(f"\n[Feature Selection] Filtering {len(candidate_features)} features by correlation (Threshold: {threshold})...")
        
        # 1. Prepare Data for Candidates
        # We need to load these features to compute correlation
        try:
            df_train = dataset.prepare("train", col_set="feature")
            # Select only candidate columns. 
            # Note: Dataset might have more columns than candidates if we loaded all 158 but only checking top N.
            # Convert candidates to flattened list if needed, but usually they match column names.
            df_selected = df_train[candidate_features]
        except KeyError as e:
            print(f"Error selecting features from dataset: {e}")
            return candidate_features

        # 2. Compute Correlation Matrix
        corr_matrix = df_selected.corr().abs()
        
        # 3. Filter Logic
        dropped_features = set()
        final_features = []
        
        for i, f1 in enumerate(candidate_features):
            if f1 in dropped_features:
                continue
            final_features.append(f1)
            
            # Compare with all subsequent (less important) features
            for f2 in candidate_features[i+1:]:
                if f2 in dropped_features:
                    continue
                
                # Check correlation
                # If correlation > threshold, drop f2
                if corr_matrix.loc[f1, f2] > threshold:
                    dropped_features.add(f2)
                    # Optional: detailed logging
                    # print(f"  Dropping {f2} (Corr with {f1}: {corr_matrix.loc[f1, f2]:.2f})")
        
        print(f"Dropped {len(dropped_features)} features. Keeping {len(final_features)} features.")
        return final_features
