from sklearn.ensemble import RandomForestClassifier
import numpy as np

class ToxicRandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,  # Use all available cores
            class_weight='balanced'  # Handle class imbalance
        )
    
    def train(self, X_train, y_train):
        """Train the Random Forest model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importances(self):
        """Get feature importances"""
        return self.model.feature_importances_ 