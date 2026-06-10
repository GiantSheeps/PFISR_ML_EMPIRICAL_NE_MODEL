# models.py
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class SimpleTabularModel:
    def __init__(self, model_type='rf', **model_params):
        """Simple tabular model for electron density prediction."""
        self.model_type = model_type
        self.model_params = model_params
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.feature_names = None

    def prepare_data(self, df):
        """Isolate features and scale the dataset."""
        target_col = 'NE'
        feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols].values
        y = df[target_col].values.reshape(-1, 1)
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        self.feature_names = feature_cols
        return X_scaled, y_scaled.flatten()

    def fit(self, df, validation_split=0.2, random_state=42):
        X_scaled, y_scaled = self.prepare_data(df)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=validation_split,
            random_state=random_state, shuffle=True
        )
        
        if self.model_type == 'rf':
            params = {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5, 
                      'min_samples_leaf': 2, 'random_state': random_state}
            params.update(self.model_params)
            self.model = RandomForestRegressor(**params)
            
        elif self.model_type == 'gb':
            params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 
                      'random_state': random_state}
            params.update(self.model_params)
            self.model = GradientBoostingRegressor(**params)
            
        elif self.model_type == 'mlp':
            params = {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'solver': 'adam', 
                      'alpha': 0.001, 'batch_size': 32, 'learning_rate': 'adaptive', 
                      'max_iter': 1000, 'early_stopping': True, 'validation_fraction': 0.1, 
                      'n_iter_no_change': 50, 'random_state': random_state}
            params.update(self.model_params)
            self.model = MLPRegressor(**params)
            
        elif self.model_type == 'lr':
            params = {}
            params.update(self.model_params)
            self.model = LinearRegression(**params)
            
        else:
            raise ValueError("model_type must be 'rf', 'gb', 'mlp', or 'lr'")
        
        self.model.fit(X_train, y_train)
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        return train_score, val_score

    def predict(self, df):
        """Generate predictions on unseen data."""
        if self.model is None:
            raise ValueError("Model not trained.")
            
        feature_cols = [col for col in df.columns if col != 'NE']
        X = df[feature_cols].values
        
        if X.shape[1] != len(self.feature_names):
            X_processed = np.full((len(df), len(self.feature_names)), np.nan)
            for i, col in enumerate(self.feature_names):
                if col in df.columns:
                    X_processed[:, i] = df[col].values
            X = X_processed
            
        X_scaled = self.scaler_X.transform(X)
        predictions_scaled = self.model.predict(X_scaled)
        
        predictions_original = self.scaler_y.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        result_df = df.copy()
        result_df['NE_pred'] = predictions_original
        return result_df

# Sklearn randomized search distributions
rf_param_dist = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 5, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None, 0.2, 0.5],
    'bootstrap': [True, False]
}

gb_param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4, 6, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.5, 0.7, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None, 0.2, 0.5]
}

mlp_param_dist = {
    'hidden_layer_sizes': [(64,), (64,32), (128,64), (128,64,32), (256,128)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'lbfgs', 'sgd'],
    'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    'learning_rate_init': [1e-4, 1e-3, 1e-2],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'batch_size': [32, 64, 128, 'auto'],
    'early_stopping': [True, False],
    'max_iter': [200, 500, 1000, 2000]
}

ridge_param_grid = {
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    'fit_intercept': [True, False]
}

estimator_map = {
    'rf': (RandomForestRegressor(random_state=42), rf_param_dist),
    'gb': (GradientBoostingRegressor(random_state=42), gb_param_dist),
    'mlp': (MLPRegressor(random_state=42, max_iter=2000), mlp_param_dist),
    'lr': (Ridge(random_state=42), ridge_param_grid)
}
