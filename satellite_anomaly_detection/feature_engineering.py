import pandas as pd
from sklearn.preprocessing import StandardScaler
class TelemetryPreprocessor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.scaler = StandardScaler()
    def fit_transform(self, df):
        data = df.copy()
        
        # 1. Smoothness (Trend)
        data['rolling_mean'] = data['sensor_reading'].rolling(window=self.window_size).mean()
        
        # 2. Volatility (To detect noise level changes or freezes)
        data['rolling_std'] = data['sensor_reading'].rolling(window=self.window_size).std()
        
        # 3. Velocity (Delta between time steps)
        data['delta'] = data['sensor_reading'].diff()
        
        # 4. Lag Features (Context)
        data['lag_1'] = data['sensor_reading'].shift(1)
        data['lag_5'] = data['sensor_reading'].shift(5)
        
        # Drop NaNs created by rolling/shifting
        data.dropna(inplace=True)
   
        y = data['is_anomaly']
        X = data.drop(['timestamp', 'is_anomaly'], axis=1)
        
        # Scale features (Critical for Anomaly Detection models)
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, data['timestamp']