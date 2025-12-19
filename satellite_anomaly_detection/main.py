import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')
class TelemetryGenerator:
    # Simulates satellite telemetry data with anomalies.
    def __init__(self, n_samples: int = 6000, noise_level: float = 0.5, random_seed: int = 42):
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.rng = np.random.RandomState(random_seed)
    # Generates the telemetry DataFrame.
    def generate(self) -> pd.DataFrame:
        time = np.arange(self.n_samples)
        # 1. Base Signal (Sinusoidal)
        signal = 20 + 10 * np.sin(time / 50)
        # 2. Sensor Noise
        noise = self.rng.normal(0, self.noise_level, self.n_samples)
        data = signal + noise
        labels = np.zeros(self.n_samples, dtype=int)
        # Inject Anomalies
        data[1000:1005] += 25  
        labels[1000:1005] = 1
        drift_start, drift_end = 3000, 3400 
        drift_slope = np.linspace(0, 15, drift_end - drift_start) 
        data[drift_start:drift_end] += drift_slope
        labels[drift_start:drift_end] = 1
        # Anomaly: System Freeze
        vib_start, vib_end = 5200, 5400
        data[vib_start:vib_end] += self.rng.normal(0, self.noise_level * 4, vib_end - vib_start)
        labels[vib_start:vib_end] = 1
        # Return DataFrame
        return pd.DataFrame({
            'timestamp': time,
            'sensor_reading': data,
            'is_anomaly': labels
        })
class TelemetryPreprocessor:
    # Processes telemetry data to engineer features for anomaly detection.
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.scaler = StandardScaler()
    # Transforms the DataFrame and returns features, labels, timestamps, and raw sensor values.
    def fit_transform(self, df: pd.DataFrame):
        data = df.copy()
        # 1. Trend (Rolling Mean) - Smoothes out the noise
        data['rolling_mean'] = data['sensor_reading'].rolling(window=self.window_size).mean()
        
        # 2. Volatility (Rolling Std) - Catches Freezes (low std) or Instability (high std)
        data['rolling_std'] = data['sensor_reading'].rolling(window=self.window_size).std()
        
        # 3. Velocity (Delta) - Catches Spikes
        data['delta'] = data['sensor_reading'].diff()
        
        # 4. Context (Lags) - Provides history to the model
        data['lag_1'] = data['sensor_reading'].shift(1)
        data['lag_20'] = data['sensor_reading'].shift(20)
        
        # Drop NaNs created by rolling windows (Clean start)
        data.dropna(inplace=True)
        
        # Split Data
        # Drop non-feature columns
        X = data.drop(['timestamp', 'is_anomaly', 'sensor_reading'], axis=1)
        y = data['is_anomaly']
        
        # Scale features (Critical for SVM and LOF)
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, data['timestamp'], data['sensor_reading']

class ModelEvaluator:
    # Compares multiple anomaly detection models and selects the best one.
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models = {
            
            "Isolation Forest": IsolationForest(
                n_estimators=500,  
                max_samples='auto',
                contamination=contamination, 
                random_state=42
            ),
            
          
            "One-Class SVM": OneClassSVM(
                nu=contamination, 
                kernel="rbf", 
                gamma='scale'
            ),
            
           
            "Local Outlier Factor": LocalOutlierFactor(
                n_neighbors=5, 
                contamination=contamination,
                novelty=False
            )
        }
    # Evaluates models and selects the best based on F1 Score.
    def evaluate(self, X, y_true):
        results = {}
        best_f1 = 0
        best_model_name = ""
        best_preds = []
        print(f"\n{'='*60}")
        print(f"{'ALGORITHM BENCHMARK RESULTS':^60}")
        print(f"{'='*60}")
        print(f"{'Model':<25} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10}")
        print(f"{'-'*60}")

        for name, model in self.models.items():
            # Training & Prediction
            if name == "Local Outlier Factor":
                y_pred_raw = model.fit_predict(X)
            else:
                model.fit(X)
                y_pred_raw = model.predict(X)
            # Convert predictions to binary format
            y_pred = np.where(y_pred_raw == -1, 1, 0)
            # Calculate Metrics
            f1 = f1_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            results[name] = f1
            print(f"🔹 {name:<24} | {f1:.4f}     | {prec:.4f}     | {rec:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_preds = y_pred
        # Final Summary
        print(f"{'-'*60}")
        print(f"BEST MODEL: {best_model_name.upper()} (F1: {best_f1:.4f})")
        print(f"{'='*60}\n")
        
        return best_model_name, best_preds

# Visualization Function
def plot_results(timestamps, sensor_values, y_true, y_pred, model_name):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(timestamps, sensor_values, color='#00f2ff', alpha=0.7, linewidth=1, label='Telemetry (Thermal)')
    ax.fill_between(timestamps, min(sensor_values), max(sensor_values), 
                    where=y_true==1, color='#00ff41', alpha=0.15, label='Actual Fault')
    anomaly_idx = np.where(y_pred == 1)[0]
    ax.scatter(timestamps.iloc[anomaly_idx], sensor_values.iloc[anomaly_idx], 
               color='#ff0055', s=20, label=f'AI Detection ({model_name})', zorder=5)
    ax.set_title(f"SATELLITE SUBSYSTEM HEALTH | ANOMALY DETECTION | MODEL: {model_name.upper()}", 
                 fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_ylabel("Temperature (°C)", color='#aaaaaa')
    ax.set_xlabel("Orbit Cycles (Time)", color='#aaaaaa')
    ax.grid(True, linestyle='--', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    leg = ax.legend(loc='upper right', frameon=True, facecolor='#111111', edgecolor='#333333')
    for text in leg.get_texts():
        text.set_color('white')
    plt.tight_layout()
    plt.show()

# Main Pipeline
def run_pipeline():
    print("INITIALIZING SATELLITE TELEMETRY SYSTEM...")
    # 1. Generate Data
    print("   [1/4] Generating Digital Twin data...")
    gen = TelemetryGenerator(n_samples=6000, noise_level=0.5)
    raw_df = gen.generate()
    # 2. Process Features
    print("   [2/4] Engineering physics-based features...")
    preprocessor = TelemetryPreprocessor(window_size=20)
    X, y_true, timestamps, sensor_vals = preprocessor.fit_transform(raw_df)
    # 3. Evaluate Models
    print("   [3/4] Running model tournament...")
    evaluator = ModelEvaluator(contamination=0.1) 
    best_model_name, y_pred = evaluator.evaluate(X, y_true)
    # 4. Visualize
    print(f"   [4/4] Launching dashboard for {best_model_name}...")
    plot_results(timestamps, sensor_vals, y_true, y_pred, best_model_name)

if __name__ == "__main__":
    run_pipeline()