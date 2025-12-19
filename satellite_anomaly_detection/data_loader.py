import numpy as np
import pandas as pd

class TelemetryGenerator:
    """
    Simulates satellite telemetry data (Thermal & Power subsystems).
    Includes logic to inject 'noise' vs 'degradation'.
    """
    def __init__(self, n_samples=5000, noise_level=0.5):
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.rng = np.random.RandomState(42)

    def generate(self):
        time = np.arange(self.n_samples)
        
        # 1. Base Signal (Simulating Orbital Thermal Cycles)
        # Orbit period approx 100 samples
        signal = 20 + 10 * np.sin(time / 50) 
        
        # 2. Sensor Noise (High frequency, low amplitude)
        noise = self.rng.normal(0, self.noise_level, self.n_samples)
        
        # 3. Inject Anomalies
        data = signal + noise
        labels = np.zeros(self.n_samples)
        
        # Anomaly A: Power Spike (Transient)
        data[1000:1005] += 15
        labels[1000:1005] = 1
        
        # Anomaly B: Sensor Degradation (Drift/Bias)
        # Linear drift added to the sine wave
        drift_start = 3000
        data[drift_start:] += np.linspace(0, 10, self.n_samples - drift_start)
        labels[drift_start:] = 1 # Mark as anomalous behavior
        
        # Anomaly C: System Freeze (Variance collapse)
        data[4200:4400] = 25  # Value gets stuck
        labels[4200:4400] = 1

        df = pd.DataFrame({
            'timestamp': time,
            'sensor_reading': data,
            'is_anomaly': labels
        })
        return df