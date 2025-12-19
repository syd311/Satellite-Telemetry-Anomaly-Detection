import matplotlib.pyplot as plt
import seaborn as sns

def plot_telemetry(timestamps, sensor_values, labels, preds):
    plt.figure(figsize=(15, 6))
    plt.style.use('bmh') # Professional scientific style
    
    # Plot raw signal
    plt.plot(timestamps, sensor_values, label='Sensor Telemetry (Thermal)', color='#2c3e50', alpha=0.6)
    
    # Highlight True Anomalies (Green shading)
    # Highlight Predicted Anomalies (Red points)
    
    # Find indices where prediction is anomaly
    anomaly_indices = [i for i, x in enumerate(preds) if x == 1]
    plt.scatter(timestamps.iloc[anomaly_indices], 
                sensor_values.iloc[anomaly_indices], 
                color='#e74c3c', s=20, label='AI Detected Anomaly')

    plt.title('Satellite Subsystem Health Monitoring | Isolation Forest Detection', fontsize=14)
    plt.xlabel('Time (Orbit Cycles)')
    plt.ylabel('Temperature (Celsius)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()