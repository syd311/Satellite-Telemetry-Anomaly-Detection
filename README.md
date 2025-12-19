# Satellite-Telemetry-Anomaly-Detection

## Project Overview

This project is an end-to-end Machine Learning and Reliability Engineering pipeline designed to monitor the physical health of spacecraft subsystems.
The goal is to computationally detect early-onset anomalies in satellite thermal sensors before they lead to critical failure. By engineering a robust Digital Twin simulation, this project validates the ability of unsupervised learning to distinguish between harmless sensor noise and genuine physical degradation.


## Key Achievements:
Scale: Architected a Digital Twin generator capable of simulating thousands of orbital cycles with stochastic failure injection.

Precision: Achieved a 0.982 F1 Score in detecting system instability, successfully distinguishing high-variance vibration from Gaussian background noise.

Engineering: Implemented a modular Champion/Challenger evaluation system, proving Isolation Forest superiority over density-based methods (LOF) for high-frequency telemetry.


## How do we Monitor Spacecraft?
Distinguishing Signal from Noise is incredibly difficult.
### 1. The Concept: Orbital Thermal Cycling
A satellite in Low Earth Orbit (LEO) experiences extreme temperature swings every 90 minutes (Day/Night cycle).

$$T_{sensor} = T_{base} + A \cdot \sin(\omega t) + \epsilon$$

*   $T$: Temperature Reading.
*   $\sin(\omega t)$: The periodic heating from the Sun.
*   $\epsilon$: Sensor Noise (Gaussian).

### 2. The Data Engineering Challenge
In a vacuum, heat transfer is slow. Therefore, temperature data should always be "smooth" (high autocorrelation). However, raw telemetry is noisy due to radiation and sensor imperfections. A simple threshold alert (e.g., *"If Temp > 100"*) is useless because it triggers false alarms on noise spikes and misses subtle drifts that indicate localized failure.

### 3. The Solution: Physics-Based Feature Extraction
To solve for health, I implemented a Feature Engineering engine that converts raw time-series data into a physical context. We don't just look at the *value*; we look at the *behavior*.

*   **Volatility ($\sigma$):** If the standard deviation spikes, the system is vibrating, or the processor is unstable.
*   **Velocity ($\Delta$):** If the rate of change exceeds physical thermal limits, it is a voltage surge, not heat.

This pipeline filters for these violations using rolling window statistics:
$$Z_{score} = \frac{x - \mu_{rolling}}{\sigma_{rolling}}$$

---

### Results
After processing the telemetry data and running a benchmark tournament between three anomaly detection algorithms, the pipeline produced the following results:

| Metric | Value |
| :--- | :--- |
| **Target F1 Score (Production Requirement)** | **> 0.90** |
| **Calculated F1 Score (My Model)** | **0.982** |
| **Precision** | **0.99** |

---
