# 🌱 Smart Greenhouse System

**Advanced Greenhouse Automation with Machine Learning**
Compatible with Windows 11, Windows 10, Linux, and macOS

## 🚀 Overview

A comprehensive smart greenhouse management system featuring:

- **Real-time Environmental Monitoring** - Temperature, humidity, soil moisture, light, and CO2 tracking
- **Automated Lighting Control** - Intelligent grow light management with scheduling and intensity control
- **Smart Irrigation System** - Moisture-based automatic watering with multiple zones
- **Machine Learning Predictions** - Forecast environmental conditions and optimize plant growth
- **Interactive Web Dashboard** - Real-time visualization and control
- **Windows 11 Native Alerts** - Desktop notifications for critical events
- **Growth Analytics** - Track plant health and predict harvest yields

---

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Dashboard](#dashboard)
- [Machine Learning](#machine-learning)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

### 1. **Environmental Monitoring**

- **Multi-Sensor Support**
  - Temperature sensors (°C/°F)
  - Humidity sensors (%)
  - Soil moisture sensors (%)
  - Light intensity sensors (lux)
  - CO2 concentration sensors (ppm)

- **Data Logging**
  - Continuous monitoring at configurable intervals
  - Historical data storage in JSON format
  - Statistical analysis (min, max, average)
  - Configurable retention periods

### 2. **Lighting Control**

- **Multiple Control Modes**
  - Manual on/off control
  - Scheduled operation (photoperiod)
  - Auto-adjust based on ambient light
  - Intensity control (0-100%)

- **Smart Features**
  - Zone-based control
  - Energy-saving mode
  - DLI (Daily Light Integral) optimization
  - Power consumption tracking

### 3. **Irrigation Management**

- **Automated Watering**
  - Soil moisture-based triggers
  - Multiple watering zones
  - Scheduled irrigation
  - Flow rate monitoring

- **Safety Features**
  - Maximum duration limits
  - Daily water usage caps
  - Emergency stop function
  - Water consumption tracking

### 4. **Machine Learning**

- **Condition Prediction**
  - Forecast temperature, humidity, moisture, and light levels
  - Random Forest and Gradient Boosting models
  - Multi-hour ahead predictions
  - Model retraining on new data

- **Growth Optimization**
  - Growth score calculation (0-100)
  - Yield prediction
  - Limiting factor identification
  - Recommendation engine

- **Resource Optimization**
  - Energy efficiency recommendations
  - Water usage optimization
  - Multi-objective optimization

### 5. **Web Dashboard**

- **Real-time Visualization**
  - Live sensor readings
  - Historical trend charts
  - Actuator status display
  - Alert notifications

- **Interactive Controls**
  - Lighting on/off buttons
  - Emergency stop controls
  - Growth metrics display
  - Auto-refresh (5 seconds)

### 6. **Alert System**

- **Windows 11 Integration**
  - Native toast notifications
  - Customizable alert levels
  - Sound notifications
  - Alert history

- **Alert Types**
  - Threshold violations
  - System events
  - Irrigation activities
  - Daily summaries

---

## 💻 Requirements

### Hardware
- **Operating System**: Windows 11 (also compatible with Windows 10, Linux, macOS)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB free space
- **Processor**: Any modern CPU (2 cores+)

### Software
- **Python**: 3.8 or higher
- **Internet**: Required for initial package installation

### Optional Hardware
- Temperature/humidity sensors (DHT22, BME280)
- Soil moisture sensors
- Light sensors (LDR or digital)
- Relay modules for actuator control
- Raspberry Pi or Arduino (for hardware integration)

---

## 🔧 Installation

### Step 1: Install Python

Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/)

**Windows 11 Installation:**
```bash
# Download Python installer and check "Add Python to PATH"
# Verify installation:
python --version
```

### Step 2: Install Dependencies

```bash
# Navigate to greenhouse directory
cd path/to/greenhouse

# Install required packages
pip install -r greenhouse_requirements.txt
```

**For Windows 11 users**, the notification libraries will be automatically installed.

### Step 3: Verify Installation

```bash
# Test the system
python greenhouse_app.py test
```

---

## 🎯 Quick Start

### 1. Basic Monitoring

Start real-time monitoring with default settings:

```bash
python greenhouse_app.py monitor
```

**What happens:**
- Sensors start reading every 5 seconds
- Data is logged to `greenhouse_data/`
- Console displays current readings
- Press `Ctrl+C` to stop

**Example output:**
```
[2024-01-15 14:30:45] Temperature: 22.3°C | Humidity: 65.2% | Soil Moisture: 48.5% | Light: 18500 lux
```

### 2. Launch Dashboard

View data in your web browser:

```bash
python greenhouse_app.py dashboard
```

Then open: **http://127.0.0.1:8050**

### 3. Get Status Report

```bash
python greenhouse_app.py status
```

### 4. Growth Optimization

```bash
python greenhouse_app.py optimize
```

---

## 📖 Usage Guide

### Monitoring Mode

**Start continuous monitoring:**
```bash
python greenhouse_app.py monitor
```

**Features:**
- Real-time sensor readings
- Automatic data logging
- Alert checking
- Actuator control (if enabled)

### Dashboard Mode

**Launch web interface:**
```bash
python greenhouse_app.py dashboard
```

**Dashboard features:**
- 📊 Live sensor graphs
- 💡 Lighting controls
- 💧 Irrigation status
- 📈 Growth metrics
- ⚠️ Active alerts

### Training ML Models

**Collect data and train models:**
```bash
# Run monitoring to collect data (recommended: 24+ hours)
python greenhouse_app.py monitor

# Train models on collected data
python greenhouse_app.py train
```

**Requirements:**
- Minimum 100 data points
- Better performance with 1000+ points
- Automatic retraining every 7 days (configurable)

### Making Predictions

**Predict conditions 1-6 hours ahead:**
```bash
# Predict 1 hour ahead
python greenhouse_app.py predict --hours 1

# Predict 6 hours ahead
python greenhouse_app.py predict --hours 6
```

**Example output:**
```
Predictions for 6 hour(s) ahead:
  temperature: 23.4
  humidity: 62.1
  soil_moisture: 45.2
  light: 12000
```

---

## ⚙️ Configuration

### Configuration File

All settings are stored in `greenhouse_config.yaml`

### Key Settings

#### Sensor Configuration

```yaml
sensors:
  monitoring_interval: 5  # Seconds between readings
  thresholds:
    temperature:
      min: 15
      max: 30
      optimal: 22
```

#### Lighting Configuration

```yaml
lighting:
  auto_mode: true
  target_light_level: 20000  # lux
  photoperiod_hours: 16
  start_time: '06:00'
```

#### Irrigation Configuration

```yaml
irrigation:
  auto_mode: true
  moisture_threshold_min: 35
  moisture_threshold_max: 60
  schedule:
    - time: '07:00'
      duration: 60  # seconds
```

#### Alert Configuration

```yaml
alerts:
  enabled: true
  daily_summary: true
  daily_summary_time: '20:00'
```

### Editing Configuration

**Method 1: Edit YAML file directly**
```bash
notepad greenhouse_config.yaml  # Windows
nano greenhouse_config.yaml     # Linux/Mac
```

**Method 2: In Python**
```python
from greenhouse.config.greenhouse_config import GreenhouseConfig

config = GreenhouseConfig()
config.set('sensors.monitoring_interval', 10)
config.save()
```

---

## 📊 Dashboard

### Accessing the Dashboard

1. Start the dashboard:
   ```bash
   python greenhouse_app.py dashboard
   ```

2. Open browser to: `http://127.0.0.1:8050`

3. Dashboard auto-refreshes every 5 seconds

### Dashboard Sections

1. **Current Conditions Cards**
   - Temperature, Humidity, Soil Moisture, Light, CO2
   - Color-coded by parameter
   - Real-time values with units

2. **Environmental Monitoring Graphs**
   - Last 50 readings
   - 4 subplots (Temperature, Humidity, Moisture, Light)
   - Time-series visualization

3. **Lighting Control**
   - Current status of all lights
   - ON/OFF buttons
   - Intensity display
   - Mode indicator

4. **Irrigation Control**
   - Zone status (Active/Idle)
   - Water usage statistics
   - Emergency stop button

5. **Growth Metrics**
   - Growth score (0-100)
   - Daily growth rate
   - Days to harvest
   - Predicted yield

6. **Active Alerts**
   - Real-time alert notifications
   - Severity indicators

---

## 🤖 Machine Learning

### Available Models

1. **Condition Predictor**
   - Predicts future environmental conditions
   - Uses Random Forest or Gradient Boosting
   - Features: time, lagged values, rolling statistics

2. **Growth Predictor**
   - Calculates growth favorability score
   - Identifies limiting factors
   - Estimates yield

3. **Optimization Engine**
   - Recommends optimal settings
   - Multi-objective optimization
   - Resource efficiency analysis

### Training Models

```python
from greenhouse.ml_models.condition_predictor import ConditionPredictor

predictor = ConditionPredictor()
results = predictor.train(sensor_history)
predictor.save_models()
```

### Making Predictions

```python
current_conditions = {
    'temperature': 22.0,
    'humidity': 65.0,
    'soil_moisture': 50.0,
    'light': 20000
}

predictions = predictor.predict(current_conditions, hours_ahead=3)
```

### Growth Analysis

```python
from greenhouse.ml_models.growth_predictor import GrowthPredictor

growth_predictor = GrowthPredictor()
metrics = growth_predictor.estimate_growth_rate(current_conditions)

print(f"Growth Score: {metrics['growth_score']}/100")
print(f"Category: {metrics['growth_rate_category']}")
```

---

## 🏗️ Architecture

### Project Structure

```
greenhouse/
├── sensors/
│   ├── sensor_types.py       # Sensor implementations
│   └── sensor_manager.py     # Sensor coordination
├── actuators/
│   ├── lighting_controller.py
│   ├── irrigation_controller.py
│   └── actuator_manager.py
├── ml_models/
│   ├── condition_predictor.py
│   ├── growth_predictor.py
│   └── optimization_engine.py
├── dashboard/
│   └── web_dashboard.py
├── alerts/
│   └── alert_manager.py
└── config/
    └── greenhouse_config.py

greenhouse_app.py              # Main application
greenhouse_config.yaml         # Configuration
greenhouse_requirements.txt    # Dependencies
```

### Data Flow

```
Sensors → SensorManager → Data Storage
                       ↓
                  ActuatorManager
                       ↓
        Lighting/Irrigation Control
                       ↓
                  AlertManager
                       ↓
              Dashboard/Notifications
```

---

## 🔍 Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
pip install -r greenhouse_requirements.txt
```

### Issue: Notifications not working on Windows 11

**Solution:**
```bash
# Install Windows notification library
pip install winotify

# Test notifications
python greenhouse_app.py test
```

### Issue: Dashboard won't open

**Solution:**
- Check if port 8050 is in use
- Try different port in config:
  ```yaml
  dashboard:
    port: 8051
  ```
- Ensure Dash is installed:
  ```bash
  pip install dash plotly
  ```

### Issue: No sensor data

**Solution:**
- System uses simulated sensors by default
- Check `greenhouse_data/` directory for log files
- Verify monitoring interval in config

### Issue: ML models won't train

**Solution:**
- Need minimum 100 data points
- Run monitoring for longer period
- Check error messages for specific issues

---

## 🔐 Safety Features

- **Emergency Stop**: Immediately stops all actuators
- **Maximum Limits**: Prevents over-watering and extreme temperatures
- **Failsafe Defaults**: Safe defaults if configuration is invalid
- **Alert System**: Notifies of critical conditions
- **Data Backup**: Automatic data backup (configurable)

---

## 🌐 Future Enhancements

- Integration with real hardware sensors (Arduino, Raspberry Pi)
- Cloud data backup and remote monitoring
- Mobile app for iOS/Android
- Advanced ML models (LSTM, Transformers)
- Multi-greenhouse support
- Weather API integration
- Energy consumption optimization
- Plant disease detection using computer vision

---

## 📝 License

This project is for educational and personal use.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Hardware sensor integrations
- Additional crop-specific profiles
- Advanced ML models
- UI/UX enhancements
- Documentation improvements

---

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review configuration settings
3. Examine log files in `greenhouse_data/`
4. Test with `python greenhouse_app.py test`

---

## 🙏 Acknowledgments

Built with:
- Python
- scikit-learn (Machine Learning)
- Dash & Plotly (Dashboard)
- NumPy & Pandas (Data processing)
- PyYAML (Configuration)
- Winotify (Windows 11 notifications)

---

**🌱 Happy Growing! 🌿**

*Transform your greenhouse into a smart, automated growing environment with AI-powered insights.*

---
