"""
Smart Greenhouse Control Application
Main application for greenhouse automation
"""
import time
import signal
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Greenhouse modules
from greenhouse.sensors.sensor_types import TemperatureSensor, HumiditySensor, SoilMoistureSensor, LightSensor, CO2Sensor
from greenhouse.sensors.sensor_manager import SensorManager
from greenhouse.actuators.lighting_controller import LightingController, GrowLight
from greenhouse.actuators.irrigation_controller import IrrigationController, WateringZone
from greenhouse.actuators.actuator_manager import ActuatorManager
from greenhouse.ml_models.condition_predictor import ConditionPredictor
from greenhouse.ml_models.growth_predictor import GrowthPredictor
from greenhouse.ml_models.optimization_engine import OptimizationEngine
from greenhouse.alerts.alert_manager import AlertManager, AlertLevel
from greenhouse.dashboard.web_dashboard import GreenhouseDashboard
from greenhouse.config.greenhouse_config import GreenhouseConfig


class SmartGreenhouse:
    """Main greenhouse control system"""

    def __init__(self, config_file: str = "greenhouse_config.yaml"):
        print("\n" + "="*70)
        print("🌱 SMART GREENHOUSE CONTROL SYSTEM".center(70))
        print("Advanced Monitoring & Automation for Windows 11".center(70))
        print("="*70 + "\n")

        # Load configuration
        self.config = GreenhouseConfig(config_file)

        # Initialize data directories
        self.data_dir = Path(self.config.get('data_storage.data_directory', 'greenhouse_data'))
        self.model_dir = Path(self.config.get('data_storage.model_directory', 'greenhouse_models'))
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)

        # Initialize managers
        self.sensor_manager = SensorManager(data_dir=str(self.data_dir))
        self.actuator_manager = ActuatorManager(data_dir=str(self.data_dir))
        self.alert_manager = AlertManager(data_dir=str(self.data_dir))

        # Initialize ML models
        self.condition_predictor = ConditionPredictor(model_dir=str(self.model_dir))
        self.growth_predictor = GrowthPredictor()
        self.optimization_engine = OptimizationEngine()

        # Setup sensors and actuators
        self._setup_sensors()
        self._setup_actuators()

        # Dashboard
        self.dashboard = None

        # Running flag
        self.running = False

        print("✓ Greenhouse system initialized successfully\n")

    def _setup_sensors(self):
        """Setup all sensors"""
        print("Setting up sensors...")

        # Temperature sensors
        temp1 = TemperatureSensor("temp_01", "Temperature", "Zone 1")
        self.sensor_manager.register_sensor(temp1)

        # Humidity sensors
        humid1 = HumiditySensor("humid_01", "Humidity", "Zone 1")
        self.sensor_manager.register_sensor(humid1)

        # Soil moisture sensors
        soil1 = SoilMoistureSensor("soil_01", "Soil Moisture", "Bed 1")
        soil2 = SoilMoistureSensor("soil_02", "Soil Moisture", "Bed 2")
        self.sensor_manager.register_sensor(soil1)
        self.sensor_manager.register_sensor(soil2)

        # Light sensor
        light1 = LightSensor("light_01", "Light", "Zone 1")
        self.sensor_manager.register_sensor(light1)

        # CO2 sensor
        co2_1 = CO2Sensor("co2_01", "CO2", "Zone 1")
        self.sensor_manager.register_sensor(co2_1)

        print(f"✓ Registered {len(self.sensor_manager.sensors)} sensors\n")

    def _setup_actuators(self):
        """Setup all actuators"""
        print("Setting up actuators...")

        # Lighting
        light1 = GrowLight("light_01", "Main Grow Light", "Zone 1", max_intensity=100)
        light2 = GrowLight("light_02", "Supplemental Light", "Zone 1", max_intensity=80)
        self.actuator_manager.lighting.register_light(light1)
        self.actuator_manager.lighting.register_light(light2)

        # Irrigation zones
        zone1 = WateringZone("zone_01", "Bed 1", flow_rate=2.0)
        zone2 = WateringZone("zone_02", "Bed 2", flow_rate=2.0)
        self.actuator_manager.irrigation.register_zone(zone1)
        self.actuator_manager.irrigation.register_zone(zone2)

        # Link moisture sensors to irrigation zones
        self.actuator_manager.irrigation.link_moisture_sensor("zone_01", "soil_01")
        self.actuator_manager.irrigation.link_moisture_sensor("zone_02", "soil_02")

        print(f"✓ Registered lighting and irrigation systems\n")

    def start_monitoring(self):
        """Start monitoring and automation"""
        print("Starting greenhouse monitoring...")

        # Start sensor monitoring
        interval = self.config.get('sensors.monitoring_interval', 5)
        self.sensor_manager.start_monitoring(interval)

        # Apply lighting configuration
        lighting_config = self.config.get_lighting_config()
        if lighting_config.get('auto_mode'):
            target_lux = lighting_config.get('target_light_level', 20000)
            self.actuator_manager.lighting.enable_auto_mode(target_lux)

        # Apply irrigation configuration
        irrigation_config = self.config.get_irrigation_config()
        if irrigation_config.get('auto_mode'):
            min_moisture = irrigation_config.get('moisture_threshold_min', 35)
            max_moisture = irrigation_config.get('moisture_threshold_max', 60)
            self.actuator_manager.irrigation.enable_auto_irrigation(min_moisture, max_moisture)

        # Enable full automation if configured
        if self.config.is_automation_enabled():
            self.actuator_manager.enable_full_automation()

        self.running = True

        # Send startup notification
        if self.config.get('alerts.enabled'):
            self.alert_manager.send_system_alert(
                "Greenhouse system started successfully",
                AlertLevel.INFO
            )

        print("✓ Monitoring started\n")

    def monitoring_loop(self):
        """Main monitoring loop"""
        print("Entering monitoring loop (Press Ctrl+C to stop)...\n")

        try:
            while self.running:
                # Read all sensors
                readings = self.sensor_manager.read_all_sensors()

                # Check for alerts
                if self.config.get('alerts.enabled'):
                    thresholds = self.config.get_sensor_thresholds()
                    # Convert sensor names to IDs for threshold checking
                    sensor_thresholds = {}
                    for sensor_id, reading in readings.items():
                        sensor_name = reading.get('name', '').lower().replace(' ', '_')
                        if sensor_name in thresholds:
                            sensor_thresholds[sensor_id] = thresholds[sensor_name]

                    self.alert_manager.check_sensor_alerts(readings, sensor_thresholds)

                # Process sensor data with actuator manager
                actions = self.actuator_manager.process_sensor_data(readings)

                # Display current status
                self._display_status(readings)

                # Wait for next cycle
                time.sleep(self.config.get('sensors.monitoring_interval', 5))

        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
            self.stop()

    def _display_status(self, readings: dict):
        """Display current status"""
        print(f"\r[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ", end='')

        status_parts = []
        for sensor_id, reading in readings.items():
            if 'value' in reading:
                name = reading['name']
                value = reading['value']
                unit = reading['unit']
                status_parts.append(f"{name}: {value:.1f}{unit}")

        print(" | ".join(status_parts[:4]), end='', flush=True)

    def run_dashboard(self):
        """Run the web dashboard"""
        dashboard_config = self.config.get('dashboard', {})

        self.dashboard = GreenhouseDashboard(
            sensor_manager=self.sensor_manager,
            actuator_manager=self.actuator_manager,
            host=dashboard_config.get('host', '127.0.0.1'),
            port=dashboard_config.get('port', 8050)
        )

        self.dashboard.run()

    def train_ml_models(self):
        """Train ML prediction models"""
        print("\nTraining ML prediction models...")

        history = self.sensor_manager.get_history(limit=500)

        if len(history) < 100:
            print(f"⚠ Insufficient data for training (need 100+ samples, have {len(history)})")
            return

        try:
            results = self.condition_predictor.train(history)
            print("\nTraining results:")
            for condition, metrics in results.items():
                print(f"  {condition}: R² = {metrics['test_r2']:.3f}, RMSE = {metrics['rmse']:.3f}")

            # Save models
            self.condition_predictor.save_models()
            print("\n✓ Models trained and saved successfully")

        except Exception as e:
            print(f"Error training models: {e}")

    def get_predictions(self, hours_ahead: int = 1):
        """Get predictions for future conditions"""
        # Get current conditions
        readings = self.sensor_manager.read_all_sensors()

        current_conditions = {}
        for sensor_id, reading in readings.items():
            if 'value' in reading:
                sensor_name = reading['name'].lower().replace(' ', '_')
                current_conditions[sensor_name] = reading['value']

        try:
            predictions = self.condition_predictor.predict(current_conditions, hours_ahead)
            print(f"\nPredictions for {hours_ahead} hour(s) ahead:")
            for condition, value in predictions.items():
                print(f"  {condition}: {value:.1f}")
            return predictions

        except Exception as e:
            print(f"Error getting predictions: {e}")
            return {}

    def get_growth_metrics(self):
        """Get plant growth metrics"""
        readings = self.sensor_manager.read_all_sensors()

        current_conditions = {}
        for sensor_id, reading in readings.items():
            if 'value' in reading:
                sensor_name = reading['name'].lower().replace(' ', '_')
                current_conditions[sensor_name] = reading['value']

        metrics = self.growth_predictor.estimate_growth_rate(current_conditions)

        print("\n" + "="*60)
        print("GROWTH METRICS")
        print("="*60)
        print(f"Growth Score: {metrics['growth_score']}/100 ({metrics['growth_rate_category']})")
        print(f"Estimated Daily Growth: {metrics['estimated_daily_growth_cm']} cm/day")

        if metrics['limiting_factors']:
            print("\nLimiting Factors:")
            for factor in metrics['limiting_factors']:
                print(f"  • {factor['parameter']}: {factor['current']:.1f} ({factor['status']})")

        print("="*60 + "\n")

        return metrics

    def optimize_settings(self):
        """Get optimization recommendations"""
        readings = self.sensor_manager.read_all_sensors()

        current_conditions = {}
        for sensor_id, reading in readings.items():
            if 'value' in reading:
                sensor_name = reading['name'].lower().replace(' ', '_')
                current_conditions[sensor_name] = reading['value']

        # Actuator constraints (example)
        constraints = {
            'temperature': (15, 30),
            'humidity': (40, 80),
            'soil_moisture': (30, 70),
            'light': (10000, 40000)
        }

        recommendations = self.optimization_engine.optimize_for_growth(
            current_conditions,
            constraints
        )

        print("\n" + "="*60)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*60)

        for action in recommendations['priority_actions'][:5]:
            print(f"  [{action['priority']}] {action['action'].upper()} {action['parameter']} by {action['magnitude']:.1f}")

        print("="*60 + "\n")

        return recommendations

    def stop(self):
        """Stop the greenhouse system"""
        print("\nStopping greenhouse system...")

        self.running = False

        # Stop monitoring
        self.sensor_manager.stop_monitoring()

        # Stop all actuators
        self.actuator_manager.emergency_stop()

        # Send shutdown notification
        if self.config.get('alerts.enabled'):
            self.alert_manager.send_system_alert(
                "Greenhouse system shut down",
                AlertLevel.INFO
            )

        print("✓ System stopped safely\n")

    def status_report(self):
        """Generate comprehensive status report"""
        print("\n" + "="*70)
        print("GREENHOUSE STATUS REPORT".center(70))
        print("="*70)

        # Sensor statistics
        stats = self.sensor_manager.get_statistics()
        print("\nSENSOR READINGS:")
        for sensor_id, sensor_stats in stats.items():
            print(f"  {sensor_stats['name']:20s}: {sensor_stats['current']:7.1f} {sensor_stats['unit']:5s} "
                  f"(avg: {sensor_stats['avg']:.1f}, min: {sensor_stats['min']:.1f}, max: {sensor_stats['max']:.1f})")

        # Actuator status
        actuator_status = self.actuator_manager.get_full_status()
        print(f"\nACTUATORS:")
        print(f"  Automation: {'ENABLED' if actuator_status['automation_enabled'] else 'DISABLED'}")
        print(f"  Power Consumption: {actuator_status['power_consumption_watts']:.1f} W")
        print(f"  Water Used: {actuator_status['water_usage']['total_water_used']:.1f} L")

        # Active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        print(f"\nACTIVE ALERTS: {len(active_alerts)}")
        for alert in active_alerts[:5]:
            print(f"  [{alert['level']}] {alert['title']}: {alert['message']}")

        print("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Smart Greenhouse Control System')
    parser.add_argument('command', nargs='?', default='monitor',
                       choices=['monitor', 'dashboard', 'train', 'predict', 'optimize', 'status', 'test'],
                       help='Command to execute')
    parser.add_argument('--config', default='greenhouse_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--hours', type=int, default=1,
                       help='Hours ahead for prediction')

    args = parser.parse_args()

    # Initialize greenhouse
    greenhouse = SmartGreenhouse(config_file=args.config)

    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        greenhouse.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Execute command
    if args.command == 'monitor':
        greenhouse.start_monitoring()
        greenhouse.monitoring_loop()

    elif args.command == 'dashboard':
        greenhouse.start_monitoring()
        greenhouse.run_dashboard()

    elif args.command == 'train':
        greenhouse.start_monitoring()
        time.sleep(10)  # Collect some data
        greenhouse.train_ml_models()
        greenhouse.stop()

    elif args.command == 'predict':
        greenhouse.start_monitoring()
        time.sleep(5)
        greenhouse.get_predictions(hours_ahead=args.hours)
        greenhouse.stop()

    elif args.command == 'optimize':
        greenhouse.start_monitoring()
        time.sleep(5)
        greenhouse.optimize_settings()
        greenhouse.get_growth_metrics()
        greenhouse.stop()

    elif args.command == 'status':
        greenhouse.start_monitoring()
        time.sleep(5)
        greenhouse.status_report()
        greenhouse.stop()

    elif args.command == 'test':
        print("Testing notification system...")
        greenhouse.alert_manager.test_notification()
        print("✓ Test complete")


if __name__ == "__main__":
    main()
