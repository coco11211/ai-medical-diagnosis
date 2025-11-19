"""
Example: Basic Sensor Monitoring
Demonstrates simple sensor setup and monitoring
"""
import time
from greenhouse.sensors.sensor_types import TemperatureSensor, HumiditySensor, SoilMoistureSensor, LightSensor
from greenhouse.sensors.sensor_manager import SensorManager


def main():
    print("="*60)
    print("Basic Greenhouse Monitoring Example")
    print("="*60 + "\n")

    # Create sensor manager
    sensor_manager = SensorManager(data_dir="example_data")

    # Register sensors
    temp_sensor = TemperatureSensor("temp_01", "Temperature", "Main Zone")
    humid_sensor = HumiditySensor("humid_01", "Humidity", "Main Zone")
    soil_sensor = SoilMoistureSensor("soil_01", "Soil Moisture", "Bed 1")
    light_sensor = LightSensor("light_01", "Light", "Main Zone")

    sensor_manager.register_sensor(temp_sensor)
    sensor_manager.register_sensor(humid_sensor)
    sensor_manager.register_sensor(soil_sensor)
    sensor_manager.register_sensor(light_sensor)

    print("\nMonitoring for 30 seconds (press Ctrl+C to stop)...\n")

    try:
        for i in range(6):  # 6 readings over 30 seconds
            # Read all sensors
            readings = sensor_manager.read_all_sensors()

            # Display readings
            print(f"Reading {i+1}:")
            for sensor_id, reading in readings.items():
                if 'value' in reading:
                    print(f"  {reading['name']:15s}: {reading['value']:6.1f} {reading['unit']}")

            # Get statistics
            if i == 5:  # Last reading
                print("\nStatistics:")
                stats = sensor_manager.get_statistics()
                for sensor_id, sensor_stats in stats.items():
                    print(f"  {sensor_stats['name']:15s}: avg={sensor_stats['avg']:.1f}, "
                          f"min={sensor_stats['min']:.1f}, max={sensor_stats['max']:.1f}")

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == "__main__":
    main()
