"""
Example: Irrigation Control
Demonstrates automated watering based on soil moisture
"""
import time
import random
from greenhouse.actuators.irrigation_controller import IrrigationController, WateringZone


def main():
    print("="*60)
    print("Irrigation Control Example")
    print("="*60 + "\n")

    # Create irrigation controller
    controller = IrrigationController()

    # Add watering zones
    zone1 = WateringZone("zone_01", "Vegetable Bed 1", flow_rate=2.0)
    zone2 = WateringZone("zone_02", "Vegetable Bed 2", flow_rate=2.5)

    controller.register_zone(zone1)
    controller.register_zone(zone2)

    print("Demonstration of irrigation features:\n")

    # 1. Manual watering
    print("1. Manual Watering:")
    print("   Starting zone 1...")
    controller.start_zone("zone_01")
    time.sleep(3)
    print("   Stopping zone 1...")
    controller.stop_zone("zone_01")

    status = controller.get_all_status()
    print(f"   Water used: {status['zones']['zone_01']['total_water_used']:.2f} L")

    # 2. Auto irrigation setup
    print("\n2. Enabling Auto Irrigation:")
    print("   Setting thresholds: 35% min, 60% max")
    controller.enable_auto_irrigation(min_moisture=35, max_moisture=60)

    # 3. Simulate soil moisture readings
    print("\n3. Simulating Moisture-Based Irrigation:")

    # Simulate different moisture levels
    moisture_scenarios = [
        {"soil_01": 30, "soil_02": 45},  # Zone 1 needs water
        {"soil_01": 50, "soil_02": 32},  # Zone 2 needs water
        {"soil_01": 55, "soil_02": 58},  # Both OK
    ]

    for i, moisture_readings in enumerate(moisture_scenarios, 1):
        print(f"\n   Scenario {i}:")
        for sensor_id, moisture in moisture_readings.items():
            zone_id = f"zone_0{sensor_id[-1]}"
            controller.link_moisture_sensor(zone_id, sensor_id)
            print(f"   {sensor_id}: {moisture}% moisture")

        results = controller.check_moisture_levels(moisture_readings)
        for result in results:
            print(f"   → {result}")

        time.sleep(2)

    # 4. Irrigation schedule
    print("\n4. Scheduled Irrigation:")
    controller.add_schedule("zone_01", "07:00", 60)
    controller.add_schedule("zone_02", "19:00", 45)
    print("   Schedule added:")
    print("   - Zone 1: 07:00 for 60 seconds")
    print("   - Zone 2: 19:00 for 45 seconds")

    # 5. Smart recommendations
    print("\n5. Smart Irrigation Recommendations:")
    current_moisture = 38
    recommendation = controller.smart_irrigation_recommendation(
        current_moisture=current_moisture,
        weather_forecast="sunny",
        plant_type="vegetables"
    )

    print(f"   Current moisture: {current_moisture}%")
    print(f"   Should water: {recommendation['should_water']}")
    print(f"   Recommended duration: {recommendation['recommended_duration']}s")
    print(f"   Reason: {recommendation['reason']}")

    # 6. Water usage report
    print("\n6. Water Usage Report:")
    report = controller.get_water_usage_report()
    print(f"   Total water used: {report['total_water_used']:.2f} L")
    for zone_info in report['zones']:
        print(f"   {zone_info['zone']}: {zone_info['water_used']:.2f} L")

    print("\n" + "="*60)
    print("Irrigation control example completed!")
    print("="*60)


if __name__ == "__main__":
    main()
