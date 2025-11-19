"""
Example: Lighting Control
Demonstrates grow light control and scheduling
"""
import time
from datetime import datetime
from greenhouse.actuators.lighting_controller import LightingController, GrowLight


def main():
    print("="*60)
    print("Lighting Control Example")
    print("="*60 + "\n")

    # Create lighting controller
    controller = LightingController()

    # Add grow lights
    light1 = GrowLight("led_01", "Main LED Grow Light", "Zone 1")
    light2 = GrowLight("led_02", "Supplemental Light", "Zone 1")

    controller.register_light(light1)
    controller.register_light(light2)

    print("Demonstration of lighting control features:\n")

    # 1. Manual control
    print("1. Manual Control:")
    print("   Turning on lights at 80% intensity...")
    controller.turn_on_all(80)
    time.sleep(2)

    status = controller.get_all_status()
    for light_id, light_info in status['lights'].items():
        print(f"   {light_info['name']}: {light_info['intensity']}% ({'ON' if light_info['is_on'] else 'OFF'})")

    time.sleep(2)

    # 2. Intensity adjustment
    print("\n2. Intensity Adjustment:")
    print("   Setting Zone 1 to 50% intensity...")
    controller.set_zone_intensity("Zone 1", 50)
    time.sleep(1)

    for light_id, light_info in controller.get_all_status()['lights'].items():
        print(f"   {light_info['name']}: {light_info['intensity']}%")

    # 3. Scheduling
    print("\n3. Setting Photoperiod Schedule:")
    print("   16 hours on, starting at 06:00...")
    controller.set_photoperiod(16, "06:00")

    for light in controller.lights.values():
        print(f"   {light.name}: Schedule set ({len(light.schedule)} periods)")

    # 4. Auto mode
    print("\n4. Auto Mode (light-based adjustment):")
    print("   Enabling auto mode (target: 20000 lux)...")
    controller.enable_auto_mode(20000)

    # Simulate different ambient light levels
    ambient_levels = [5000, 15000, 25000, 10000]
    for ambient in ambient_levels:
        controller.auto_adjust(ambient)
        print(f"   Ambient: {ambient} lux → Lights adjusted")
        time.sleep(1)

    # 5. Power consumption
    print("\n5. Power Consumption:")
    power = controller.get_power_consumption()
    print(f"   Current power usage: {power:.1f} W")

    # Turn off all lights
    print("\n6. Turning off all lights...")
    controller.turn_off_all()

    print("\n" + "="*60)
    print("Lighting control example completed!")
    print("="*60)


if __name__ == "__main__":
    main()
