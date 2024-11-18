
"""
    Represents a Lighting system that includes multiple bulbs for smart energy management.
    The LightingBulb class manages individual bulbs, and the Lighting class orchestrates
    multiple bulbs within a system.
"""

from device import Device

class LightingBulb(Device):
    def __init__(self, device_id, priority, min_rate=0.0525, max_rate=0.0685):
        # a 60-watt bulb can consume between 52.5 and 68.5 Watt-hours (Wh)

        super().__init__(device_id, priority, min_rate, max_rate)
        self.current_energy_usage = 0.06  # Default energy usage in kWh
        self.current_brightness = 100
        self.status = "off"

    def update_state(self, allocated_energy):
        """Update the bulb's energy usage."""
        self.current_energy_usage = allocated_energy

    def turn_on(self):
        """Turn on the bulb."""
        if self.status == "off":
            self.status = "on"

    def turn_off(self):
        """Turn off the bulb."""
        if self.status == "on":
            self.status = "off"

    def set_brightness(self, level):
        """
        Set the brightness level of the bulb.

        :param level: Brightness level (0 to 100).
        """
        if 0 <= level <= 100:
            self.current_brightness = level
            if self.status == "off":
                self.turn_on()

    def calculate_power_usage(self):
        """
        Calculate the current power usage based on brightness and allocated energy.

        :return: Power usage in kWh.
        """
        return (self.current_brightness / 100) * self.current_energy_usage


class Lighting:
    def __init__(self, duration, num_bulbs, priority, min_rate=0.0525, max_rate=0.0685):
        """
        Represents a lighting system managing multiple bulbs.

        :param num_bulbs: Number of bulbs in the system.
        :param duration: Duration for energy tracking (used to initialize power consumption lists).
        :param priority: Priority for the entire lighting system.
        :param min_rate: Minimum energy usage rate per bulb in kWh.
        :param max_rate: Maximum energy usage rate per bulb in kWh.
        """
        self.bulbs = [
            LightingBulb(f"Bulb-{i+1}", priority, min_rate, max_rate) for i in range(num_bulbs)
        ]
        self.power_consumption = [0] * (duration * 4 + 10)

    def allocate_energy(self, energy_allocation):
        """
        Allocate energy to each bulb in the system.
        """
        for bulb in self.bulbs:
            bulb.update_state(energy_allocation)
        
        

    def turn_on_all(self):
        """Turn on all bulbs in the system."""
        for bulb in self.bulbs:
            bulb.turn_on()

    def turn_off_all(self):
        """Turn off all bulbs in the system."""
        for bulb in self.bulbs:
            bulb.turn_off()

    def calculate_total_power_usage(self, time_interval):
        """
        Calculate the total power usage for the entire lighting system.
        """
        total = sum(bulb.calculate_power_usage() for bulb in self.bulbs)
        
        if self.power_consumption[time_interval] == 0:
            self.power_consumption[time_interval] = total * 1000

    def set_brightness_all(self, level):
        """
        Set the brightness for all bulbs.

        :param level: Brightness level (0 to 100).
        """
        for bulb in self.bulbs:
            bulb.set_brightness(level)
            
    
    