"""
    Abstract base class representing a generic device in an energy management system.
    Each device has a name, priority, and operational energy bounds (minimum and maximum energy requirements).
"""

from abc import ABC, abstractmethod


class Device(ABC):
    def __init__(self, name, priority, energy_min, energy_max):
        self.name = name
        self.priority = priority
        self.energy_min = energy_min  
        self.energy_max = energy_max  
        self.status = "off"
       
        self.current_energy_usage = 0 

    # The class is designed to be subclassed by specific types of devices which must implement their own state update logic.
    @abstractmethod
    def update_state(self, allocated_energy):
        """Update the state of the device based on allocated energy."""
        pass

    