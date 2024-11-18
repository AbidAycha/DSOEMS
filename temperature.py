import numpy as np


class SimulatedTemperature:
    def __init__(self, daily_low=15, daily_high=25, peak_temp_hour=15):
        """
        Initializes the temperature simulation settings.
        
        :param daily_low: The lowest temperature in degrees Celsius expected during the 24-hour cycle.
        :param daily_high: The highest temperature in degrees Celsius expected during the 24-hour cycle.
        :param peak_temp_hour: The hour of the day (0-23) when the highest temperature is expected.
        """
        self.daily_low = daily_low
        self.daily_high = daily_high
        self.peak_temp_hour = peak_temp_hour

    def get_temperature_for_time(self, time_interval):
        """
        Calculates the temperature for a given time interval based on a simple sinusoidal model.

        :param time_interval: Time interval in the format of quarters (0-95, where 0 corresponds to 00:00).
        :return: Simulated temperature at the given time interval.
        """
        hour = (time_interval / 4) % 24  # Convert time interval to hour
        # Use a sinusoidal function to simulate daily temperature change, peaking at peak_temp_hour
        temperature = self.daily_low + (self.daily_high - self.daily_low) * \
                      (0.5 + 0.5 * np.sin((hour - self.peak_temp_hour) * np.pi / 12 - np.pi / 2))
        return temperature
