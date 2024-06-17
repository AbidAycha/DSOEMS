"""
    Represents the available power settings for a building, distinguishing between peak and off-peak hours.
"""

class AvailablePower:
    
    def __init__(self, peak_hours=None, peak_power=None, off_peak_power=None):
        self.peak_hours = peak_hours                 #List of tuples indicating the start and end hours for peak power usage, e.g., [(6, 10), (18, 21)]
        self.peak_power = peak_power*1000            #Maximum power (in kW) available during peak hours.
        self.off_peak_power = off_peak_power*1000    #Maximum power (in kW) available during off-peak hours.
       


    def get_power_for_interval(self, time_interval):
        hour = (time_interval / 4) % 24  

        try:
            for start, end in self.peak_hours:
                if start <= hour < end:
                    return self.peak_power
            return self.off_peak_power
        except Exception as e:
            print(f"Error in get_power_for_interval: {e}")
            raise
