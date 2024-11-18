"""
    This module defines the structure and functionality for an Electric Vehicle (EV) charging station simulation. 
    It includes classes for representing electric vehicles (EV), charging spots, and the charging station itself. 
    Each class is equipped with methods to handle the specifics of EV charging processes, such as power management, 
    charging status updates, and allocation of charging spots based on EV arrival and charging requirements. 

    The EV class defines individual electric vehicles with attributes for tracking their charging status and battery levels.
    The ChargingSpot class represents a physical spot in the charging station and manages the power allocation and the connection status of an EV.
    The ChargingStation class orchestrates the overall operation of the station, managing multiple charging spots, queuing systems for incoming EVs, 
    and the dynamic adjustment of power distribution to accommodate current power availability and EV charging demands.
"""


from device import Device


level_priority = {'Level1': 3, 'Level2': 2, 'Level3': 1}
CHARGING_LEVELS = {'Level1': 2, 'Level2': 7.6, 'Level3': 50}  # kW
SPOTS_PER_LEVEL = 2
battery_capacity = 60 #kW


class EV:
    """
        Represents an electric vehicle in the simulation environment.
        Includes attributes to manage the charging status, battery state, and charging parameters.
    """
    def __init__(self, id, arrival, desired_level, battery_state, capacity, max_rate, min_rate=1.3):
        """
            Initializes the electric vehicle with the given parameters.

            Args:
            id (int): The identifier for the electric vehicle.
            arrival (int): Time of arrival of the EV.
            desired_level (str): Desired charging level (e.g., 'Level1', 'Level2', 'Level3').
            battery_state (float): Current state of the battery as a percentage.
            capacity (float): Total capacity of the battery in kW.
            max_rate (float): Maximum charging rate.
            min_rate (float): Minimum charging rate.
        """
        self.id = id
        self.arrival = arrival
        self.desired_level = desired_level
        self.battery_state = battery_state  
        self.min_rate = min_rate 
        self.max_rate = max_rate 
        self.charging_rate = min(CHARGING_LEVELS[desired_level], max_rate)
        self.capacity = capacity
        self.status = 'waiting'  # different status: waiting, paused, charging, finished
        self.start_time = None
        self.end_time = None
        self.paused_periods = []
        
    def update_charging_rate(self,rate):
        self.charging_rate = min(rate, self.max_rate)
    def update_battery_state(self):
        
        self.battery_state += (self.charging_rate / self.capacity) * 100 / 4 
        
        
        
class ChargingSpot(Device):
    """
        Represents a single charging spot in the EV charging station that extends the generic Device class.
    """
    def __init__(self, level, priority, rate, min, max):
        """
            Initialize the charging spot with specific parameters related to power management.

            Args:
            level (str): Charging level of the spot (e.g., 'Level1').
            priority (int): Operational priority of the spot.
            rate (float): Initial charging rate.
            min (float): Minimum power allocation.
            max (float): Maximum power allocation.
        """
        super().__init__("CSpot",priority, min, max)

        self.level = level
        self.occupied = False
        self.current_ev = None
        self.current_energy_usage=rate
  
    # Updates the state of the charging spot based on the allocated energy.
    def update_state(self, allocated_energy):
        self.current_ev.update_charging_rate(allocated_energy)
       
        #self.current_ev.charging_rate=allocated_energy
        self.current_energy_usage=allocated_energy
    # Calculates the remaining charge time for an EV based on its current state.    
    def remaining_charge_time_old(self, ev=None):
        if not ev:
            return 0
        return (max(int(4 * ev.capacity * (100 - ev.battery_state) / (100*ev.charging_rate)), 0))
    
    def remaining_charge_time(self, ev=None):
        if not ev:
            return 0
        #print(ev.id, ',',ev.capacity,',', ev.battery_state,',', ev.charging_rate)
        if ev.charging_rate:
            return (max(int(4 * ev.capacity * (100 - ev.battery_state) / (100*ev.charging_rate)), 0))
        return (max(int(4 * ev.capacity * (100 - ev.battery_state) / (100*ev.min_rate)), 0))
      
    
class ChargingStation():
    """
        Manages the operations of an entire charging station, including handling of EVs, charging spots, and power management.
    """
    def __init__(self, duration, level_config):
        self.charging_levels = CHARGING_LEVELS 
        self.spots = []
        self.level_config=level_config
        # Iterate through each level configuration provided in level_config
        for config in level_config:
            level = config['level']
            level_rate = config['rate']
            priority = config['priority']
            min_energy = config['min_energy']
            max_energy = config['max_energy']
            spots_count = config['spots']

            # Create the specified number of ChargingSpot instances for the current level
            for _ in range(spots_count):
                spot = ChargingSpot(level, priority, level_rate, min_energy, max_energy)
                self.spots.append(spot)
        #print("ev station", self.spots)
        self.queue = []
        self.charging_evs = []
        self.finished_evs = []
        self.power_consumption = [0] * (duration*4+10)  
    
    def get_priority( self, desired_level):
        for config in self.level_config:
            if config['level'] == desired_level:
                return config['priority']
        return None  # Return None or an appropriate default value if the level is not found

    def check_spot(self,ev):
        for spot in self.spots:
            if (spot.occupied and spot.current_ev==ev) :
                #print("hey", spot.occupied)
                return spot
        return False

    def adjust_charging_to_power_availability(self,time_interval, power):
        total_power_needed = sum(ev.charging_rate for ev in self.charging_evs if ev.status == 'charging') * 1000
        
        if total_power_needed > power:
            # If total needed power exceeds available power, sort charging EVs by their level priority, charging rate and battery state
            sorted_evs = sorted(self.charging_evs, key=lambda ev: (self.get_priority(ev.desired_level), ev.charging_rate, ev.battery_state))

            for ev in sorted_evs:
                if ev.status == 'charging':
                    ev.status = 'paused'
                    ev.paused_periods.append([time_interval, None])
                    total_power_needed -= ev.charging_rate * 1000
                    if total_power_needed <= power:
                        break  
            
                
        else:
            for ev in sorted(self.charging_evs, key=lambda ev: (-(self.get_priority(ev.desired_level)), -ev.charging_rate)):
                if ev.status == 'paused':
                    additional_power_needed = ev.charging_rate * 1000
                    if total_power_needed + additional_power_needed <= power:
                        ev.paused_periods[-1][1] = time_interval
                        ev.status = 'charging'
                        total_power_needed += additional_power_needed
                        #self.power_consumption[time_interval] += additional_power_needed
                    else:
                        
                        break  
        for ev in self.charging_evs:
            if ev.status == 'paused':
                ev.end_time+=1
                # print ("end paused", ev.end_time)
           
    def process_evs(self, power, time_interval):

        for ev in self.charging_evs:
            

            if time_interval >= ev.end_time or ev.battery_state==100 :
                ev.status = 'finished'
                self.finished_evs.append(ev)
                self.charging_evs.remove(ev)
                self.free_spot(ev)
            if ev.status == 'charging':
                
                current_spot=self.check_spot(ev)
                ev.end_time = time_interval + current_spot.remaining_charge_time(ev)
                #print("time_interval charging", time_interval)
                #print ("end charging", ev.end_time)
                ev.update_battery_state()
                #print("hey", time_interval, ev.battery_state, ev.charging_rate, ev.end_time)
                #ev.end_time = time_interval + self.check_spot(ev).remaining_charge_time(ev)
        for ev in self.queue.copy():
            if self.check_power_availability(ev, power):
                if self.start_charging(ev, time_interval):
                    self.queue.remove(ev)
            elif not self.check_spot(ev):
                
                for spot in self.spots:
                    
                    if not spot.occupied and spot.level == ev.desired_level:   
                             
                        spot.occupied = True
                        spot.current_ev = ev
                        break
                        
                         
        self.adjust_charging_to_power_availability(time_interval, power)

        
    def start_charging(self, ev, time_interval):
        current_spot=self.check_spot(ev)
        if not current_spot:
            
            for spot in self.spots:
                if not spot.occupied and spot.level == ev.desired_level:
                    current_spot= spot
                    spot.current_ev = ev
                    
                    spot.occupied = True
                    break
            if not current_spot: 
                return False
        #print("here", current_spot, current_spot.occupied)
        ev.start_time = time_interval
        # print("time_interval start", time_interval)
        ev.end_time = ev.start_time + current_spot.remaining_charge_time(ev)
        #print ("end start", ev.end_time, "remaining", current_spot.remaining_charge_time(ev))
        ev.status = 'charging'
        self.charging_evs.append(ev)            
        return True

    def free_spot(self, ev):
        for spot in self.spots:
            if spot.current_ev == ev:
                spot.occupied = False
                spot.current_ev = None
                break

    def check_power_availability(self, ev,power):
        projected_consumption = sum(ev.charging_rate for ev in self.charging_evs if ev.status == 'charging')*1000
        projected_consumption += ev.charging_rate*1000
        return projected_consumption <= power
