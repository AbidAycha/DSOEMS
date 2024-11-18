"""
    This module leverages Fuzzy Logic (FL) for energy management in an IoT-based energy allocation system.
    FL uses rule-based reasoning to allocate available power to devices such as HVAC systems and EV charging stations,
    considering energy demand and power availability.

    The module contains two main components:
        - Fuzzy Controller: Implements a Fuzzy Logic-based decision-making system to allocate energy dynamically
          based on fuzzy rules for energy demand and available power.
        - ForecastModel4: Provides simplified energy demand predictions for HVAC and EV spots, feeding into the FL controller.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from hvac import Model
from ev_station import EV, ChargingStation

# Definition of priorities by level
level_priority = {
    "Level1": 3,
    "Level2": 2,
    "Level3": 1,
}


# Define fuzzy sets for energy demand
energy_demand = ctrl.Antecedent(np.arange(0, 101, 1), 'energy_demand')
energy_demand['faible'] = fuzz.trimf(energy_demand.universe, [0, 0, 50])
energy_demand['moyen'] = fuzz.trimf(energy_demand.universe, [0, 50, 75])
energy_demand['élevé'] = fuzz.trimf(energy_demand.universe, [50, 75, 100])

# Define fuzzy sets for power availability
available_power = ctrl.Antecedent(np.arange(0, 101, 1), 'available_power')
available_power['faible'] = fuzz.trimf(available_power.universe, [0, 0, 50])
available_power['moyen'] = fuzz.trimf(available_power.universe, [0, 50, 75])
available_power['élevé'] = fuzz.trimf(available_power.universe, [50, 75, 100])

# Define fuzzy sets for controlling energy demand
control_output = ctrl.Consequent(np.arange(0, 101, 1), 'control_output')
control_output['faible'] = fuzz.trimf(control_output.universe, [0, 0, 50])
control_output['moyen'] = fuzz.trimf(control_output.universe, [0, 50, 75])
control_output['élevé'] = fuzz.trimf(control_output.universe, [50, 75, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(energy_demand['faible'] & available_power['élevé'], control_output['faible'])
rule2 = ctrl.Rule(energy_demand['faible'] & available_power['moyen'], control_output['faible'])
rule3 = ctrl.Rule(energy_demand['faible'] & available_power['faible'], control_output['faible'])

rule4 = ctrl.Rule(energy_demand['moyen'] & available_power['élevé'], control_output['moyen'])
rule5 = ctrl.Rule(energy_demand['moyen'] & available_power['moyen'], control_output['faible'])
rule6 = ctrl.Rule(energy_demand['moyen'] & available_power['faible'], control_output['faible'])

rule7 = ctrl.Rule(energy_demand['élevé'] & available_power['élevé'], control_output['élevé'])
rule8 = ctrl.Rule(energy_demand['élevé'] & available_power['moyen'], control_output['moyen'])
rule9 = ctrl.Rule(energy_demand['élevé'] & available_power['faible'], control_output['faible'])

# Create a fuzzy control system
control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
control_simulation = ctrl.ControlSystemSimulation(control_system)

class FuzzyController:
    def __init__(self, hvac_unit, ev_charging_station_spots, forecast_model):
        self.hvac_unit = hvac_unit
        self.ev_charging_station_spots = ev_charging_station_spots or []
        self.forecast_model = forecast_model

    def control_step(self, available_power):
        
        future_demands = self.forecast_model.predict_demand(self.hvac_unit, self.ev_charging_station_spots, available_power)

    
        devices = []
        if self.hvac_unit is not None:
            devices.append(self.hvac_unit)
        devices += [spot for spot in self.ev_charging_station_spots if spot.occupied]
        devices.sort(key=lambda x: x.priority)  # Sort the devices by priority

        if any(device.priority for device in devices):
            devices.sort(key=lambda x: x.priority) 



        # Calculate the optimal power for each device
        optimal_powers = []
        for device in devices:
            current_demand = future_demands[devices.index(device)]
            control_simulation.input['energy_demand'] = current_demand
            control_simulation.input['available_power'] = available_power
            control_simulation.compute()
            optimal_power = control_simulation.output['control_output']
            #print( "hey", optimal_power, available_power,current_demand)
            device.update_state(optimal_power)
            optimal_powers.append(optimal_power)
        
        return optimal_powers

class ForecastModel4:
    def predict_demand(self, hvac_unit, ev_charging_spots, power_capacity):
        predictions = []

        if hvac_unit is not None:
            hvac_demand = hvac_unit.calculate_energy_required(hvac_unit.var.TIn, hvac_unit.var.TRoom) * 1000
            predictions.append(hvac_demand)
        
        total_demand = 0
        for spot in ev_charging_spots:
            if spot.occupied:
                remaining_time = spot.remaining_charge_time(spot.current_ev)
                demand_rate = np.clip(spot.current_energy_usage, spot.energy_min, spot.energy_max)
                spot_demand = demand_rate
                total_demand += spot_demand
                predictions.append(spot_demand)
            else:
                predictions.append(0)
        
        return predictions


