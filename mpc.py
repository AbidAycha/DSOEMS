"""
     This module is integral to an energy management system, specifically designed for forecasting energy demand
    and optimizing power allocation through Model Predictive Control (MPC). 
    It utilizes the convex optimization library cvxpy to manage and allocate energy resources effectively across different IoT devices.
    The module contains two main classes: 
        - ForecastModel, which predicts future energy demands
        - MPCController, which applies MPC techniques to optimize energy usage and minimize cost while adhering to operational constraints and priorities.
"""

import cvxpy as cp
import numpy as np
from lighting import Lighting
from temperature import SimulatedTemperature

level_priority = {'Level1': 3, 'Level2': 2, 'Level3': 1}

class ForecastModel:
    """
        Forecasts energy demands based on the current state and specifications of HVAC units and EV charging stations.
    """
    def __init__(self):
        self.previous_allocations = None
        
    def predict_demand(self, lighting, hvac_unit, ev_charging_spots, station, power_capacity, prediction_horizon, current_time_interval):
        """
            Predicts energy demand using available data on HVAC, EV spots, and Lighting.
            This method constructs a time-series prediction over a specified horizon for each device based on its current status and power usage profiles. 
            The result is an array where each row corresponds to a device and each column corresponds to a time step in the prediction horizon.      
        """
        temperature_simulator = SimulatedTemperature()
        predicted_temperatures = [temperature_simulator.get_temperature_for_time(current_time_interval + i)
                                for i in range(prediction_horizon)]

        predictions = []
        if ev_charging_spots: 
            spot_availability = np.full((len(ev_charging_spots), prediction_horizon), False)  # Track availability of each spot
        else:
            spot_availability = np.full((0, prediction_horizon), False)  # No spots available
        total_demand = np.zeros(prediction_horizon)
        if hvac_unit is not None:
            hvac_energy_predictions = []
            for temp in predicted_temperatures:
                x = hvac_unit.calculate_energy_required(temp, hvac_unit.var.TRoom) * 1000
                hvac_energy_predictions.append(x)
            hvac_prediction = np.array(hvac_energy_predictions).reshape(1, -1)
            predictions.append(hvac_prediction)
            total_demand += hvac_prediction.flatten()
        else:
            predictions.append(np.zeros((1, prediction_horizon)))
        
        # Forecast energy demand for each bulb
        if lighting is not None:
            for bulb in lighting.bulbs:
                bulb_demand = np.array([bulb.current_energy_usage for _ in range(prediction_horizon)])
                predictions.append(bulb_demand.reshape(1, -1))
                total_demand += bulb_demand.flatten()
           

       
        if ev_charging_spots is not None:
            # Process each charging spot to project its demand based on occupancy and charging rate.
            for i, spot in enumerate(ev_charging_spots):
                if spot.occupied:
                    remaining_time = spot.remaining_charge_time(spot.current_ev)
                    demand_rate = np.clip(spot.current_energy_usage, spot.energy_min, spot.energy_max)
                    demand_until_free = np.full(min(remaining_time, prediction_horizon), demand_rate)
                    spot_demand = np.pad(demand_until_free, (0, prediction_horizon - len(demand_until_free)), 'constant', constant_values=0)
                else:
                    spot_demand = np.zeros(prediction_horizon)
                    spot_availability[i, :] = True

                total_demand += spot_demand
                predictions.append(spot_demand.reshape(1, -1))
                if spot.occupied and remaining_time < prediction_horizon:
                    spot_availability[i, remaining_time:] = True

            # Predicting new vehicle arrivals, considering conditions
            for i, spot in enumerate(ev_charging_spots):
                for t in range(prediction_horizon):
                    if spot_availability[i, t]:
                        additional_capacity = power_capacity - total_demand[t]
                        if (spot.level == 3 and additional_capacity > 30) or (spot.level != 3):
                            estimated_demand_rate = np.random.uniform(spot.energy_min, spot.energy_max)
                            estimated_demand_duration = min(int(60 * (100 - np.random.uniform(20, 80)) / 100), prediction_horizon - t)
                            new_demand_profile = np.zeros(prediction_horizon)
                            new_demand_profile[t:t + estimated_demand_duration] = estimated_demand_rate
                            total_demand[t:t + estimated_demand_duration] += new_demand_profile[t:t + estimated_demand_duration]
                            predictions.append(new_demand_profile.reshape(1, -1))
                            break
       
     
        
        for i, spot in enumerate(ev_charging_spots):
            spot_demand = np.zeros(prediction_horizon)

            if spot.occupied:
                remaining_time = spot.remaining_charge_time(spot.current_ev)
                demand_rate = np.clip(spot.current_energy_usage, spot.energy_min, spot.energy_max)
                spot_demand[:remaining_time] = demand_rate

            # Predict new vehicle arrivals and add to the same spot_demand
            for t in range(prediction_horizon):
                if spot_availability[i, t]:
                    additional_capacity = power_capacity - total_demand[t]
                    if (spot.level == 3 and additional_capacity > 30) or (spot.level != 3):
                        estimated_demand_rate = np.random.uniform(spot.energy_min, spot.energy_max)
                        estimated_demand_duration = min(int(60 * (100 - np.random.uniform(20, 80)) / 100), prediction_horizon - t)
                        spot_demand[t:t + estimated_demand_duration] += estimated_demand_rate
                        total_demand[t:t + estimated_demand_duration] += spot_demand[t:t + estimated_demand_duration]
                        break

            predictions.append(spot_demand.reshape(1, -1))

        # Handle EVs in the queue
        if station:
            for ev in station.queue:
                assigned_spot = None
                for i, spot in enumerate(ev_charging_spots):
                    if spot.level == ev.desired_level:
                        # Check when the spot will be free and available for this EV
                        for t in range(prediction_horizon):
                            if spot_availability[i, t]:
                                additional_capacity = power_capacity - total_demand[t]
                                if additional_capacity > 0:
                                    # Assign the EV to this spot at this time
                                    assigned_spot = spot
                                    estimated_demand_rate = np.random.uniform(spot.energy_min, spot.energy_max)
                                    estimated_demand_duration = min(int(60 * (100 - np.random.uniform(20, 80)) / 100), prediction_horizon - t)
                                    spot_demand[t:t + estimated_demand_duration] += estimated_demand_rate
                                    total_demand[t:t + estimated_demand_duration] += spot_demand[t:t + estimated_demand_duration]
                                    spot_availability[i, t:t + estimated_demand_duration] = False  # Mark the spot as occupied
                                    break
                        if assigned_spot:
                            break  # Move on to the next EV in the queue if a spot has been found

      
        return np.concatenate(predictions, axis=0) if predictions else np.zeros((0, prediction_horizon))


   
    
class MPCController:
    """
        Utilizes Model Predictive Control to allocate energy optimally across HVAC and EV charging stations 
        based on predicted demands.
    """
    def __init__(self, lighting, hvac_unit, ev_charging_station_spots,station,  forecast_model, control_horizon=4, prediction_horizon=8):
        self.hvac_unit = hvac_unit
        self.lighting = lighting
        self.ev_charging_station_spots = ev_charging_station_spots or []
        self.forecast_model = forecast_model
        self.control_horizon = control_horizon
        self.prediction_horizon = prediction_horizon
        self.station = station

  
    def control_step(self, available_power, time_interval):
        """
            Control step in the MPC involves:
            - predicting future demands, 
            - optimizing power allocation
            - updating device states.
            
            This method first calls the forecasting model to predict demands, 
            then sets up and solves a convex optimization problem to distribute available power 
            among active devices in a way that minimizes cost while meeting operational constraints.
        """
        
        # Predict future energy demand over the prediction horizon
        future_demands = self.forecast_model.predict_demand(self.lighting, self.hvac_unit, self.ev_charging_station_spots, self.station, available_power, self.prediction_horizon, time_interval)
        

        # Determine active and passive devices
        active_indices = []
        if self.hvac_unit is not None:
            active_indices.append(0)  # Assuming the first row of future_demands is always for the HVAC unit
        active_indices += [i + 1 for i, spot in enumerate(self.ev_charging_station_spots) if spot.occupied]  # +1 because HVAC is the first if it exists
        if self.lighting is not None:
            bulb_start_idx = len(active_indices)
            active_indices += list(range(bulb_start_idx, bulb_start_idx + len(self.lighting.bulbs)))
        
        filtered_future_demands = future_demands[active_indices, :] if active_indices else np.zeros((0, self.prediction_horizon))
        
        total_indices = range(len(future_demands))  # Indices for all devices
        passive_indices = [i for i in total_indices if i not in active_indices]
        # Filter future_demands for passive devices only
        filtered_passive_demands = future_demands[passive_indices, :] if passive_indices else np.zeros((0, self.prediction_horizon))
        
        # Prepare the list of devices being actively controlled
        devices = []
        if self.hvac_unit is not None:
            devices.append(self.hvac_unit)
         
        devices += [spot for spot in self.ev_charging_station_spots if spot.occupied]
        if self.lighting is not None:
            devices += self.lighting.bulbs
        devices.sort(key=lambda x: x.priority)  # Sort occupied spots by priority

        if any(device.priority for device in devices):
            devices.sort(key=lambda x: x.priority)  # Sort occupied spots by priority
        # Initialize optimization variables for each device for the control horizon

        device_powers = [cp.Variable(self.control_horizon, nonneg=True) for _ in devices]
        constraints = []
        total_cost  = 0
        total_allocated_power = 0  # Total power allocated (which will be maximized)
        operational_weight = 10
        reserve_weight = 0.1  # Weight for cost associated with not meeting reserve power requirements
        penalty_weight = 1000 # Penalty for exceeding the available power

        for t in range(self.control_horizon):
            total_power_at_t = cp.sum([device_powers[device_idx][t] for device_idx in range(len(devices))]) if devices else 0
            constraints.append(total_power_at_t <= available_power)  # Power allocation can't exceed available power

            # Track the total allocated power (which we will maximize)
            total_allocated_power += total_power_at_t

            reserve_shortfall = cp.pos(np.sum(filtered_passive_demands[:, t]) - (available_power - total_power_at_t))
            reserve_penalty = reserve_weight * reserve_shortfall  # Apply reserve penalty (optional)
            total_cost += reserve_penalty
            for device_idx, device in enumerate(devices):
                
                if device.priority:  # Ensure device has a priority for power allocation
                    power = device_powers[device_idx][t]
                    predicted_demand = filtered_future_demands[device_idx, t]

                    constraints.append(power >= device.energy_min)
                    constraints.append(power <= device.energy_max)

                    # Apply operational cost and penalties (optional, but can influence behavior)
                    priority_weight = 1 / device.priority if device.priority else 1
                    deviation_cost = priority_weight * cp.square(power - predicted_demand)  # Cost for deviation from predicted demand
                    operational_cost = operational_weight * priority_weight * cp.abs(power - device.current_energy_usage)  # Cost for operational deviation

                    # You can still accumulate device-specific costs (optional)
                    total_cost += (deviation_cost + operational_cost)

        # Define and solve the optimization problem
        problem = cp.Problem(cp.Maximize(total_allocated_power- total_cost), constraints)
        problem.solve()
        #print(f"Problem Status: {problem.status}")
        total= 0 
        for device_idx, device in enumerate(devices):
            if device.priority:    
                optimal_power = device_powers[device_idx].value[0] if device_powers[device_idx].value is not None else 0
                device.update_state(optimal_power)
                print(device.name, optimal_power)
                total += optimal_power
                
