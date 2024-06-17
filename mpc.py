"""
    This module is integral to an energy management system, specifically designed for forecasting energy demand
    and optimizing power allocation through Model Predictive Control (MPC). 
    It utilizes the convex optimization library cvxpy to manage and allocate energy resources effectively 
    across different devices such as HVAC units and electric vehicle (EV) charging stations. 
    The module contains two main classes: 
        - ForecastModel, which predicts future energy demands
        - MPCController, which applies MPC techniques to optimize energy usage and minimize cost while
        adhering to operational constraints and priorities.
"""


import cvxpy as cp
import numpy as np

level_priority = {'Level1': 3, 'Level2': 2, 'Level3': 1}

class ForecastModel:
    """
        Forecasts energy demands based on the current state and specifications of HVAC units and EV charging stations.
    """
    def predict_demand(self, hvac_unit, ev_charging_spots, power_capacity, prediction_horizon):
        """
            Predicts energy demand using available data on HVAC settings and EV charging spot statuses.
            This method constructs a time-series prediction over a specified horizon for each device 
        based on its current status and power usage profiles. 
            The result is an array where each row corresponds to a device (HVAC or individual EV spots), 
        and each column represents a future time interval.
        """
        predictions = []
        if ev_charging_spots: 
            spot_availability = np.full((len(ev_charging_spots), prediction_horizon), False)  # Track availability of each spot
        else:
            spot_availability = np.full((0, prediction_horizon), False)  # No spots available

        if hvac_unit is not None:
            x = hvac_unit.calculate_energy_required(hvac_unit.var.TIn,hvac_unit.var.TRoom)*1000
            hvac_prediction = np.full((1, prediction_horizon), x)
        else:    
            hvac_prediction = np.zeros((1, prediction_horizon))
        predictions.append(hvac_prediction)
        # Total demand array for cumulative tracking and power constraint checking
        total_demand = np.zeros(prediction_horizon)

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
      
        return np.concatenate(predictions, axis=0) if predictions else np.zeros((0, prediction_horizon))


    

class MPCController:
    """
        Utilizes Model Predictive Control to allocate energy optimally across HVAC and EV charging stations 
        based on predicted demands.
    """
    def __init__(self, hvac_unit, ev_charging_station_spots, forecast_model, control_horizon=4, prediction_horizon=8):
        self.hvac_unit = hvac_unit
        self.ev_charging_station_spots = ev_charging_station_spots or []
        self.forecast_model = forecast_model
        self.control_horizon = control_horizon
        self.prediction_horizon = prediction_horizon


    def control_step(self, available_power):
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
        future_demands = self.forecast_model.predict_demand(self.hvac_unit, self.ev_charging_station_spots, available_power, self.prediction_horizon)
    

        # Filter future_demands for only the HVAC and occupied EV charging spots
        active_indices = []
        if self.hvac_unit is not None:
            active_indices.append(0)  # Assuming the first row of future_demands is always for the HVAC unit
        active_indices += [i + 1 for i, spot in enumerate(self.ev_charging_station_spots) if spot.occupied]  # +1 because HVAC is the first if it exists
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
        devices.sort(key=lambda x: x.priority)  # Sort occupied spots by priority
        if any(device.priority for device in devices):
            devices.sort(key=lambda x: x.priority)  # Sort occupied spots by priority

        # Initialize optimization variables for each device for the control horizon
        device_powers = [cp.Variable(self.control_horizon, nonneg=True) for _ in devices]

        constraints = []
        total_cost = 0
        operational_weight = 10
        reserve_weight = 1  # Weight for cost associated with not meeting reserve power requirements
        penalty_weight = 1000 # Penalty for exceeding the available power

        for t in range(self.control_horizon):
            total_power_at_t = cp.sum([device_powers[device_idx][t] for device_idx in range(len(devices))]) if devices else 0
            constraints.append(total_power_at_t <= available_power)
            power_excess = cp.pos(total_power_at_t - available_power)
            penalty = penalty_weight * power_excess
            total_cost += penalty

            reserve_shortfall = cp.pos(np.sum(filtered_passive_demands[:, t]) - (available_power - total_power_at_t))
            reserve_penalty = reserve_weight * reserve_shortfall
            total_cost += reserve_penalty

            for device_idx, device in enumerate(devices):
                if device.priority: ## Ensure device has a priority for power allocation
                    power = device_powers[device_idx][t]
                    predicted_demand = filtered_future_demands[device_idx, t]

                    # Enforce power allocation to be within device-specific minimum and maximum limits
                    
                    constraints += [
                        device.energy_min <= power,
                        power <= device.energy_max,
                    ]

                    # Calculate cost components
                    
                    priority_weight = 1 / device.priority  # Priority affects the weighting of the cost
                    deviation_cost = priority_weight * cp.square(power - predicted_demand)  # Cost for deviation from predicted demand
                    operational_cost = operational_weight * priority_weight * cp.abs(power - device.current_energy_usage)   # Cost for operational deviation
                    # Accumulate device-specific costs into total cost
                    total_cost += deviation_cost + operational_cost

        # Define and solve the optimization problem
        problem = cp.Problem(cp.Minimize(total_cost), constraints)
        problem.solve()

        # Update device states with the first set of optimal powers
        total = 0
        for device_idx, device in enumerate(devices):
            if device.priority:    
                optimal_power = device_powers[device_idx].value[0] if device_powers[device_idx].value is not None else 0
                device.update_state(optimal_power)
                total += optimal_power


