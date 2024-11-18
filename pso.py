"""
    This module is integral to an energy management system, implementing Particle Swarm Optimization (PSO)
    for dynamic energy allocation. PSO is a bio-inspired optimization technique that efficiently distributes
    available power among IoT devices such as HVAC systems and EV charging stations.

    The module contains two main classes:
        - ForecastModel3: Predicts future energy demands for HVAC, EV spots, and other devices.
        - PSOController: Optimizes energy allocation using the PSO algorithm, balancing operational costs,
          priorities, and device-specific constraints while maximizing power utilization.
"""

import numpy as np
import pyswarm
from hvac import Model
from hvac import Variables
from ev_station import EV, ChargingStation

# Definition of priorities by level

level_priority = {
    "Level1": 1,
    "Level2": 2,
    "Level3": 3,
}

class PSOController:
    def __init__(self, hvac_unit, ev_charging_station_spots, forecast_model, control_horizon=4, prediction_horizon=8):
        self.hvac_unit = hvac_unit
        self.ev_charging_station_spots = ev_charging_station_spots or []
        self.forecast_model = forecast_model
        self.control_horizon = control_horizon
        self.prediction_horizon = prediction_horizon

    def control_step(self, available_power):
        future_demands = self.forecast_model.predict_demand(self.hvac_unit, self.ev_charging_station_spots, available_power, self.prediction_horizon)

        # Filtering future demands 
        active_indices = []
        if self.hvac_unit is not None:
            active_indices.append(0)  
        active_indices += [i + 1 for i, spot in enumerate(self.ev_charging_station_spots) if spot.occupied]  
        filtered_future_demands = future_demands[active_indices, :] if active_indices else np.zeros((0, self.prediction_horizon))

        total_indices = range(len(future_demands))  
        passive_indices = [i for i in total_indices if i not in active_indices]
        filtered_passive_demands = future_demands[passive_indices, :] if passive_indices else np.zeros((0, self.prediction_horizon))

        # Prepare the list of actively controlled devices
         devices = []
        if self.hvac_unit is not None:
            devices.append(self.hvac_unit)
        devices += [spot for spot in self.ev_charging_station_spots if spot.occupied]
        devices.sort(key=lambda x: x.priority)  # Sort the devices by priority

        if any(device.priority for device in devices):
            devices.sort(key=lambda x: x.priority) 

        def objective_function(powers):
            total_cost = 0
            operational_weight = 10
            reserve_weight = 1
            penalty_weight = 1000

            for t in range(self.control_horizon):
                total_power_at_t = np.sum(powers[t::self.control_horizon])
                power_excess = max(0, total_power_at_t - available_power)
                penalty = penalty_weight * power_excess
                total_cost += penalty

                reserve_shortfall = max(0, np.sum(filtered_future_demands[:, t]) - total_power_at_t)
                reserve_penalty = reserve_weight * reserve_shortfall
                total_cost += reserve_penalty

                for device_idx, device in enumerate(devices):
                    if device.occupied:
                        power = powers[t + device_idx * self.control_horizon]
                        predicted_demand = filtered_future_demands[device_idx, t]
                        priority_weight = 1 / device.priority
                        deviation_cost = priority_weight * (power - predicted_demand) ** 2
                        operational_cost = operational_weight * priority_weight * abs(power - device.current_energy_usage)
                        total_cost += deviation_cost + operational_cost

                        # Add constraints as part of the cost function
                        if power < device.energy_min or power > device.energy_max:
                            total_cost += 1000  
            return total_cost

        num_particles = 20
        num_dimensions = len(devices) * self.control_horizon
        bounds = ([0] * num_dimensions, [available_power] * num_dimensions)

        pso = PSO(num_particles, num_dimensions, bounds, max_iter=30)
        best_powers, _ = pso.optimize(objective_function)

        # Update the states of the devices with the first set of optimal power allocations
        total = 0
        for device_idx, device in enumerate(devices):
            #print("devicespso", devices)
            if device.priority:
                optimal_power = best_powers[device_idx * self.control_horizon]
                device.update_state(optimal_power)
                total += optimal_power

class PSO:
    def __init__(self, num_particles, num_dimensions, bounds, max_iter):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.max_iter = max_iter

    def optimize(self, objective_function):
        lb, ub = self.bounds
        xopt, fopt = pyswarm.pso(objective_function, lb, ub, swarmsize=self.num_particles, maxiter=self.max_iter)
        return xopt, fopt


# Classe pour le modèle de prévision
class ForecastModel3:
    def predict_demand(self, hvac_unit, ev_charging_spots, power_capacity, prediction_horizon):
        predictions = []
        if ev_charging_spots:
            spot_availability = np.full((len(ev_charging_spots), prediction_horizon), False)
        else:
            spot_availability = np.full((0, prediction_horizon), False)

        if hvac_unit is not None:
            x = hvac_unit.calculate_energy_required(hvac_unit.var.TIn,hvac_unit.var.TRoom)*1000
            hvac_prediction = np.full((1, prediction_horizon), x)
        else:
            hvac_prediction = np.zeros((1, prediction_horizon))
        predictions.append(hvac_prediction)

        total_demand = np.zeros(prediction_horizon)

        if ev_charging_spots is not None:
            for i, spot in enumerate(ev_charging_spots):
                if spot.occupied:
                    remaining_time = (spot.remaining_charge_time(spot.current_ev))
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





