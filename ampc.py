"""
    This module is an advanced extension of the energy management system, designed for Adaptive Model Predictive Control (AMPC).
    AMPC dynamically adjusts control horizons, optimization weights, and predictions based on performance feedback and real-time variability.
    It leverages the convex optimization library cvxpy to optimize energy allocation across IoT devices under changing conditions.

    The module contains two main classes:
        - AdaptiveForecastModel, which predicts future energy demands while adapting to deviations and real-time data
        - AMPCController, which enhances traditional MPC by incorporating adaptive mechanisms to optimize energy usage
          and minimize operational costs while maintaining flexibility under dynamic constraints and priorities.
"""

import numpy as np
import cvxpy as cp

class MPCController:
    def __init__(self, lighting, hvac_unit, ev_charging_station_spots, forecast_model, control_horizon=4, prediction_horizon=8):
        self.lighting = lighting
        self.hvac_unit = hvac_unit
        self.ev_charging_station_spots = ev_charging_station_spots or []
        self.forecast_model = forecast_model
        self.control_horizon = control_horizon
        self.prediction_horizon = prediction_horizon

class AMPCController(MPCController):
    def __init__(self, hvac_unit, ev_charging_station_spots, forecast_model, control_horizon=4, prediction_horizon=8):
        super().__init__(hvac_unit, ev_charging_station_spots, forecast_model, control_horizon, prediction_horizon)
        self.recent_power_usages = []
        self.performance_history = []
        self.operational_weight = 10
        self.reserve_weight = 0.1
        self.penalty_weight = 1000


    def update_weights_based_on_performance(self):
        if len(self.performance_history) < 5:
            return  # Not enough data to make an informed adjustment

        average_deviation = np.mean([metrics['deviation'] for metrics in self.performance_history])
        average_shortfall = np.mean([metrics['shortfall'] for metrics in self.performance_history])

        if average_deviation > 10:  # Threshold for high deviation
            self.penalty_weight *= 1.1  # Increase penalty weight by 10%
        else:
            self.penalty_weight *= 0.9  # Decrease penalty weight if performing well

        if average_shortfall > 5:  # Threshold for reserve shortfall
            self.reserve_weight *= 1.1  # Increase reserve weight
        else:
            self.reserve_weight *= 0.9  # Decrease reserve weight

    def adjust_horizon(self):
        if len(self.recent_power_usages) < 2:
            self.control_horizon = 4
            self.prediction_horizon = 8
            return
        variability = np.var(self.recent_power_usages)
        if variability > 50:  # High variability
            self.control_horizon = 2
            self.prediction_horizon = 4
        else:
            self.control_horizon = 4
            self.prediction_horizon = 8
  

    def control_step(self, available_power, time_interval):
        self.adjust_horizon()
        self.update_weights_based_on_performance()
        
        future_demands = self.forecast_model.predict_demand(self.lighting, self.hvac_unit, self.ev_charging_station_spots, available_power, self.prediction_horizon, time_interval)
        self.predicted_demands = future_demands

        active_indices = [0] if self.hvac_unit else []
        active_indices += [i + 1 for i, spot in enumerate(self.ev_charging_station_spots) if spot.occupied]
        if self.lighting is not None:
            bulb_start_idx = len(active_indices)
            active_indices += list(range(bulb_start_idx, bulb_start_idx + len(self.lighting.bulbs)))

        filtered_future_demands = future_demands[active_indices, :] if active_indices else np.zeros((0, self.prediction_horizon))

        total_indices = range(len(future_demands))
        passive_indices = [i for i in total_indices if i not in active_indices]
        filtered_passive_demands = future_demands[passive_indices, :] if passive_indices else np.zeros((0, self.prediction_horizon))

        devices = [self.hvac_unit] if self.hvac_unit else []
        devices += [spot for spot in self.ev_charging_station_spots if spot.occupied]
        if self.lighting is not None:
            devices += self.lighting.bulbs  # Add individual bulbs
        devices.sort(key=lambda x: x.priority)

        device_powers = [cp.Variable(self.control_horizon, nonneg=True) for _ in devices]

        constraints = []
        total_allocated_power = 0 
        total_cost = 0

        for t in range(self.control_horizon):
            total_power_at_t = cp.sum([device_powers[device_idx][t] for device_idx in range(len(devices))]) if devices else 0
            constraints.append(total_power_at_t <= available_power)
            power_excess = cp.pos(total_power_at_t - available_power)
            penalty = self.penalty_weight * power_excess
            total_cost += penalty
            total_allocated_power += total_power_at_t
            reserve_shortfall = cp.pos(np.sum(filtered_passive_demands[:, t]) - (available_power - total_power_at_t))
            reserve_penalty = self.reserve_weight * reserve_shortfall
            total_cost += reserve_penalty

            for device_idx, device in enumerate(devices):
                if device.priority:
                    power = device_powers[device_idx][t]
                    predicted_demand = filtered_future_demands[device_idx, t]

                    constraints.append(power >= device.energy_min)
                    constraints.append(power <= device.energy_max)

                    priority_weight = 1 / device.priority
                    deviation_cost = priority_weight * cp.square(power - predicted_demand)
                    operational_cost = self.operational_weight * priority_weight * cp.abs(power - device.current_energy_usage)
                    total_cost += deviation_cost + operational_cost

        problem = cp.Problem(cp.Maximize(total_allocated_power - total_cost), constraints)
        problem.solve(solver=cp.SCS)

        total= 0 
        for device_idx, device in enumerate(devices):
            if device.priority:    
                optimal_power = device_powers[device_idx].value[0] if device_powers[device_idx].value is not None else 0
                #print(f"Device {device_idx} ({device}): Allocated Power = {optimal_power}")
                device.update_state(optimal_power)
                print(device.name, optimal_power)

                total += optimal_power
        #print("Total Allocated Power:", total)
        #print("available:", available_power)
        self.actual_demands = self.get_actual_demands()
        deviation = np.abs(self.predicted_demands - self.actual_demands).mean()
        shortfall = max(0, np.sum(self.actual_demands) - available_power)
        self.performance_history.append({'deviation': deviation, 'shortfall': shortfall})

        if len(self.performance_history) > 20:
            self.performance_history.pop(0)

    
    def get_actual_demands(self):
        # Simulate actual demands for now
        return self.predicted_demands + np.random.normal(0, 1, size=self.predicted_demands.shape)

    def evaluate_performance(self):
        performance_metrics = {
            "deviation": np.mean(np.abs(self.predicted_demands - self.actual_demands))
        }
        return performance_metrics

    def adapt_based_on_performance(self, metrics):
        if metrics["deviation"] > 1:  # Example threshold
            self.adjust_model_parameters()

    def adjust_model_parameters(self):
        self.forecast_model.update_parameters(self.predicted_demands, self.actual_demands)

    
class AdaptiveForecastModel:
    def __init__(self):
        self.model_parameters = {"param1": 1.0, "param2": 0.5}

    def predict_demand(self, lighting, hvac_unit, ev_charging_spots, power_capacity, prediction_horizon, time_interval):
        # Example prediction logic incorporating HVAC unit and available power
        hvac_demand = np.zeros(prediction_horizon)
        if hvac_unit:
            hvac_demand = hvac_unit.calculate_energy_required(hvac_unit.var.TIn, hvac_unit.var.TRoom) * np.ones(prediction_horizon)

        ev_demands = np.zeros((len(ev_charging_spots), prediction_horizon))
        for i, spot in enumerate(ev_charging_spots):
            if spot.occupied:
                remaining_time = spot.remaining_charge_time(spot.current_ev)
                demand_rate = np.clip(spot.current_energy_usage, spot.energy_min, spot.energy_max)
                ev_demands[i, :remaining_time] = demand_rate
        lighting_demands = []
        if lighting is not None:
            for bulb in lighting.bulbs:
                bulb_demand = np.array([bulb.current_energy_usage for _ in range(prediction_horizon)])
                lighting_demands.append(bulb_demand.reshape(1, -1))
        total_demand = np.vstack([hvac_demand, ev_demands])
        if lighting_demands:
            total_demand = np.vstack([total_demand] + lighting_demands)
        return total_demand

    def update_parameters(self, predicted_demands, actual_demands):
        error = np.mean(np.abs(predicted_demands - actual_demands))
        self.model_parameters["param1"] += 0.1 * error
        self.model_parameters["param2"] -= 0.1 * error
        
        
        
        """import numpy as np
import cvxpy as cp

class MPCController:
    def __init__(self, hvac_unit, ev_charging_station_spots, forecast_model, control_horizon=4, prediction_horizon=8):
        self.hvac_unit = hvac_unit
        self.ev_charging_station_spots = ev_charging_station_spots or []
        self.forecast_model = forecast_model
        self.control_horizon = control_horizon
        self.prediction_horizon = prediction_horizon

class AMPCController(MPCController):
    def __init__(self, hvac_unit, ev_charging_station_spots, forecast_model, control_horizon=4, prediction_horizon=8):
        super().__init__(hvac_unit, ev_charging_station_spots, forecast_model, control_horizon, prediction_horizon)
        self.recent_power_usages = []
        self.performance_history = []
        self.operational_weight = 10
        self.reserve_weight = 1
        self.penalty_weight = 1000


    def update_weights_based_on_performance(self):
        if len(self.performance_history) < 5:
            return  # Not enough data to make an informed adjustment

        average_deviation = np.mean([metrics['deviation'] for metrics in self.performance_history])
        average_shortfall = np.mean([metrics['shortfall'] for metrics in self.performance_history])

        if average_deviation > 10:  # Threshold for significant deviation
            self.penalty_weight *= 1.1  # Increase penalty weight by 10%
        else:
            self.penalty_weight *= 0.9  # Decrease penalty weight if performing well

        if average_shortfall > 5:  # Threshold for reserve shortfall
            self.reserve_weight *= 1.1  # Increase reserve weight
        else:
            self.reserve_weight *= 0.9  # Decrease reserve weight

    def adjust_horizon(self):
        if len(self.recent_power_usages) < 2:
            self.control_horizon = 4
            self.prediction_horizon = 8
            return
        variability = np.var(self.recent_power_usages)
        if variability > 50:  # High variability
            self.control_horizon = 2
            self.prediction_horizon = 4
        else:
            self.control_horizon = 4
            self.prediction_horizon = 8
  

    def control_step(self, available_power, time_interval):
        self.adjust_horizon()
        self.update_weights_based_on_performance()

        future_demands = self.forecast_model.predict_demand(self.hvac_unit, self.ev_charging_station_spots, available_power, self.prediction_horizon, time_interval)
        self.predicted_demands = future_demands

        active_indices = [0] if self.hvac_unit else []
        active_indices += [i + 1 for i, spot in enumerate(self.ev_charging_station_spots) if spot.occupied]
        filtered_future_demands = future_demands[active_indices, :] if active_indices else np.zeros((0, self.prediction_horizon))

        total_indices = range(len(future_demands))
        passive_indices = [i for i in total_indices if i not in active_indices]
        filtered_passive_demands = future_demands[passive_indices, :] if passive_indices else np.zeros((0, self.prediction_horizon))

        devices = [self.hvac_unit] if self.hvac_unit else []
        devices += [spot for spot in self.ev_charging_station_spots if spot.occupied]
        devices.sort(key=lambda x: x.priority)

        device_powers = [cp.Variable(self.control_horizon, nonneg=True) for _ in devices]

        constraints = []
        total_cost = 0

        for t in range(self.control_horizon):
            total_power_at_t = cp.sum([device_powers[device_idx][t] for device_idx in range(len(devices))]) if devices else 0
            constraints.append(total_power_at_t <= available_power)
            power_excess = cp.pos(total_power_at_t - available_power)
            penalty = self.penalty_weight * power_excess
            total_cost += penalty

            reserve_shortfall = cp.pos(np.sum(filtered_passive_demands[:, t]) - (available_power - total_power_at_t))
            reserve_penalty = self.reserve_weight * reserve_shortfall
            total_cost += reserve_penalty

            for device_idx, device in enumerate(devices):
                if device.priority:
                    power = device_powers[device_idx][t]
                    predicted_demand = filtered_future_demands[device_idx, t]

                    constraints += [
                        device.energy_min <= power,
                        power <= device.energy_max,
                    ]

                    priority_weight = 1 / device.priority
                    deviation_cost = priority_weight * cp.square(power - predicted_demand)
                    operational_cost = self.operational_weight * priority_weight * cp.abs(power - device.current_energy_usage)
                    total_cost += deviation_cost + operational_cost

        problem = cp.Problem(cp.Minimize(total_cost), constraints)
        problem.solve()

        for device_idx, device in enumerate(devices):
            if device.priority:
                optimal_power = device_powers[device_idx].value[0] if device_powers[device_idx].value is not None else 0
                device.update_state(optimal_power)

        self.actual_demands = self.get_actual_demands()
        deviation = np.abs(self.predicted_demands - self.actual_demands).mean()
        shortfall = max(0, np.sum(self.actual_demands) - available_power)
        self.performance_history.append({'deviation': deviation, 'shortfall': shortfall})

        if len(self.performance_history) > 20:
            self.performance_history.pop(0)

    
    def get_actual_demands(self):
        # Simulate actual demands for now
        return self.predicted_demands + np.random.normal(0, 1, size=self.predicted_demands.shape)

    def evaluate_performance(self):
        performance_metrics = {
            "deviation": np.mean(np.abs(self.predicted_demands - self.actual_demands))
        }
        return performance_metrics

    def adapt_based_on_performance(self, metrics):
        if metrics["deviation"] > 1:  # Example threshold
            self.adjust_model_parameters()

    def adjust_model_parameters(self):
        self.forecast_model.update_parameters(self.predicted_demands, self.actual_demands)

    
class AdaptiveForecastModel:
    def __init__(self):
        self.model_parameters = {"param1": 1.0, "param2": 0.5}

    def predict_demand(self, hvac_unit, ev_charging_spots, power_capacity, prediction_horizon, time_interval):
        # Example prediction logic incorporating HVAC unit and available power
        hvac_demand = np.zeros(prediction_horizon)
        if hvac_unit:
            hvac_demand = hvac_unit.calculate_energy_required(hvac_unit.var.TIn, hvac_unit.var.TRoom) * np.ones(prediction_horizon)

        ev_demands = np.zeros((len(ev_charging_spots), prediction_horizon))
        for i, spot in enumerate(ev_charging_spots):
            if spot.occupied:
                remaining_time = spot.remaining_charge_time(spot.current_ev)
                demand_rate = np.clip(spot.current_energy_usage, spot.energy_min, spot.energy_max)
                ev_demands[i, :remaining_time] = demand_rate

        total_demand = np.vstack([hvac_demand, ev_demands])
        return total_demand

    def update_parameters(self, predicted_demands, actual_demands):
        error = np.mean(np.abs(predicted_demands - actual_demands))
        self.model_parameters["param1"] += 0.1 * error
        self.model_parameters["param2"] -= 0.1 * error

        """
