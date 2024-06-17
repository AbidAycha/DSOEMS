"""
    This module is designed to simulate the energy distribution and consumption in a system 
consisting of HVAC and EV charging station devices. 
    It integrates multiple components including the handling of different power settings (peak and off-peak), 
forecasting models for energy consumption, and controllers for managing power allocation dynamically using
Model Predictive Control (MPC).
    The simulation covers plotting functionalities to visually represent the behavior of HVAC systems and 
EV charging over a given time period. 
"""

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import matplotlib.patches as mpatches
from mpc import  MPCController, ForecastModel
from available_power import AvailablePower



class Simulation:
    """
        Manages the simulation of an energy orchestration system including HVAC and EV charging stations over a specified duration.
        The simulation can use a Model Predictive Control (MPC) approach to optimize energy distribution based on a forecast model.
    """
    def __init__(self, duration_hours, peak_power, off_peak_power, peak_hours, tolerance, use_mpc, EVS, hvac, station,spots=None):
        
        """
            Initializes the Simulation object with all necessary settings and parameters.

            Parameters:
            duration_hours (int): Total duration of the simulation in hours.
            peak_power (float): Maximum power available during peak hours in kW.
            off_peak_power (float): Maximum power available during off-peak hours in kW.
            peak_hours (list): Time intervals that define peak hours.
            tolerance (float): Percentage of the power that is safely usable.
            use_mpc (bool): Flag to determine whether to use MPC for control.
            EVS (list): List of electric vehicles participating in the simulation.
            hvac (HVAC object): HVAC system object.
            station (ChargingStation object): EV charging station object.
        """
        forecast_model=ForecastModel()
        self.duration_hours = duration_hours
        self.time_intervals = duration_hours * 4  # Adjust for your time interval needs
        self.model = hvac
        self.station = station
        self.evs = EVS
        self.tolerance = tolerance  
        self.use_mpc = use_mpc
        self.power = AvailablePower(peak_hours, peak_power, off_peak_power)
        self.initialize_power_consumption()
        if station and self.use_mpc:
            self.controller = MPCController(self.model, spots,forecast_model)
        elif self.use_mpc:
            self.controller = MPCController(self.model, None  ,forecast_model)

    
    def initialize_power_consumption(self):
        """
            Initializes the power consumption arrays for the HVAC and the charging station to track their energy usage.
        """
        if self.model:
            self.model.power_consumption = [0] * (self.time_intervals +1)
        if self.station:
            self.station.power_consumption = [0] * (self.time_intervals +1)

    def plot_simulation_hvac(self, hvac, duration_hours):
        """
            Plots the room temperature and power consumption for the HVAC over the simulation period.
        """
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        
        time_hours = np.linspace(0, duration_hours, len(hvac.roomtemp))
        axs[0].plot(time_hours, hvac.roomtemp, marker='o', linestyle='-', color='blue')
        axs[0].set_ylabel('Room Temperature (°C)')
        axs[0].set_title('Room Temperature vs Time over ' + str(duration_hours) + ' hours')
        hour_ticks = np.arange(0, duration_hours + 1, 1)
        axs[0].set_xticks(hour_ticks)
        axs[0].grid(True)
        
        interval = duration_hours * 4 
        power_availability = [self.power.get_power_for_interval(t) for t in range(0, interval + 1)]
        power_availability_safe = [self.power.get_power_for_interval(t)*(self.tolerance / 100) for t in range(0, interval + 1)]
        time_intervals = np.linspace(0, duration_hours, len(power_availability)) 
        
        axs[1].plot(time_intervals, power_availability, label='Available Power', color='red')
        axs[1].plot(time_intervals, power_availability_safe, label='Safe Operating Power', color='orange')
        axs[1].plot(time_intervals, hvac.power_consumption[:len(power_availability)], label='Power Consumed', color='blue', linestyle='--')
        axs[1].set_xlabel('Time (Hours)')
        axs[1].set_ylabel('Power (W)')
        axs[1].set_title('Power Availability vs. Consumption')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    def plot_simulation_ev(self, evs, station,time_interval):
        """
            Plots the charging status and power consumption for electric vehicles over the simulation period.
        """
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))

        colors = {
        'Level1_charging': 'blue', 'Level1_finished': 'cyan',
        'Level2_charging': 'green', 'Level2_finished': 'lightgreen',
        'Level3_charging': 'red', 'Level3_finished': 'pink',
        'arrival': 'black',
        'waiting': 'grey'}

        legend_patches = [mpatches.Patch(color=colors[f'{level}_{status}'], label=f'{level}_{status}')
                    for level in station.charging_levels for status in [ 'charging', 'finished']]
        legend_patches.append(mpatches.Patch(color='grey', label='waiting'))
        legend_patches.append(mpatches.Patch(color='purple', label='paused'))

        legend_patches.append(Line2D([0], [0], marker='o', color='w', label='arrival',
                                    markerfacecolor='black', markersize=10))
        current_time = time_interval

        for ev in evs:
            color_key = f'{ev.desired_level}_{ev.status}'
            if ev.start_time:  # Check if the EV has started charging
                # Plot the waiting period if there's a gap between arrival and start time
                if ev.arrival < ev.start_time:
                    ax[0].barh(ev.id, width=ev.start_time - ev.arrival, left=ev.arrival,
                            color='gray', edgecolor='black', label='Waiting' if ev.id == evs[0].id else "")
                # Plot the charging period
                ax[0].barh(ev.id, width=ev.end_time - ev.start_time,
                        left=ev.start_time, color=colors[color_key], edgecolor='black')
                if hasattr(ev, 'paused_periods'):
                    for start, end in ev.paused_periods:
                        ax[0].barh(ev.id, width=end - start, left=start, color='purple', edgecolor='black')
            elif ev.status == 'waiting':
                ax[0].barh(ev.id, width=current_time - ev.arrival, left=ev.arrival,
                        color='grey', edgecolor='black',label='Waiting' if ev.id == evs[0].id else "")
            # Plot the arrival marker
            ax[0].plot(ev.arrival, ev.id, 'o', color=colors['arrival'],
                    markersize=10, label='Arrival' if ev.id == evs[0] else "")

        ax[0].xaxis.set_major_locator(MultipleLocator(4))  
        ax[0].xaxis.set_minor_locator(MultipleLocator(1))  
        ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        ax[1].grid(True, which='both', linestyle='--', linewidth=0.5) 
        ax[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/4)}h')) 
        ax[0].set_xlabel('Time Interval (15-minute intervals)')
        ax[0].set_ylabel('EV ID')
        ax[0].set_title('EV Charging Status Over Time')
        ax[0].legend(handles=legend_patches)
        

        power_availability = [self.power.get_power_for_interval(t) for t in range(self.time_intervals)]
        power_availability_safe = [self.power.get_power_for_interval(t)*(self.tolerance/100) for t in range(self.time_intervals)]
        ax[1].plot(power_availability, label='Available Power', color='red')
        ax[1].plot(power_availability_safe, label='Safe Operating Power', color='orange')
        ax[1].plot(station.power_consumption, label='Consumed Power', color='green', linestyle='--')
        ax[1].xaxis.set_major_locator(MultipleLocator(4))
        ax[1].xaxis.set_minor_locator(MultipleLocator(1))
        ax[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/4)}h'))

        ax[1].set_xlabel('Time Interval (15-minute intervals)')
        ax[1].set_ylabel('Power (W)')
        ax[1].set_title('Power Availability vs. Consumption')
        ax[1].legend()
        ax[0].set_xlim([0, self.time_intervals])
        ax[1].set_xlim([0, self.time_intervals])

        plt.tight_layout()
        plt.show()
    def plot_combined_power(self, hvac, station, duration_hours):
        """
            Plots combined power usage of HVAC and EV charging stations compared to available power.
        """
        fig, ax = plt.subplots(figsize=(15, 7))
        
        time_intervals = np.linspace(0, duration_hours, duration_hours * 4 + 1)
        if hvac is not None:
            hvac_power_consumption = np.array(hvac.power_consumption[:len(time_intervals)])
        else:
            hvac_power_consumption = np.zeros(len(time_intervals))
        if station is not None:
            ev_power_consumption = np.array(station.power_consumption[:len(time_intervals)])
        else:
            ev_power_consumption = np.zeros(len(time_intervals))
        total_consumption = hvac_power_consumption + ev_power_consumption
        available_power = np.array([self.power.get_power_for_interval(t) for t in range(len(time_intervals))])
        available_power_safe = np.array([self.power.get_power_for_interval(t)*(self.tolerance/100) for t in range(len(time_intervals))])
        
        # Plotting
        ax.plot(time_intervals, hvac_power_consumption, label='HVAC Consumption', color='blue', linestyle='--')
        ax.plot(time_intervals, ev_power_consumption, label='EV Consumption', color='green', linestyle='--')
        ax.plot(time_intervals, total_consumption, label='Total Consumption', color='grey', linestyle='--')
        ax.plot(time_intervals, available_power, label='Available Power', color='red', linestyle='-')
        ax.plot(time_intervals, available_power_safe, label='Safe Operating Power', color='orange', linestyle='-')
        
        ax.set_xlabel('Time (Hours)')
        ax.set_ylabel('Power (W)')
        ax.set_title('Energy Consumption and Availability Over Time')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """
            Executes the simulation, processing each time interval and managing power allocation.
        """
        for time_interval in range(self.time_intervals):
            available_power = self.power.get_power_for_interval(time_interval) * (self.tolerance/100)
            for ev in self.evs:
                if ev.arrival == time_interval and ev.status == 'waiting':
                    self.station.queue.append(ev)
                    for spot in self.station.spots:
                        if not spot.occupied and spot.level == ev.desired_level:        
                            spot.occupied = True
                            spot.current_ev = ev
                            break
            if self.use_mpc:        
            	self.controller.control_step(available_power/1000)
            if self.model:
                if self.use_mpc:
                    x=self.model.current_energy_usage*1000
                else:
                    x=self.model.calculate_energy_required(self.model.var.TIn,self.model.var.TRoom)*1000
           
                self.model.runModel(x,time_interval,available_power)
                available_power -= self.model.power_consumption[time_interval]
            
                  
                            
            if self.station:       
                self.station.process_evs(available_power, time_interval)
                consumed_power = sum(ev.charging_rate for ev in self.station.charging_evs if ev.status == 'charging') * 1000
                self.station.power_consumption[time_interval] = consumed_power
    
        if self.model:
            self.plot_simulation_hvac(self.model, self.duration_hours)
        if self.station:
            self.plot_simulation_ev(self.evs, self.station,time_interval)
        self.plot_combined_power(self.model, self.station, self.duration_hours)
        
     