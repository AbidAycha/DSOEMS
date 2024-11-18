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

from ampc import AMPCController, AdaptiveForecastModel
from pso import ForecastModel3, PSOController
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from fl import FuzzyController, ForecastModel4
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
    def __init__(self, duration_hours, peak_power, off_peak_power, peak_hours, tolerance, use_mpc,use_ampc,use_pso, use_fl,  EVS, lighting, hvac, station,spots=None):

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
        forecast_model1=ForecastModel3()
        forecast_model2= AdaptiveForecastModel()
        forecast_model3=ForecastModel4()
        self.duration_hours = duration_hours
        self.time_intervals = duration_hours * 4  # Adjust for your time interval needs
        self.model = hvac
        self.lighting = lighting
        self.station = station
       

        self.evs = EVS
        self.tolerance = tolerance  
        self.use_mpc = use_mpc
        self.use_ampc = use_ampc
        self.use_pso = use_pso
        self.use_fl = use_fl
        self.power = AvailablePower(peak_hours, peak_power, off_peak_power)
        self.initialize_power_consumption()
        self.spots= spots
        if station and self.use_mpc:
            self.controller = MPCController(self.lighting, self.model, spots,station, forecast_model)
        elif self.use_mpc:
            self.controller = MPCController(self.lighting, self.model, None  ,None, forecast_model)

        elif station and self.use_pso:
            self.controller = PSOController(self.model, spots, forecast_model1)

        elif self.use_pso:
            self.controller = PSOController(self.model, None, forecast_model1)
        elif station and self.use_fl:
             self.controller = FuzzyController(self.model, spots, forecast_model3)
        elif self.use_fl:
             self.controller = FuzzyController(self.model, None, forecast_model3)
        elif station and self.use_ampc:
            self.controller = AMPCController(self.lighting, self.model, spots, forecast_model2)

        elif self.use_ampc:
            self.controller = AMPCController(self, lighting, self.model, None, forecast_model2)
    
    def initialize_power_consumption(self):
        """
            Initializes the power consumption arrays for the HVAC and the charging station to track their energy usage.
        """
        if self.model:
            self.model.power_consumption = [0] * (self.time_intervals +1)
        if self.station:
            self.station.power_consumption = [0] * (self.time_intervals +1)
        if self.lighting:
            self.lighting.power_consumption = [0] * (self.time_intervals +1)

    def plot_simulation_hvac(self, hvac, duration_hours):
        """
            Plots the room temperature and power consumption for the HVAC over the simulation period.
        """
        plt.rcParams['figure.figsize'] = (17.28, 9.72)
        fig, axs = plt.subplots(2, 1)
        
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
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.wm_geometry("+0+0") 
        
        plt.show()
    def plot_simulation_ev_metrics(self, evs, station, time_interval): 
        """
            Plots the charging status and power consumption for electric vehicles over the simulation period.
            Also calculates and prints the requested statistics.
        """
        plt.rcParams['figure.figsize'] = (17.28, 9.72)
    
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        colors = {
            'Level1_charging': 'blue', 'Level1_finished': 'cyan', 'Level1_paused': 'purple',
            'Level2_charging': 'green', 'Level2_finished': 'lightgreen', 'Level2_paused': 'purple',
            'Level3_charging': 'red', 'Level3_finished': 'pink', 'Level3_paused': 'purple',
            'arrival': 'black',
            'waiting': 'grey'
        }


        legend_patches = [mpatches.Patch(color=colors[f'{level}_{status}'], label=f'{level}_{status}')
                        for level in station.charging_levels for status in ['charging', 'finished']]
        legend_patches.append(mpatches.Patch(color='grey', label='waiting'))
        legend_patches.append(mpatches.Patch(color='purple', label='paused'))

        legend_patches.append(Line2D([0], [0], marker='o', color='w', label='arrival',
                                    markerfacecolor='black', markersize=10))

        current_time = time_interval

        # Initialize metrics
        total_waiting_time = 0
        total_paused_time = 0
        max_waiting_time = 0
        total_charging_time = 0
        num_evs_charged = 0
        total_charging_time_per_ev = []

        for ev in evs:
            color_key = f'{ev.desired_level}_{ev.status}'
            waiting_time = 0

            if ev.start_time and ev.arrival:  # Check if the EV has started charging
                # Calculate waiting time if there's a gap between arrival and start time
                if ev.arrival < ev.start_time:
                    waiting_time = ev.start_time - ev.arrival
                    # print("here2", waiting_time)
                    total_waiting_time += waiting_time
                    max_waiting_time = max(max_waiting_time, waiting_time)
                    ax1.barh(ev.id, width=waiting_time, left=ev.arrival,
                            color='gray', edgecolor='black', label='Waiting' if ev.id == evs[0].id else "")
                if ev.end_time:
                    #print("end", ev.end_time, ev.start_time)
                    # Calculate charging time and update totals
                    charging_time = ev.end_time - ev.start_time
                    total_charging_time += charging_time
                    total_charging_time_per_ev.append(charging_time)

                # Only consider EVs that have finished charging
                if ev.status == 'finished':
                    num_evs_charged += 1

                # Plot the charging period
                ax1.barh(ev.id, width=charging_time, left=ev.start_time, color=colors[color_key], edgecolor='black')

                # Plot paused periods if any
                if ev.paused_periods:
                   
                    for start, end in ev.paused_periods:
                        
                        if not end:
                            end = time_interval
                        paused_time = end - start
                        total_paused_time += paused_time
                        #print("paused", paused_time, total_paused_time)
                        ax1.barh(ev.id, width=end - start, left=start, color='purple', edgecolor='black')

            elif ev.status == 'waiting' and ev.arrival < current_time:
                waiting_time = current_time - ev.arrival
                # print("here", waiting_time)
                total_waiting_time += waiting_time
                max_waiting_time = max(max_waiting_time, waiting_time)
                ax1.barh(ev.id, width=waiting_time, left=ev.arrival,
                        color='grey', edgecolor='black', label='Waiting' if ev.id == evs[0].id else "")

            # Plot the arrival marker
            ax1.plot(ev.arrival, ev.id, 'o', color=colors['arrival'], markersize=10, label='Arrival' if ev.id == evs[0].id else "")

        # Average Charging Time
        avg_charging_time = total_charging_time / num_evs_charged if num_evs_charged > 0 else 0
        avg_waiting_time = total_waiting_time / num_evs_charged if num_evs_charged > 0 else 0
        # Print calculated statistics
        
        print(f"Total Waiting Time: {total_waiting_time:.2f} units")
        print(f"Average Waiting Time: {avg_waiting_time:.2f} units")
        print(f"Maximum Waiting Time: {max_waiting_time:.2f} units")
        print(f"Average Charging Time: {avg_charging_time:.2f} units")
        print(f"Total Charging Time: {total_charging_time:.2f} units")
        print(f"Number of EVs Charged: {num_evs_charged}")
    
        # Plot settings for the charging status
        ax1.xaxis.set_major_locator(MultipleLocator(4))
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/4)}h'))
        ax1.set_xlabel('Time Interval (15-minute intervals)')
        ax1.set_ylabel('EV ID')
        ax1.set_title('EV Charging Status Over Time')
        ax1.legend(handles=legend_patches)

        # Plot the power availability and consumption
        power_availability = [self.power.get_power_for_interval(t) for t in range(self.time_intervals + 1 )]
        power_availability_safe = [self.power.get_power_for_interval(t) * (self.tolerance / 100) for t in range(self.time_intervals + 1 )]
        ax2.plot(power_availability, label='Available Power', color='red')
        ax2.plot(power_availability_safe, label='Safe Operating Power', color='orange')
        ax2.plot(station.power_consumption, label='Consumed Power', color='green', linestyle='--')
        ax2.xaxis.set_major_locator(MultipleLocator(4))
        ax2.xaxis.set_minor_locator(MultipleLocator(1))
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/4)}h'))

        ax2.set_xlabel('Time Interval (15-minute intervals)')
        ax2.set_ylabel('Power (W)')
        ax2.set_title('Power Availability vs. Consumption')
        ax2.legend()
        ax1.set_xlim([0, self.time_intervals])
        ax2.set_xlim([0, self.time_intervals])
        stats_text = (
        f"Total Waiting Time: {total_waiting_time:.2f} units\n"
        f"Avg Waiting Time: {avg_waiting_time:.2f} units\n"
        f"Max Waiting Time: {max_waiting_time:.2f} units\n"
        f"Total Charging Time: {total_charging_time:.2f} units\n"
        f"Avg Charging Time: {avg_charging_time:.2f} units\n"
        f"Number of EVs Charged: {num_evs_charged}"
        )
        # Add text box to ax2
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        
        
     
       

        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.wm_geometry("+0+0")

        plt.show()
    
    def plot_combined_power(self, lighting, hvac, station, duration_hours):
        """
            Plots combined power usage of HVAC and EV charging stations compared to available power.
        """
        plt.rcParams['figure.figsize'] = (17.28, 9.72)
        fig, ax = plt.subplots()
        
        time_intervals = np.linspace(0, duration_hours, duration_hours * 4 + 1)
        if hvac is not None:
            hvac_power_consumption = np.array(hvac.power_consumption[:len(time_intervals)])
        else:
            hvac_power_consumption = np.zeros(len(time_intervals))
        if station is not None:
            ev_power_consumption = np.array(station.power_consumption[:len(time_intervals)])
        else:
            ev_power_consumption = np.zeros(len(time_intervals))
        if lighting is not None:
            lighting_power_consumption = np.array(lighting.power_consumption[:len(time_intervals)])
        else:
            lighting_power_consumption = np.zeros(len(time_intervals))
        total_consumption = hvac_power_consumption + ev_power_consumption + lighting_power_consumption 
        available_power = np.array([self.power.get_power_for_interval(t) for t in range(len(time_intervals))])
        available_power_safe = np.array([self.power.get_power_for_interval(t)*(self.tolerance/100) for t in range(len(time_intervals))])
        
        # Plotting
        ax.plot(time_intervals, hvac_power_consumption, label='HVAC Consumption', color='blue', linestyle='--')
        ax.plot(time_intervals, lighting_power_consumption, label='Lighting Consumption', color='violet', linestyle='--')
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
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.wm_geometry("+0+0") 
        
        plt.show()
    def run(self):
        """
            Executes the simulation, processing each time interval and managing power allocation.
        """
        
        for time_interval in range(self.time_intervals + 1 ):
            available_power = self.power.get_power_for_interval(time_interval) * (self.tolerance/100)
            for ev in self.evs:
                
                if ev.arrival == time_interval and ev.status == 'waiting':
                    #print("here4")
                    self.station.queue.append(ev)
                    for spot in self.spots:
                        
                        if not spot.occupied and spot.level == ev.desired_level:        
                            spot.occupied = True
                            spot.current_ev = ev
                            break
                    #print("here", spot, spot.occupied)    
            if self.use_mpc or self.use_ampc:        
            	self.controller.control_step(available_power/1000, time_interval)
            elif self.use_pso or self.use_fl:
                self.controller.control_step(available_power/1000)
                
            if self.model:
                if self.use_mpc or self.use_pso or self.use_ampc or self.use_fl:
                    x=self.model.current_energy_usage*1000
                else:
                    x=self.model.calculate_energy_required(self.model.var.TIn,self.model.var.TRoom)*1000
           
                self.model.runModel(x,time_interval,available_power)
                available_power -= self.model.power_consumption[time_interval]
            
            if self.lighting:
                if not (self.use_mpc or self.use_pso or self.use_ampc or self.use_fl):
                    self.lighting.allocate_energy(0.1)
                self.lighting.calculate_total_power_usage(time_interval)  
                available_power -= self.lighting.power_consumption[time_interval]
                            
            if self.station:    
                #print("time_interval simulation", time_interval)   
                self.station.process_evs(available_power, time_interval)
                consumed_power = sum(ev.charging_rate for ev in self.station.charging_evs if ev.status == 'charging') * 1000
                self.station.power_consumption[time_interval] = consumed_power
                #print(time_interval, "station", self.station.power_consumption[time_interval])
        
        current_time = self.duration_hours * 4 - 1
        if self.model:
            self.plot_simulation_hvac(self.model, self.duration_hours)
            
        if self.station:
            self.plot_simulation_ev_metrics(self.evs, self.station,current_time)
        self.plot_combined_power(self.lighting, self.model, self.station, self.duration_hours)
        
        
        
    
    
    
