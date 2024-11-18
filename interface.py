"""
    This module handles the user interface for the Energy Orchestration Simulation system. 
    It allows users to input settings and parameters for power distribution, manage HVAC, lighting and EV charging station 
    devices, and simulate the energy orchestration based on the provided configurations. 
    The interface is structured to capture input for:
    - peak and off-peak power settings
    - device priorities
    - simulation durations
    - orchestrators

    and view results.

    Components:
    - Peak Power Settings: Input fields for defining peak power, off-peak power, and peak hours.
    - Device Management: Controls to configure HVAC and lighting settings and electric vehicle charging station parameters.
    - Simulation Controls: Options to set the simulation duration and initiate the simulation process.
    - Dynamic Data Entry: Interface allows for the dynamic addition of electric vehicles with specific charging needs.
    - Results Display: Area to view the simulation output and adjust parameters for re-simulation if necessary.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import json
from tkinter import filedialog
from ev_station import EV, ChargingStation
from hvac import Model
from lighting import Lighting
from simulation import Simulation
from PIL import Image, ImageTk

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.label = None
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
    def on_enter(self, event=None):
        if self.tipwindow is None:
            self.tipwindow = tk.Toplevel(self.widget)
            self.tipwindow.wm_overrideredirect(True)
            self.tipwindow.wm_geometry("+0+0")
            self.label = tk.Label(self.tipwindow, text=self.text, justify=tk.LEFT,
                                  background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                                  font=("tahoma", "8", "normal"))
            self.label.pack(ipadx=1)
            self.tipwindow.withdraw()  # Start with the window withdrawn
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow.wm_geometry("+%d+%d" % (x, y))
        self.tipwindow.deiconify()
    def on_leave(self, event=None):
        if self.tipwindow is not None:
            self.tipwindow.withdraw()
            
# Initialize the main window
root0 = tk.Tk()
root0.title("Energy Orchestration Simulation")
root0.geometry("1260x860")
# Tkinter variables for HVAC
hvac_var = tk.BooleanVar()
outside_temp_var = tk.DoubleVar()
desired_temp_var = tk.DoubleVar()
priority_hvac_var = tk.IntVar()
# Tkinter variables for lighting
lighting_var = tk.BooleanVar()
bulb_count_var = tk.IntVar()
lighting_priority_var = tk.IntVar()
# Tkinter variables for EV Station
ev_station_active_var = tk.BooleanVar()
spots_level1_var = tk.IntVar()
priority_level1_var = tk.IntVar()
spots_level2_var = tk.IntVar()
priority_level2_var = tk.IntVar()
spots_level3_var = tk.IntVar()
priority_level3_var = tk.IntVar()

# Tkinter variables for Orchestrator
use_none_var = tk.BooleanVar(value=False)
use_mpc_var = tk.BooleanVar(value=False)  
use_ampc_var = tk.BooleanVar(value=False)  
use_pso_var = tk.BooleanVar(value=False) 
use_fl_var = tk.BooleanVar(value=False)

selection_made_var = tk.BooleanVar(value=False)

# Initialize Tkinter variable for simulation duration
simulation_duration_var = tk.IntVar()

# Track car IDs automatically
next_car_id = 1
car_entries = []
def add_car_section():
    global next_car_id

    car_frame = ttk.Frame(cars_frame)
    car_frame.pack(fill='x', expand=True, padx=5, pady=5)
    
    original_icon = Image.open("trash.png")
    resized_icon = original_icon.resize((32, 32), Image.Resampling.LANCZOS)  # Resize icon to 16x16 pixels
    trash_icon = ImageTk.PhotoImage(resized_icon)
    delete_button = ttk.Button(car_frame, image=trash_icon, command=lambda frame=car_frame: delete_car(frame))
    delete_button.image = trash_icon  
    delete_button.pack(side='right', padx=2)
    
    car_id_label = ttk.Label(car_frame, text="Car ID:")
    car_id_label.pack(side='left', padx=2)
    Tooltip(car_id_label, "Unique identifier for each car.")
    # ttk.Label(car_frame, text="Car ID:").pack(side='left', padx=2)
    car_id_entry = ttk.Entry(car_frame, width=5)
    car_id_entry.pack(side='left', padx=2)
    car_id_entry.insert(0, str(next_car_id))  # Set the car ID automatically
    car_id_entry.configure(state='disabled')  # Disable editing of car ID
    charging_level = ttk.Combobox(car_frame, values=['Level1', 'Level2', 'Level3'], state='readonly', width=5)
    charging_level.set('Level1')  # Set default level
    desired_level_label = ttk.Label(car_frame, text="Desired Level:")
    desired_level_label.pack(side='left', padx=2)
    Tooltip(desired_level_label, "Specifies the charging level (1, 2, or 3) desired for the EV's battery.")
    # ttk.Label(car_frame, text="Desired Level:").pack(side='left', padx=2)
    charging_level.pack(side='left', padx=2)

    arrival_var = tk.IntVar()
    battery_state_var = tk.DoubleVar()
    capacity_var = tk.DoubleVar()
    min_var = tk.DoubleVar()
    max_var = tk.DoubleVar()

    entries = {
        "frame": car_frame,
        "delete_button": delete_button,
        "id_entry": car_id_entry,
        "arrival (each 15min)": ttk.Entry(car_frame, textvariable=arrival_var, width=5),
        "desired_level": charging_level,
        "battery (%)_state": ttk.Entry(car_frame, textvariable=battery_state_var, width=5),
        "capacity (kW)_entry": ttk.Entry(car_frame, textvariable=capacity_var, width=5),
        "min (kW)_rate": ttk.Entry(car_frame, textvariable=min_var, width=5),
        "max (kW)_rate": ttk.Entry(car_frame, textvariable=max_var, width=5)
    }

    labels_descriptions = {
        "arrival (each 15min)": "Entry represents the car's scheduled arrival time, where each unit equals 15 minutes. For example, entering '6' schedules arrival at 01:30.",
        "battery (%)_state": "Current charge level of the EV's battery as a percentage of total capacity.",
        "capacity (kW)_entry": "The total power output capacity of the electric vehicle in kilowatts.",
        "min (kW)_rate": "The lowest speed at which the vehicle's battery can be charged.",
        "max (kW)_rate": "The highest speed at which the vehicle's battery can be charged."
    }
    for label, entry in entries.items():
        if label not in ["id_entry", "desired_level", "frame", "delete_button"]:  # Skip the ID and Desired Level entries
            #ttk.Label(car_frame, text=label.split('_')[0].replace("entry", "").capitalize() + ":").pack(side='left', padx=2)
            text = label.split('_')[0].replace("entry", "").capitalize() + ":"
            entry_label = ttk.Label(car_frame, text=text)
            entry_label.pack(side='left', padx=2)
            Tooltip(entry_label, labels_descriptions[label])
            entry.pack(side='left', padx=2)
    car_entries.append(entries)
    update_charging_level_options()

    next_car_id += 1  # Increment the car ID for the next addition
def delete_car(car_frame):
    global car_entries
    car_entries = [entry for entry in car_entries if entry["frame"] != car_frame]
    car_frame.destroy()
    update_car_ids()

def update_car_ids():
    global next_car_id
    new_id = 1
    for entry in car_entries:
        entry["id_entry"].config(state='normal')
        entry["id_entry"].delete(0, tk.END)
        entry["id_entry"].insert(0, str(new_id))
        entry["id_entry"].config(state='disabled')
        new_id += 1
    next_car_id = new_id

def run_simulation():
    if not validate_inputs():
        return

    duration_hours = simulation_duration_var.get()  # Get the simulation duration from the GUI
    peak_hours = parse_peak_hours(peak_hours_var.get())

    peak_power=int(peak_power_var.get()) 
    off_peak_power=int(off_peak_power_var.get())
    tolerance= int(tolerance_var.get())
    # Check if the Lighting system should be included
    if lighting_var.get() == 1:
        bulb_count = bulb_count_var.get()
        lighting_priority = lighting_priority_var.get()
        lighting_model = Lighting(duration_hours, bulb_count, lighting_priority)
    else:
        lighting_model = None
   
    use_mpc=use_mpc_var.get()
    use_ampc=use_ampc_var.get()
    use_pso=use_pso_var.get()
    use_fl=use_fl_var.get()
   
   # Check if the HVAC system should be included
    if hvac_var.get() == 1:
        outside_temperature = outside_temp_var.get()
        desired_temperature = desired_temp_var.get()
        hvac_priority = priority_hvac_var.get()
        hvac_model = Model(duration_hours, outside_temperature, desired_temperature, energy_min=1, energy_max=4, priority=hvac_priority)
    else:
        hvac_model = None
    # Check if the EVCS system should be included
    if ev_station_active_var.get() == 1:
        charging_levels_config = [
            {'level': 'Level1', 'rate': 2, 'min_energy': 1.2, 'max_energy': 2, 'priority': priority_level1_var.get(), 'spots': spots_level1_var.get()},
            {'level': 'Level2', 'rate': 7.6, 'min_energy': 7.6, 'max_energy': 19.2, 'priority': priority_level2_var.get(), 'spots': spots_level2_var.get()},
            {'level': 'Level3', 'rate': 50, 'min_energy': 30, 'max_energy': 50, 'priority': priority_level3_var.get(), 'spots': spots_level3_var.get()}
        ]

        ev_station = ChargingStation(duration_hours, charging_levels_config)
        cars = []
        for entry in car_entries:
            car_info = {
                'id': int(entry['id_entry'].get()),
                'arrival': int(entry['arrival (each 15min)'].get()),
                'desired_level': entry['desired_level'].get(),
                'battery_state': float(entry['battery (%)_state'].get()),
                'capacity': float(entry['capacity (kW)_entry'].get()),
                'min_rate': float(entry['min (kW)_rate'].get()),
                'max_rate': float(entry['max (kW)_rate'].get()),
            }
            cars.append(EV(**car_info))
        simulation = Simulation(duration_hours, peak_power, off_peak_power, peak_hours, tolerance, use_mpc,use_ampc,use_pso,use_fl, cars, lighting_model, hvac_model, ev_station, ev_station.spots)
    else:
        cars = []
        ev_station = None
        simulation = Simulation(duration_hours, peak_power, off_peak_power, peak_hours, tolerance, use_mpc,use_ampc,use_pso, use_fl,  cars, lighting_model, hvac_model, ev_station)

    simulation.run()


def validate_inputs():
    if peak_power_var.get() <= 0:
        messagebox.showerror("Error", "Peak power must be greater than zero")
        return False
    if off_peak_power_var.get() <= 0:
        messagebox.showerror("Error", "Off-peak power must be greater than zero")
        return False
    if not peak_hours_var.get() and parse_peak_hours(peak_hours_var.get()) is None:
        messagebox.showerror("Error", "Invalid format for peak hours")
        return False
    
    if tolerance_var.get() < 0 or tolerance_var.get() >= 100:
        messagebox.showerror("Error", "Tolerance must be greater than zero and less than 100")
        return False
    if simulation_duration_var.get() <= 0:
        messagebox.showerror("Error", "Simulation duration must be greater than zero")
        return False
    if hvac_var.get() == 1 and priority_hvac_var.get() <= 0:
        messagebox.showerror("Error", "HVAC priority must be greater than zero")
        return False
    if lighting_var.get() == 1 and lighting_priority_var.get() <= 0:
        messagebox.showerror("Error", "Lighting priority must be greater than zero")
        return False
    if lighting_var.get() == 1 and bulb_count_var.get() <= 0:
        messagebox.showerror("Error", "Number of bulbs must be greater than zero else turn off lighting device")
        return False
    if ev_station_active_var.get() == 1:
        if not car_entries:
            messagebox.showerror("Error", "No cars initialized but EV Station is active")
            return False
        for level_var, priority_var in [
            (spots_level1_var, priority_level1_var),
            (spots_level2_var, priority_level2_var),
            (spots_level3_var, priority_level3_var)
        ]:
            if level_var.get() > 0 and priority_var.get() <= 0:
                messagebox.showerror("Error", f"Priority for spots must be greater than zero")
                return False
        for entry in car_entries:
            if not all(entry[field].get() for field in entry if field not in ["id_entry", "delete_button", "frame"]):
                messagebox.showerror("Error", "All car fields must be filled")
                return False
            if float(entry["min (kW)_rate"].get()) >= float(entry["max (kW)_rate"].get()):
                messagebox.showerror("Error", "Minimum rate must be less than maximum rate")
                return False
            if float(entry["battery (%)_state"].get()) >= 100:
                messagebox.showerror("Error", "Battery state must be less than 100")
                return False
    if not selection_made_var.get():
        messagebox.showerror("Error", "Please select whether to use an orchestrator or not.")
        return False
    return True

def toggle_hvac_controls():
    state = 'normal' if hvac_var.get() == 1 else 'disabled'
    hvac_temp_entry.configure(state=state)
    hvac_desired_temp_entry.configure(state=state)
    hvac_priority_entry.configure(state=state)
def toggle_lighting_controls():
    state = 'normal' if lighting_var.get() == 1 else 'disabled'
    bulb_count_entry.configure(state=state)
    lighting_priority_entry.configure(state=state)

def toggle_ev_station_controls():
    state = 'normal' if ev_station_active_var.get() else 'disabled'
    for widget in ev_station_widgets:
        widget.configure(state=state)
    add_car_button.configure(state=state)
    update_charging_level_options()

    # Adjust state for car entry widgets
    for entry in car_entries:
        for widget_name, widget in entry.items():
            if widget_name not in ["id_entry", "frame"]:
                widget.configure(state=state)  # Apply the state change to all car widgets

    # Print debugging statement to ensure this function is called
def update_charging_level_options():
    available_levels = []
    if spots_level1_var.get() > 0:
        available_levels.append('Level1')
    if spots_level2_var.get() > 0:
        available_levels.append('Level2')
    if spots_level3_var.get() > 0:
        available_levels.append('Level3')
    
    # Update all charging level dropdowns
    for entry in car_entries:
        entry['desired_level'].config(values=available_levels)
        if entry['desired_level'].get() not in available_levels:
            entry['desired_level'].set(available_levels[0] if available_levels else '')

def update_spot_priority_state(spot_var, priority_entry):
    priority_entry.configure(state='normal' if spot_var.get() > 0 else 'disabled')
    update_charging_level_options()  # Call this function to update options

ev_station_widgets = []

def add_spot_level_configuration(parent, level, spots_var, priority_var, level_description):
    frame = ttk.Frame(parent)
    frame.pack(fill='x')
    level_label = ttk.Label(frame, text=f"Number of spots for {level}:")
    level_label.pack(side='left', padx=2)
    	
    Tooltip(level_label, level_description)
    spot_entry = ttk.Entry(frame, textvariable=spots_var, width=5)
    spot_entry.pack(side='left', padx=2)
    spot_priority_label = ttk.Label(frame, text="Priority:")
    spot_priority_label.pack(side='left', padx=2)
    Tooltip(spot_priority_label, f"Priority of {level} spots, influencing their energy allocation relative to other levels.")
    priority_entry = ttk.Entry(frame, textvariable=priority_var, width=5)
    priority_entry.pack(side='left', padx=2)
    priority_entry.configure(state='disabled')  # Priority is disabled by default

    ev_station_widgets.extend([spot_entry, priority_entry])
    spots_var.trace_add('write', lambda *args, sv=spots_var, pe=priority_entry: update_spot_priority_state(sv, pe))
    update_spot_priority_state(spots_var, priority_entry)


def parse_peak_hours(peak_hours_str):
    if not peak_hours_str.strip():
        return []
    try:
        hours_list = peak_hours_str.split(',')
        peak_hours = []
        for hours in hours_list:
            start_end = tuple(map(int, hours.strip().split('-')))
            if len(start_end) != 2:
                raise ValueError("Each interval must have exactly two numbers (start and end).")
            peak_hours.append(start_end)
        return peak_hours
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid format for peak hours: {e}")
        return None




def select_option(button, use_var):
    # Reset all buttons to default colors
    none_button.config(bg='light gray', fg='black')
    mpc_button.config(bg='light gray', fg='black')
    ampc_button.config(bg='light gray', fg='black')
    pso_button.config(bg='light gray', fg='black')
    
    fl_button.config(bg='light gray', fg='black')

    # Set the selected button's colors
    button.config(bg='lightblue', fg='white')

    # Reset all variables to False
    use_mpc_var.set(False)
    use_ampc_var.set(False)
    use_pso_var.set(False)
    use_fl_var.set(False)

    # Set the corresponding variable to True
    use_var.set(True)

    # Indicate a selection has been made
    selection_made_var.set(True)
      
def populate_default_values():
    # Setting default power settings
    peak_power_var.set(100) 
    off_peak_power_var.set(80) 
    peak_hours_var.set("6-10") 
    tolerance_var.set(80)  
    """
    # Default HVAC settings
    hvac_var.set(1)  # Enable HVAC
    outside_temp_var.set(25.0) 
    desired_temp_var.set(22.0)  
    priority_hvac_var.set(5)  """

    # Default EV Station settings
    ev_station_active_var.set(1)  # Activate EV Station
    spots_level1_var.set(1)
    priority_level1_var.set(2)
    spots_level2_var.set(0)
    priority_level2_var.set(0)
    spots_level3_var.set(2)
    priority_level3_var.set(1)

    simulation_duration_var.set(12)
    
    # EV specifications for default data
    ev_specifications = [
        {"arrival": 4, "battery_state": 70, "capacity": 100, "min_rate": 1, "max_rate": 2, "level": 'Level1'},
        {"arrival": 6, "battery_state": 20, "capacity": 150, "min_rate": 35, "max_rate": 45, "level": 'Level3'},
        {"arrival": 8, "battery_state": 15, "capacity": 100, "min_rate": 30, "max_rate": 40, "level": 'Level3'},
        {"arrival": 20, "battery_state": 2, "capacity": 200, "min_rate": 40, "max_rate": 50, "level": 'Level3'}
    ]

    # Add cars based on specifications
    for ev in ev_specifications:
        add_car_section()  # Add a car section
        car_index = len(car_entries) - 1  # Get index of the newly added car
        car_entries[car_index]['arrival (each 15min)'].insert(0, str(ev["arrival"]))
        car_entries[car_index]['battery (%)_state'].insert(0, str(ev["battery_state"]))
        car_entries[car_index]['capacity (kW)_entry'].insert(0, str(ev["capacity"]))
        car_entries[car_index]['min (kW)_rate'].insert(0, str(ev["min_rate"]))
        car_entries[car_index]['max (kW)_rate'].insert(0, str(ev["max_rate"]))
        car_entries[car_index]['desired_level'].set(ev["level"])


def load_settings_from_file():
    filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])  # Prompt user to select a JSON file
    if not filepath:
        return  # Exit if no file is selected

    with open(filepath, 'r') as file:
        data = json.load(file)

    # Example of how to populate GUI components from file data
    peak_power_var.set(data['peak_power'])
    off_peak_power_var.set(data['off_peak_power'])
    peak_hours_var.set(data['peak_hours'])
    tolerance_var.set(data['tolerance'])

    hvac_var.set(data['hvac']['active'])
    outside_temp_var.set(data['hvac']['outside_temperature'])
    desired_temp_var.set(data['hvac']['desired_temperature'])
    priority_hvac_var.set(data['hvac']['priority'])

    lighting_var.set(data['lighting']['active'])
    bulb_count_var.set(data['lighting']['bulb_count'])
    lighting_priority_var.set(data['lighting']['priority'])

    ev_station_active_var.set(data['ev_station']['active'])
    spots_level1_var.set(data['ev_station']['level1']['spots'])
    priority_level1_var.set(data['ev_station']['level1']['priority'])
    spots_level2_var.set(data['ev_station']['level2']['spots'])
    priority_level2_var.set(data['ev_station']['level2']['priority'])
    spots_level3_var.set(data['ev_station']['level3']['spots'])
    priority_level3_var.set(data['ev_station']['level3']['priority'])

    simulation_duration_var.set(data['simulation_duration'])

    # Remove existing cars and re-add from file
    for entry in car_entries[:]:
        delete_car(entry['frame'])

    for car in data['cars']:
        add_car_section()
        car_index = len(car_entries) - 1
        car_entries[car_index]['arrival (each 15min)'].insert(0, str(car["arrival"]))
        car_entries[car_index]['battery (%)_state'].insert(0, str(car["battery_state"]))
        car_entries[car_index]['capacity (kW)_entry'].insert(0, str(car["capacity"]))
        car_entries[car_index]['min (kW)_rate'].insert(0, str(car["min_rate"]))
        car_entries[car_index]['max (kW)_rate'].insert(0, str(car["max_rate"]))
        car_entries[car_index]['desired_level'].set(car["level"])

    toggle_hvac_controls()
    toggle_ev_station_controls()


# Create a main frame
main_frame = ttk.Frame(root0)
main_frame.pack(fill=tk.BOTH, expand=1)

# Create a Canvas
canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Add a scrollbar to the Canvas
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas
canvas.configure(yscrollcommand=scrollbar.set)
#canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create another frame inside the Canvas
root = ttk.Frame(canvas)
root.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Add that new frame to a window in the canvas
canvas.create_window((0, 0), window=root, anchor="nw")




toolbar_frame = ttk.Frame(root)
toolbar_frame.pack(side='top', fill='x', padx=10, pady=5)
right_frame = ttk.Frame(toolbar_frame)
right_frame.pack(side='right', padx=10, pady=5)  # This frame aligns to the right

# Load an icon for the button
icon_path = 'upload.png'  # Update this path to where your actual icon file is stored
icon_image = Image.open(icon_path)
icon_photo = ImageTk.PhotoImage(icon_image.resize((32, 32), Image.Resampling.LANCZOS))  # Resize the icon as needed

# Create the button and add it to the right-aligned frame
input_file_button = ttk.Button(right_frame, image=icon_photo, command=load_settings_from_file)
input_file_button.image = icon_photo  # keep a reference to the image to avoid garbage collection
input_file_button.pack()

power_frame = ttk.LabelFrame(root, text="Available Power Settings", padding=(10, 5))
power_frame.pack(fill='x', padx=10, pady=5)


# Peak Power Entry
peak_power_label = ttk.Label(power_frame, text="Peak Power (kW):")
peak_power_label.pack(side='left', padx=2)
Tooltip(peak_power_label, "The maximum electrical power available during peak usage hours.")
peak_power_var = tk.IntVar()  # Default value
peak_power_entry = ttk.Entry(power_frame, textvariable=peak_power_var, width=10)
peak_power_entry.pack(side='left', padx=2)

# Off-Peak Power Entry
off_peak_power_label = ttk.Label(power_frame, text="Off-Peak Power (kW):")
off_peak_power_label.pack(side='left', padx=2)
Tooltip(off_peak_power_label, "The maximum electrical power available during off-peak hours, typically lower-cost.")
off_peak_power_var = tk.IntVar()  # Default value
off_peak_power_entry = ttk.Entry(power_frame, textvariable=off_peak_power_var, width=10)
off_peak_power_entry.pack(side='left', padx=2)

# Peak Hours Entry
peak_hours_label = ttk.Label(power_frame, text="Peak Hours (e.g., 6-10,18-21):")
peak_hours_label.pack(side='left', padx=2)
Tooltip(peak_hours_label, "Specific hours during which energy demand and cost are highest, usually during morning and evening.")
peak_hours_var = tk.StringVar()  # Default value
peak_hours_entry = ttk.Entry(power_frame, textvariable=peak_hours_var, width=20)
peak_hours_entry.pack(side='left', padx=2)


# Tolerance Entry
tolerance_label = ttk.Label(power_frame, text="Tolerance (%):")
tolerance_label.pack(side='left', padx=2)
Tooltip(tolerance_label, "Set the Tolerance (%) to limit the usage of total available power. This acts as a safety buffer by specifying what percentage of the total power can be utilized, reducing the risk of overloading the system.")
tolerance_var = tk.IntVar()  # Default value
tolerance_entry = ttk.Entry(power_frame, textvariable=tolerance_var, width=10)
tolerance_entry.pack(side='left', padx=2)





device_frame = ttk.LabelFrame(root, text="Devices", padding=(10, 5))
device_frame.pack(fill='x', padx=10, pady=5)

hvac_frame = ttk.Frame(device_frame)
hvac_frame.pack(fill='x')
hvac_label = ttk.Label(hvac_frame, text="HVAC:")
hvac_label.pack(side='left', padx=2)
Tooltip(hvac_label, "Heating, Ventilation, and Air Conditioning system that regulates indoor temperature and air quality.")
ttk.Radiobutton(hvac_frame, text="Yes", variable=hvac_var, value=1).pack(side='left', padx=2)
ttk.Radiobutton(hvac_frame, text="No", variable=hvac_var, value=0).pack(side='left', padx=2)
outside_temp_label = ttk.Label(hvac_frame, text="Outside Temp:")
outside_temp_label.pack(side='left', padx=2)
Tooltip(outside_temp_label, "Current or typical temperature outside, used to adjust HVAC settings.")
hvac_temp_entry = ttk.Entry(hvac_frame, textvariable=outside_temp_var, width=5)
hvac_temp_entry.pack(side='left', padx=2)
desired_temp_label = ttk.Label(hvac_frame, text="Desired Temp:")
desired_temp_label.pack(side='left', padx=2)
Tooltip(desired_temp_label, "Target indoor temperature that the HVAC system aims to achieve.")

hvac_desired_temp_entry = ttk.Entry(hvac_frame, textvariable=desired_temp_var, width=5)
hvac_desired_temp_entry.pack(side='left', padx=2)

hvac_priority_label = ttk.Label(hvac_frame, text="Priority:")
hvac_priority_label.pack(side='left', padx=2)
Tooltip(hvac_priority_label, "Priority level of the HVAC system, influencing its energy allocation relative to other systems")

hvac_priority_entry = ttk.Entry(hvac_frame, textvariable=priority_hvac_var, width=5)
hvac_priority_entry.pack(side='left', padx=2)
hvac_var.trace_add('write', lambda *args: toggle_hvac_controls())
############################
lighting_frame = ttk.Frame(device_frame)
lighting_frame.pack(fill='x', pady=5)

lighting_label = ttk.Label(lighting_frame, text="Lighting:")
lighting_label.pack(side='left', padx=2)
Tooltip(lighting_label, "Control the lighting system, including the number of bulbs.")

ttk.Radiobutton(lighting_frame, text="Yes", variable=lighting_var, value=1).pack(side='left', padx=2)
ttk.Radiobutton(lighting_frame, text="No", variable=lighting_var, value=0).pack(side='left', padx=2)

bulb_count_label = ttk.Label(lighting_frame, text="Number of Bulbs:")
bulb_count_label.pack(side='left', padx=2)
Tooltip(bulb_count_label, "Enter the total number of light bulbs managed by the system.")

bulb_count_entry = ttk.Entry(lighting_frame, textvariable=bulb_count_var, width=5)
bulb_count_entry.pack(side='left', padx=2)



lighting_priority_label = ttk.Label(lighting_frame, text="Priority:")
lighting_priority_label.pack(side='left', padx=2)
Tooltip(lighting_priority_label, "Set the priority of lighting compared to other devices.")

lighting_priority_entry = ttk.Entry(lighting_frame, textvariable=lighting_priority_var, width=5)
lighting_priority_entry.pack(side='left', padx=2)

lighting_var.trace_add('write', lambda *args: toggle_lighting_controls())

#############################
ev_station_frame = ttk.Frame(device_frame)
ev_station_frame.pack(fill='x', pady=5)
ev_station_label = ttk.Label(ev_station_frame, text="EV Station:")
ev_station_label.pack(side='left', padx=2)
	
Tooltip(ev_station_label, "Electric Vehicle charging station, equipped with multiple charging levels for faster or slower charge times.")
ttk.Radiobutton(ev_station_frame, text="Yes", variable=ev_station_active_var, value=1).pack(side='left', padx=2)
ttk.Radiobutton(ev_station_frame, text="No", variable=ev_station_active_var, value=0).pack(side='left', padx=2)
ev_station_active_var.trace_add('write', lambda *args: toggle_ev_station_controls())

add_spot_level_configuration(ev_station_frame, "Level1", spots_level1_var, priority_level1_var, "Number of basic charging spots.They offer a gentle range of 1.2 to 2 kW. Suitable for slow charging over extended periods.")
add_spot_level_configuration(ev_station_frame, "Level2", spots_level2_var, priority_level2_var, "Number of intermediate charging spots. Offers faster charging  from 7.6 to 19.2 kW, suitable for quicker turnarounds.")
add_spot_level_configuration(ev_station_frame, "Level3", spots_level3_var, priority_level3_var, "Number of advanced charging spots. Provides the fastest charging speeds for rapid top-ups, ranging from 30 to 50 kW.")

duration_frame = ttk.LabelFrame(root, text="Simulation Duration (in hours)", padding=(10, 5))
duration_frame.pack(fill='x', padx=10, pady=5)
ttk.Entry(duration_frame, textvariable=simulation_duration_var).pack(fill='x', expand=True, padx=2)

cars_frame = ttk.LabelFrame(root, text="Cars Coming", padding=(10, 5))
cars_frame.pack(fill='x', padx=10, pady=5)
add_car_button = ttk.Button(cars_frame, text="Add Car", command=add_car_section)
add_car_button.pack(side='top', pady=5)


style = ttk.Style()

# Define styles for normal and selected buttons
style.configure('TButton', font=('Helvetica', 10), background='gray')  # Default button color
style.configure('Selected.TButton', font=('Helvetica', 10), background='lightblue')  # Color for selected button

orchestrator_frame = ttk.LabelFrame(root, text="Choose Orchestrator", padding=(10, 5))
orchestrator_frame.pack(fill='x', padx=10, pady=5)


none_button = tk.Button(orchestrator_frame, text="None", command=lambda:select_option(none_button, use_none_var))
none_button.pack(side='left', padx=5)

mpc_button = tk.Button(orchestrator_frame, text="Model Predictive Control", command=lambda:select_option(mpc_button, use_mpc_var))
mpc_button.pack(side='left', pady=5)

ampc_button = tk.Button(orchestrator_frame, text="Adaptive MPC", command=lambda:select_option(ampc_button, use_ampc_var))
ampc_button.pack(side='left', pady=5)

pso_button = tk.Button(orchestrator_frame, text="Particle Swarm Optimization", command=lambda:select_option(pso_button, use_pso_var))
pso_button.pack(side='left', pady=5)

fl_button = tk.Button(orchestrator_frame, text="Fuzzy Logic", command=lambda:select_option(fl_button, use_fl_var))
fl_button.pack(side='left', pady=5)





submit_button = ttk.Button(root, text="Start Simulation", command=run_simulation)
submit_button.pack(side='bottom', pady=10)



# return when we remove populate_default_values()
#add_car_section() 

toggle_hvac_controls()
toggle_ev_station_controls()  # Call to set initial states based on the above line

populate_default_values() # when removed return the add_car_section() above




root.mainloop()
