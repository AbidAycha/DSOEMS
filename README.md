## DSOEMS-IoT (Dependable Software-Optimized Energy Management System for IoT enabled smart buildings)

Welcome to the DSOEMS-IoT project! This system is designed to simulate and optimize energy distribution among HVAC systems, electric vehicle (EV) charging stations, and lighting devices within a smart building. Whether you’re exploring energy flows, testing control and optimization strategies, or simulating various management scenarios, this tool is for you.

### Features

- **Simulation Interface**: Intuitive interface for setting up and monitoring energy distribution simulations.
- **Advanced Orchestrators**:
  - **MPC (Model Predictive Control)**: Predictive, real-time optimization of energy distribution.
  - **AMPC (Adaptive MPC)**: Dynamically adjusts control settings based on feedback and performance metrics.
  - **PSO (Particle Swarm Optimization)**: Bio-inspired optimization for energy allocation.
  - **FL (Fuzzy Logic)**: Rule-based energy allocation for simplified control.
- **Dynamic Device Management**: Add, modify, or remove HVAC systems, EV charging spots, and lighting during simulation.
- **Visualization Tools**: Graphical representations of real-time and historical energy allocation results.

### Getting Started

#### Prerequisites

Before running the simulation, you will need to install Python and the necessary libraries. The project is built with Python 3.8+ in mind. Here’s how to set up your environment:

1. **Clone the repository**:

   ```bash
   git clone https://
   cd
   ```

2. **Install dependencies**:
   Run the following command in the project directory:
   ```bash
   pip install -r requirements.txt
   ```

#### Running the Simulation

To start the simulation, execute the interface script:

```bash
python interface.py
```

This will launch the graphical user interface (GUI) to configure and run the simulation.

### How to Use the Interface

1. **Launch the Application**: Start the application by running the main script. The GUI will open with default settings pre-loaded.

2. **Configure Settings**:

   - **Energy Settings**:
     - Input peak and off-peak power settings directly through the interface.
   - **Device Management**:
     - Adjust settings for HVAC systems and charging spots, and lighting devices.
     - Adjust device-specific settings, such as priorities, power constraints, and energy demands.
   - **Simulation Duration**:
     - Adjust the duration of the simulation as needed.

3. **Modify EV Entries**:

   - **Add New EVs**: Use the 'Add Car' button to include new electric vehicles into the simulation.
   - **Delete EVs**: Select an EV entry and use the 'Delete' icon to remove it.
   - **Edit EVs**: Double-click on an existing entry to modify its details.

4. **Choose an Orchestrator**:

   - Select one of the available orchestrators (MPC, AMPC, PSO, FL) or run without an orchestrator for a baseline comparison.

5. **Run Simulation**: Click 'Start Simulation' to begin. The system will process the input data and display the results graphically.

6. **View Results**: Check the plots for detailed visual feedback on how the energy is being utilized over time in response to your configurations.

## Default Example Explanation

### Scenario Overview

The default example in our Energy Management System simulation provides an insightful demonstration of how different orchestrator settings can impact the performance and efficiency of energy distribution among devices.

### None Orchestrator Setting

Under the "None" orchestrator setting, the system does not employ any advanced algorithms to optimize or manage energy distribution. This scenario often leads to suboptimal usage of available resources, as demonstrated in the default setup:

- **Observation**: When using the "None" setting, the third electric vehicle (EV), which requires a high power level (Level 3 charging), experiences significant delays. Despite the availability of power and an unoccupied charging spot, the system fails to start charging this EV immediately.
- **Reason**: The system under the "None" setting lacks the capability to dynamically allocate available power. As a result, the EV must wait until other ongoing processes are completed, or until the system manually reallocates resources to accommodate high-power demands.

![None Orchestrator Example](Without-Using-MPC.png)

### MPC Orchestrator Setting

Switching to the "MPC" orchestrator setting, the system leverages Model Predictive Control algorithms to intelligently manage and distribute power based on predicted demands and operational constraints.

- **Enhanced Management**: With MPC, the system predicts energy needs and dynamically adjusts allocations to optimize utilization across all devices, including EV charging stations and HVAC systems.
- **Specific Outcome**: In the same scenario, MPC allows the third EV to begin charging immediately by orchestrating the exact amount of power required between specific specifications of EVs, even when energy resources are limited.
- **Result**: This leads to more efficient energy usage, reduced wait times, and enhanced overall performance, demonstrating the effectiveness of using advanced control strategies in complex systems.

![MPC Orchestrator Example](Using-MPC.png)

### Conclusion

This example clearly illustrates the advantage of employing an advanced orchestrator like MPC over using no orchestration. Users are encouraged to explore these settings to see firsthand the impact of each on energy management efficiency.

### Contribution

Contributions are welcome! Whether you’re improving orchestrator algorithms or adding new device types, feel free to submit pull requests or raise issues.

### License

This project is licensed under the MIT License.
