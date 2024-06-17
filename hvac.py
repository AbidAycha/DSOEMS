"""
    Represents an HVAC device model that extends the generic Device class.
    This model is used for running thermal simulations to determine how changes in energy allocation
    affect room temperature under various operational conditions.
"""

import math
from device import Device


class Variables:
    """
        Holds all the relevant physical and environmental variables used in HVAC simulations.
        This includes properties of air, thermal coefficients, and state conditions necessary 
        for energy calculations in different heating or cooling scenarios.
    """
    def __init__(self, t1,t2):
        self.side = 0.9
        self.airDensity = 1.225
        self.cAir = 1.01  # J/gram-celcius
        self.diameter = 12.0 / 100.0
        self.transCoeff = [0.1 / .02, 4]
        self.TRoom = t1
        self.TIn = t2
        self.inletSpeed = 10
        self.stefanBoltzmann = 5.670373 * (10 ** (-8))
        self.emissivity = 0.4
        self.tungstenArea = 3 * (10 ** (-5))
        self.tungstenTemp = 3000
        self.sample = 1
        self.percentOpen = 100
        self.wallRoomCoeff = 10
        self.measuredTemp = self.TRoom
        self.COP_heating = 3.0  # Coefficient of Performance for heating
        self.EER_cooling = 10.0  # Energy Efficiency Ratio for cooling
        self.mode = 'heating'  # Operation mode: 'heating' or 'cooling'
        self.R_value = 5
        self.recalculate()

    def recalculate(self):
        self.boxAirMass = self.airDensity * (self.side ** 3)
        self.area = self.side ** 2
        self.specMassAir = self.boxAirMass * self.cAir * 1000
        self.massPerSecond = self.sample * (math.pi * self.diameter * self.diameter / 4) * self.inletSpeed * self.airDensity * (self.percentOpen / 100)
        self.plywoodCMass = (self.area) * (2 / 100.0) * 545 * 1215 * 5.2 * 0.2
        

class Model(Device):
    def __init__(self, duration, t1,t2,energy_min=1, energy_max=4, priority=1):
        super().__init__("HVAC", priority, energy_min, energy_max)
        self.var = Variables(t1,t2)
        self.allTemps = []
        self.i = 0
        self.setPoint = 0
        self.wallTemp = self.var.TRoom
        self.energy_min = energy_min  
        self.energy_max = energy_max 
        self.power_consumption = [0] * (duration*4+10)
        self.roomtemp=[]
        self.current_energy_usage=0
    def update_state(self, allocated_energy):
        self.current_energy_usage=allocated_energy
    
    def getConduction(self):
        conduction = 0
        for i in range(len(self.var.transCoeff)):
            conduction += (self.var.transCoeff[i] * self.var.area)
        conduction = conduction * (self.wallTemp - self.var.TIn)
        return conduction

    def getNewMixedHeat(self):
        massLeft = self.var.boxAirMass - self.var.massPerSecond
        t1 = (massLeft * self.var.TRoom)
        t2 = (self.var.massPerSecond * self.var.TIn)
        t = (t1 + t2) * (self.var.cAir * 1000)
        return t

    def wallTempFunc(self):
        self.wallTemp -= ( self.getConduction()) / self.var.plywoodCMass
    
    def runModel(self, needed,time_interval,power):
        for _ in range(3): # each 5 minutes
            self.roomtemp.append(self.var.TRoom)

            if self.check_power_availability(needed,power):
                if self.power_consumption[time_interval] == 0:
                    self.power_consumption[time_interval] = needed
                newTRoom = self.getNewMixedHeat() / (self.var.boxAirMass * self.var.cAir * 1000)
                self.var.TRoom = newTRoom
                self.wallTempFunc()
            else:
                rate_of_change = (self.wallTemp - self.var.TRoom) / self.var.R_value
                self.var.TRoom += rate_of_change * 0.02
    
    def calculate_energy_required(self,t1,t2):
        delta_T = t1 - t2
        if delta_T > 0:
            self.var.mode = 'heating'
        else:
            self.var.mode = 'cooling'
            delta_T = -delta_T  # For cooling, we need a positive temperature difference

        heat_required = self.var.specMassAir * delta_T  # J

        if self.var.mode == 'heating':
            energy_required = heat_required / (self.var.COP_heating * 3600) + 0.8  # Convert J to kWh
        else:
            # For cooling, assuming EER is given as BTU/(Wh), and 1 BTU = 1055 J
            energy_required = heat_required / (self.var.EER_cooling * 1055) + 0.8 # kWh
        return max(self.energy_min, min(energy_required, self.energy_max))
    
    def check_power_availability(self, needed,power):
        return needed <= power


