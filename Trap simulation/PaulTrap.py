import numpy as np
import matplotlib.pyplot as plt
import os

class PaulTrap():
    def __init__(self, EC_L, EC_R, DC_L, DC_R, BIAS, position_vector, mass):
        self.mass = mass
        self.EC_L = EC_L
        self.EC_R = EC_R
        self.DC_L = DC_L
        self.DC_R = DC_R
        self.BIAS = BIAS
        self.position_vector = position_vector-9.0 # to center the trap around 0

    def set_voltages(self, V_EC_L, V_DC_L, V_BIAS, V_DC_R, V_EC_R):
        self.V_EC_L = V_EC_L
        self.V_DC_L = V_DC_L
        self.V_BIAS = V_BIAS
        self.V_DC_R = V_DC_R
        self.V_EC_R = V_EC_R

    def get_trap_potential(self):
        return self.V_EC_R*self.EC_R + self.V_EC_L*self.EC_L + self.V_DC_R*self.DC_R + self.V_DC_L*self.DC_L + self.V_BIAS*self.BIAS

    def plot_trap_potential(self):
        total_potential = self.get_trap_potential()
        plt.figure(figsize=(6, 3))
        plt.plot(self.position_vector, total_potential, label='Electric Potential')
        plt.title('Axial axis vs Electric Potential')
        plt.xlabel('Axial axis (mm)')
        plt.ylabel('Electric Potential (V)')
        plt.legend()
        plt.tight_layout()
        plt.show()
