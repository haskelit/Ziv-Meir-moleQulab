import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from scipy.interpolate import interp1d
from scipy.constants import k


def find_closest_index(array, value):
    differences = np.abs(array - value)
    index = np.argmin(differences)
    return index

class PaulTrap():
    def __init__(self, mass, RF_freq, charge=1):
        self.mass = mass * 1.66e-27 # atomic mass unit in kg
        self.charge = charge * 1.602e-19  # Convert charge to Coulombs
        self.RF_freq = RF_freq
        self.initialize_voltage_responses()
        self.create_position_vector()
        self.interpolate_potentials()
        self.mirror_potentials()
        self.shift_position_vector()
        self.add_effective_AC_potential()
        self.V_EC_L = 0
        self.V_DC_L = 0
        self.V_BIAS = 0
        self.V_DC_R = 0
        self.V_EC_R = 0
        self.AC_Voltage = 0

    def initialize_voltage_responses(self):
        script_dir = os.getcwd()
        file_EC = os.path.join(script_dir, 'electrodes responses\Shuttling Endcap Smoothed.txt')
        file_DC = os.path.join(script_dir, 'electrodes responses\Shuttling DC Smoothed.txt')
        file_BIAS = os.path.join(script_dir, 'electrodes responses\Shuttling BIAS Smoothed.txt')

        data_EC = np.loadtxt(file_EC, skiprows=8)
        data_DC = np.loadtxt(file_DC, skiprows=8)
        data_BIAS = np.loadtxt(file_BIAS, skiprows=8)

        self.arc_length_EC_R = data_EC[:, 0]
        self.EC_R = data_EC[:, 1]

        self.arc_length_DC_R = data_DC[:, 0]
        self.DC_R = data_DC[:, 1]

        self.arc_length_BIAS = data_BIAS[:, 0]
        self.BIAS = data_BIAS[:, 1]

    def create_position_vector(self):
        self.position_vector = np.linspace(self.arc_length_DC_R[0], self.arc_length_EC_R[-1], len(self.arc_length_EC_R))

    def shift_position_vector(self):
        self.position_vector -= self.position_vector[-1] / 2

    def interpolate_potentials(self):
        self.EC_R = interp1d(self.arc_length_EC_R, self.EC_R, kind='quadratic', fill_value="extrapolate")(self.position_vector)
        self.DC_R = interp1d(self.arc_length_DC_R, self.DC_R, kind='quadratic', fill_value="extrapolate")(self.position_vector)
        self.BIAS = interp1d(self.arc_length_BIAS, self.BIAS, kind='quadratic', fill_value="extrapolate")(self.position_vector)

    def add_effective_AC_potential(self):
        self.effective_AC_potential = (self.charge / (4 * self.mass * self.RF_freq**2)) * (np.gradient(self.DC_R + self.DC_L + self.BIAS, self.position_vector/1000)**2)

    def mirror_potentials(self):
        self.EC_L = self.EC_R[::-1]
        self.DC_L = self.DC_R[::-1]

    def set_DC_voltages(self, V_EC_L, V_DC_L, V_BIAS, V_DC_R, V_EC_R):
        self.V_EC_L = V_EC_L
        self.V_DC_L = V_DC_L
        self.V_BIAS = V_BIAS
        self.V_DC_R = V_DC_R
        self.V_EC_R = V_EC_R

    def set_AC_voltage(self, AC_voltage):
        self.AC_Voltage = AC_voltage

    def get_trap_potential(self):
        return (self.V_EC_R * self.EC_R +
                self.V_EC_L * self.EC_L +
                self.V_DC_R * self.DC_R +
                self.V_DC_L * self.DC_L +
                self.V_BIAS * self.BIAS +
                (self.AC_Voltage**2) * self.effective_AC_potential
                )

    def plot_trap_potential(self):
        total_potential = self.get_trap_potential()
        plt.figure(figsize=(7, 5))
        plt.plot(self.position_vector, total_potential, label='Electric Potential')
        plt.title('Axial axis vs Electric Potential')
        plt.xlabel('Axial axis (mm)')
        plt.ylabel('Electric Potential (V)')
        plt.tight_layout()
        plt.show()

    def plot_electrode_potentials(self):
        plt.figure(figsize=(7, 5))
        plt.plot(self.position_vector, self.EC_R, label='Endcap Right')
        plt.plot(self.position_vector, self.DC_R, label='DC Right')
        plt.plot(self.position_vector, self.BIAS, label='BIAS')
        plt.plot(self.position_vector, self.EC_L, label='Endcap Left')
        plt.plot(self.position_vector, self.DC_L, label='DC Left')
        plt.plot(self.position_vector, self.effective_AC_potential, label='Effective AC potential')

        plt.title('Electrode Potentials')
        plt.xlabel('Axial axis (mm)')
        plt.ylabel('Electric Potential (V)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_interactive_trap_potential(self):
        def update_potential(index, value):
            # Update the corresponding voltage
            voltages[index] = float(value)
            self.set_voltages(*voltages)
            # Update the plot
            total_potential = self.get_trap_potential()
            ax.clear()
            ax.plot(self.position_vector, total_potential, label='Electric Potential')
            ax.set_title('Axial axis vs Electric Potential')
            ax.set_xlabel('Axial axis (mm)')
            ax.set_ylabel('Electric Potential (V)')
            ax.legend()
            canvas.draw()

        # Create the main window
        root = tk.Tk()
        root.title("Interactive Trap Potential")

        # Create a frame for the sliders
        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Define initial voltages
        voltages = [self.V_EC_L, self.V_DC_L, self.V_BIAS, self.V_DC_R, self.V_EC_R]

        # Create sliders for each potential
        labels = ["V_EC_L", "V_DC_L", "V_BIAS", "V_DC_R", "V_EC_R"]
        for i, label_text in enumerate(labels):
            label = ttk.Label(frame, text=label_text)
            label.grid(row=i, column=0, sticky=tk.W)

            slider = ttk.Scale(frame, from_=-10, to=50, orient=tk.HORIZONTAL,
                               command=lambda value, idx=i: update_potential(idx, value))
            slider.set(voltages[i])
            slider.grid(row=i, column=1, sticky=(tk.W, tk.E))

            value_label = ttk.Label(frame, text=str(voltages[i]))
            value_label.grid(row=i, column=2, sticky=tk.W)

            slider.config(command=lambda value, idx=i, l=value_label: [update_potential(idx, value),
                                                                       l.config(text=f"{float(value):.1f}")])

        # Configure column weights
        frame.columnconfigure(1, weight=1)

        # Create a matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(4, 3))
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Initial plot
        total_potential = self.get_trap_potential()
        ax.plot(self.position_vector, total_potential, label='Electric Potential')
        ax.set_title('Axial axis vs Electric Potential')
        ax.set_xlabel('Axial axis (mm)')
        ax.set_ylabel('Electric Potential (V)')
        ax.legend()
        canvas.draw()

        # Start the main event loop
        root.mainloop()

    def fit_parabola(self, center_position, width): # all in mm
        center_index = find_closest_index(self.position_vector, center_position)
        gap_between_indices = (self.position_vector[1] - self.position_vector[0])
        #print(gap_between_indices)
        start_index = center_index - int(width / (2 * gap_between_indices))
        end_index = center_index + int(width / (2 * gap_between_indices))
        #print(start_index, end_index)
        coefficients = np.polyfit(self.position_vector[start_index:end_index], self.get_trap_potential()[start_index:end_index], 2)
        self.plot_with_fit(coefficients, start_index, end_index)
        return coefficients

    def plot_with_fit(self, coefficients, start_index, end_index):
        plt.figure(figsize=(7, 5))
        plt.scatter(self.position_vector[start_index:end_index], self.get_trap_potential()[start_index:end_index], label='Electric Potential', color='red', s=4)
        x_fit = self.position_vector[start_index:end_index]
        y_fit = np.polyval(coefficients, x_fit)
        plt.plot(x_fit, y_fit, label='Parabola Fit')
        plt.title('Axial axis vs Electric Potential with Parabola Fit')
        plt.xlabel('Axial axis (mm)')
        plt.ylabel('Electric Potential (V)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_trap_frequency(self, center_position, width):
        coefficients = self.fit_parabola(center_position, width)
        alpha = coefficients[0] * 1e6
        omega_z = np.sqrt(2 * alpha * self.charge / self.mass)
        return omega_z

    def get_RF_barrier(self):
        RF_barrier = np.max(self.AC_Voltage**2 * self.effective_AC_potential)
        return RF_barrier, RF_barrier*self.charge/k



