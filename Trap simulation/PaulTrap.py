import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

def create_complementary_arc_length(arc_length):
    diff = np.diff(arc_length)
    complementary_arc_length = np.zeros(len(arc_length))
    diff = diff[::-1]
    for i in range(1, len(arc_length)):
        complementary_arc_length[i] = complementary_arc_length[i - 1] + diff[i - 1]
    return complementary_arc_length

class PaulTrap():
    def __init__(self, mass):
        self.mass = mass
        self.initialize_voltage_responses()
        self.create_position_vector()
        self.V_EC_L = 0
        self.V_DC_L = 0
        self.V_BIAS = 0
        self.V_DC_R = 0
        self.V_EC_R = 0

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
        self.EC_L = self.EC_R[::-1]
        self.arc_length_EC_L = create_complementary_arc_length(self.arc_length_EC_R)

        self.arc_length_DC_R = data_DC[:, 0]
        self.DC_R = data_DC[:, 1]
        self.DC_L = self.DC_R[::-1]
        self.arc_length_DC_L = create_complementary_arc_length(self.arc_length_DC_R)

        self.arc_length_BIAS = data_BIAS[:, 0]
        self.BIAS = data_BIAS[:, 1]

    def create_position_vector(self):
        self.position_vector = np.mean(np.vstack((self.arc_length_BIAS, self.arc_length_DC_L, self.arc_length_EC_L, self.arc_length_DC_R, self.arc_length_EC_R)), axis=0)

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
        plt.figure(figsize=(12, 9))
        plt.plot(self.position_vector, total_potential, label='Electric Potential')
        plt.title('Axial axis vs Electric Potential')
        plt.xlabel('Axial axis (mm)')
        plt.ylabel('Electric Potential (V)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def interactive_trap_potential(self):
        #TODO: fix this function
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
        fig, ax = plt.subplots(figsize=(8, 5))
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
