import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

    def interactive_trap_potential(self):
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
