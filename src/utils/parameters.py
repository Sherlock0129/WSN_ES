import json


class Parameters:
    def __init__(self, config_file=None):
        """
        Initialize the Parameters class. This will either load configurations from a file
        or set default parameters.

        :param config_file: Optional path to a configuration JSON file.
        """
        self.config_file = config_file
        self.parameters = {}

        if self.config_file:
            self.load_from_file(self.config_file)
        else:
            self.set_default_parameters()

    def set_default_parameters(self):
        """Set default parameters for the network and sensor nodes."""
        self.parameters = {
            "num_nodes": 100,  # Number of nodes in the network
            "low_threshold": 0.1,  # Low energy threshold as percentage of capacity
            "high_threshold": 0.9,  # High energy threshold as percentage of capacity
            "node_initial_energy": 10000,  # Initial energy of each node in Joules
            "max_hops": 5,  # Maximum hops allowed for routing
            "solar_efficiency": 0.2,  # Efficiency of solar energy collection
            "network_area": [100, 100],  # The area (e.g., [width, height]) in which nodes are distributed

            # Transmission and energy parameters
            "E_elec": 50e-9,  # Energy consumption per bit for electronics
            "epsilon_amp": 100e-12,  # Energy consumption per bit per distance^2 for amplification
            "E_char": 1000,  # Energy for charging during WET (Wireless Energy Transfer)
            "G_max": 900,  # Maximum solar irradiance (W/m^2)
            "V": 3.7,  # Battery voltage (V)
            "capacity_mAh": 5200,  # Battery capacity (mAh)
            "frequency": 2.4e9,  # Wireless communication frequency (2.4 GHz)
            "tx_gain": 1.0,  # Transmit antenna gain (linear)
            "rx_gain": 1.0,  # Receive antenna gain (linear)
            "speed_of_light": 3e8,  # Speed of light in m/s
        }

    def load_from_file(self, config_file):
        """Load parameters from a JSON configuration file."""
        try:
            with open(config_file, "r") as f:
                self.parameters = json.load(f)
                print(f"Configuration loaded from {config_file}")
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            self.set_default_parameters()

    def get(self, key, default=None):
        """
        Get the value of a parameter.

        :param key: The parameter key.
        :param default: The default value to return if the key does not exist.
        :return: The value of the parameter.
        """
        return self.parameters.get(key, default)

    def set(self, key, value):
        """
        Set the value of a parameter.

        :param key: The parameter key.
        :param value: The new value for the parameter.
        """
        self.parameters[key] = value

    def save_to_file(self, filename):
        """Save the current parameters to a JSON file."""
        try:
            with open(filename, "w") as f:
                json.dump(self.parameters, f, indent=4)
                print(f"Configuration saved to {filename}")
        except Exception as e:
            print(f"Error saving configuration file: {e}")

    def __str__(self):
        """Return a string representation of the parameters."""
        return json.dumps(self.parameters, indent=4)

