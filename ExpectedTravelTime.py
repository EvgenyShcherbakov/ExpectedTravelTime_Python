import numpy as np
import time
import matplotlib.pyplot as plt

# Parameters for different simulations
parameters = [
    {
        'cruising_speed': 20,
        'acceleration': 2,
        'deceleration': -4,
        'intervals': [1500, 900, 650, 1200, 800, 700, 1250],
        'lights_cycle': (-59, 70),
        'n_simulations': 10000
    },
    {
        'cruising_speed': 30,
        'acceleration': 5,
        'deceleration': -6,
        'intervals': [850, 900, 1900, 1200, 800, 1050, 1750, 1550],
        'lights_cycle': (-49, 60),
        'n_simulations': 10000
    },
    {
        'cruising_speed': 24,
        'acceleration': 4,
        'deceleration': -6,
        'intervals': [750, 900, 1500, 1200, 800, 1050, 1350, 1450],
        'lights_cycle': (-54, 65),
        'n_simulations': 10000
    }
]

class Route:

    def __init__(self, **kwargs):
        """
        Initialize Route with provided parameters.

        Args:
            cruising_speed (int): The cruising speed of the car.
            acceleration (int): The acceleration rate of the car.
            deceleration (int): The deceleration rate of the car.
            intervals (list): List of interval distances.
            lights_cycle (tuple): Cycle range for traffic lights.
            n_simulations (int): Number of simulations to run.
        """
        self.v_cruise = kwargs['cruising_speed']
        self.a = kwargs['acceleration']
        self.d = kwargs['deceleration']
        self.time_to_stop = Route.time_v_a(self.v_cruise, 0, self.d)
        self.distance_to_stop = Route.distance(self.v_cruise, self.d,
                                               self.time_to_stop)
        self.intervals = kwargs['intervals']
        self.lights_cycle = kwargs['lights_cycle']
        self.n_simulations = kwargs['n_simulations']

    @staticmethod
    def distance(v0, a, t):
        """
        Calculate distance based on initial velocity, acceleration, and time.

        Args:
            v0 (float): Initial velocity.
            a (float): Acceleration.
            t (float): Time.

        Returns:
            float: Calculated distance.
        """
        return v0 * t + 0.5 * a * t * t

    @staticmethod
    def time_d_v_a(st, a, v0):
        """
        Calculate time based on distance, acceleration, and initial velocity.

        Args:
            st (float): Distance.
            a (float): Acceleration.
            v0 (float): Initial velocity.

        Returns:
            float: Calculated time.
        """
        return (pow(v0 * v0 + 2 * a * st, 0.5) - v0) / a

    @staticmethod
    def distance_const_v(v0, t):
        """
        Calculate distance traveled at constant velocity.

        Args:
            v0 (float): Velocity.
            t (float): Time.

        Returns:
            float: Calculated distance.
        """
        return v0 * t

    @staticmethod
    def time_d_const_v(st, v0):
        """
        Calculate time based on distance and constant velocity.

        Args:
            st (float): Distance.
            v0 (float): Velocity.

        Returns:
            float: Calculated time.
        """
        return st / v0

    @staticmethod
    def speed(v0, a, t):
        """
        Calculate speed based on initial velocity, acceleration, and time.

        Args:
            v0 (float): Initial velocity.
            a (float): Acceleration.
            t (float): Time.

        Returns:
            float: Calculated speed.
        """
        return v0 + a * t

    @staticmethod
    def time_v_a(v0, vt, a):
        """
        Calculate time based on initial and final velocities and acceleration.

        Args:
            v0 (float): Initial velocity.
            vt (float): Final velocity.
            a (float): Acceleration.

        Returns:
            float: Calculated time.
        """
        return (vt - v0) / a

    def interval(self, v0, length, time_to_green):
        """
        Computes interval travel time and exit speed.

        Args:
            v0 (float): Initial velocity.
            length (float): Length of the interval.
            time_to_green (int): Time until the traffic light turns green.

        Returns:
            tuple: (total interval travel time, exit speed)
        """
        time_to_accelerate = 0
        distance_to_accelerate = 0

        # Accelerate to cruising speed if needed
        if v0 < self.v_cruise:
            time_to_accelerate = Route.time_v_a(v0, self.v_cruise, self.a)
            distance_to_accelerate = Route.distance(v0, self.a,
                                                    time_to_accelerate)

        # Calculate cruising distance and time
        distance_to_cruise = (
                    length - distance_to_accelerate - self.distance_to_stop)
        time_to_cruise = Route.time_d_const_v(distance_to_cruise, self.v_cruise)

        # If the light is green at the decision point
        if time_to_green <= 0:
            time_to_finish = Route.time_d_const_v(self.distance_to_stop,
                                                  self.v_cruise)
            return (
            time_to_accelerate + time_to_cruise + time_to_finish, self.v_cruise)

        # If the light will stay red longer than the time needed to stop
        if time_to_green >= self.time_to_stop:
            return (time_to_accelerate + time_to_cruise + time_to_green, 0)

        # Deceleration phase
        decelerate_time = time_to_green
        distance_to_decelerate = Route.distance(self.v_cruise, self.d,
                                                decelerate_time)
        speed_at_switch = Route.speed(self.v_cruise, self.d, decelerate_time)

        # Second acceleration phase
        distance_to_accelerate = self.distance_to_stop - distance_to_decelerate
        time_to_accelerate2 = Route.time_d_v_a(distance_to_accelerate, self.a,
                                               speed_at_switch)
        speed_at_finish = Route.speed(speed_at_switch, self.a,
                                      time_to_accelerate2)

        return (
        time_to_accelerate + time_to_cruise + decelerate_time + time_to_accelerate2,
        speed_at_finish)

    def path(self, lengths, lst_time_to_green):
        """
        Computes total travel time through the path.

        Args:
            lengths (list): List of interval lengths.
            lst_time_to_green (list): List of times until the traffic lights turn green.

        Returns:
            float: Total travel time through the path.
        """
        # Append the predetermined time to stop for the last interval
        lst_time_to_green.append(self.time_to_stop)
        total_time = 0
        enter_speed = 0

        # Calculate travel time for each interval
        for idx, interval_length in enumerate(lengths):
            interval_time, enter_speed = self.interval(
                enter_speed,
                interval_length,
                lst_time_to_green[idx]
            )
            total_time += interval_time

        return total_time

    def simulate(self):
        """
        Simulates the travel times and computes statistics.

        Returns:
            tuple: (average time, 5th percentile time, 95th percentile time)
        """
        times = np.zeros(self.n_simulations)

        # Generate random times until traffic lights turn green for all simulations
        times_to_green = np.random.randint(
            self.lights_cycle[0], self.lights_cycle[1],
            self.n_simulations * (len(self.intervals) - 1)
        ).reshape(self.n_simulations, len(self.intervals) - 1)

        # Simulate travel times for each simulation
        for i in range(self.n_simulations):
            switch_to_green = list(times_to_green[i])
            switch_to_green.append(round(self.time_to_stop))
            times[i] = self.path(self.intervals, switch_to_green)

        # Calculate 5th and 95th percentile times
        low_time_s, high_time_s = np.percentile(times, [5, 95])

        return np.average(times), low_time_s, high_time_s

    def avg_simulation(self, n):
        """
        Runs multiple simulations and averages the results.

        Args:
            n (int): Number of simulations to run.

        Returns:
            tuple: (average travel time, 5th percentile average, 95th percentile average)
        """
        avg_sum, low_sum, high_sum = 0, 0, 0

        # Run multiple simulations and sum the results
        for _ in range(n):
            avg_s, low_s, high_s = self.simulate()
            avg_sum += avg_s
            low_sum += low_s
            high_sum += high_s

        # Calculate the average of each metric
        return (
        round(avg_sum / n, 4), round(low_sum / n, 4), round(high_sum / n, 4))

    def simulate_sample(self):
        """
        Simulates the travel times and returns the results.

        Returns:
            numpy.ndarray: Array of travel times for each simulation.
        """
        times = np.zeros(self.n_simulations)

        # Generate random times until traffic lights turn green for all simulations
        times_to_green = np.random.randint(
            self.lights_cycle[0], self.lights_cycle[1],
            self.n_simulations * (len(self.intervals) - 1)
        ).reshape(self.n_simulations, len(self.intervals) - 1)

        # Simulate travel times for each simulation
        for i in range(self.n_simulations):
            switch_to_green = list(times_to_green[i])
            switch_to_green.append(round(self.time_to_stop))
            times[i] = self.path(self.intervals, switch_to_green)

        return times


# Create Route instances for each parameter set
rt = [Route(**parameters[i]) for i in range(len(parameters))]

# Simulate sample travel times and gather data
data = [obj.simulate_sample() for obj in rt]

# Draw histograms of the simulated travel times
n_bins = 30
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, tight_layout=True)
for idx, sim_data in enumerate(data):
    axs[idx].hist(sim_data, bins=n_bins)

plt.show()
