A self-driven electric car needs to make a delivery from point A to point B. The path consists of intervals with a traffic light at the end of each interval. Before the journey, the car calculates the expected, lucky, unlucky time travel estimates and uploads this information to the server.

Travel through an interval can be modeled as having these parts:

Initial acceleration. The speed at t = 0 can be zero or any value less than cruising speed coming from the previous interval. The car accelerates to the cruising speed. If it is already at cruising speed, then it does not need to accelerate.

Cruising distance. The car travels at constant speed until the decision point. Stoppage distance is the length required for the car to drop its speed from cruising speed to zero. Decision point is at the end of interval minus the stoppage distance.

Deceleration. If the light is yellow or red, the car starts dropping the speed.

Second acceleration. If the light suddenly turns green before the car is completely stopped, it accelerates and exits the interval with some speed.

If the light is green at the decision point, the car does not decelerate. It procced and exits the interval with cruising speed. If the light is yellow or red, the car stops at the end of the interval and continues to wait until the lights turns green. It enters the next interval with speed equal zero.

The traffic lights are modeled as a random number from the inclusive interval e.g. (-59, 70). If the number is <= 0, the light is green. A positive number represents a number of seconds when the lights turn green after the car passes the decision point. For example if time_to_green == 2, the car will decelerate for 2 seconds, then switch to accelerating; if time_to_green == 50, the car will decelerate, stop and wait until 50 seconds pass since the decision point, then enters next interval with zero speed.

Travel through the path is summation of travels through connected intervals. It can enter the next interval with non-zero speed if got lucky with the light. At the last interval time_to_green == stoppage_time, not random! In this way it’s guaranteed that the car arrived at the end of the path and stopped.

Write the implementation of the class Route. It receives a dictionary with parameters of the car, intervals, lights cycle, and number of simulations. Description of three required methods is in the Code window. Per each simulation randomly generate n-1 time_to_green numbers to model the traffic lights, last interval has defined time_to_green.

After having a sample of travel times compute: average, 5-percentile (lucky), 95-percentile (unlucky) of the sample.
