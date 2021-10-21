# overhead

import logging
import math
import random
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# environment parameters

FRAME_TIME = 0.1  # time interval
GRAVITY_ACCEL = 0.16  # gravity constant
BOOST_ACCEL = 0.20  # thrust constant

# # the following parameters are not being used in the sample code
PLATFORM_WIDTH = 0.25  # landing platform width
PLATFORM_HEIGHT = 0.06  # landing platform height
ROTATION_ACCEL = 20  # rotation constant

"""Constraint 1: Trying to include drag in y direction (upward) but it's going to e less then the thrust"""
"""Constraint 2: Including the crosswind as a randomness variable"""
"""Constraint 3: This constraint is sort of side counter thrust to crosswind which acts
with 90% the power of crosswind because if its same as cross wind it basically cancels each other"""


class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):
        """
        action: there are three of them
        action[0]: take off or landing thrust in y direction range (0, 1)
        action[1]: cross wind velocity in x direction range (-1, 1)
                    -1 - cross wind to the left
                    0 - no cross wind
                    1 - cross wind to the right
        action[2]: counter side thrust to cross wind but at 90% power range (-1, 1)
                    -1 - side thrust to the right
                    0 - no side thrust
                    1 - side thrust to left
        state[0] = x
        state[1] = v_x
        state[2] = -0.9*v_x
        state[3] = -v_x
        state[4] = 0.9*v_x
        state[5] = y
        state[6] = v_y
        """
        # Apply gravity
        delta_state_gravity = -t.tensor([0., 0., 0., 0., 0., 0., GRAVITY_ACCEL * FRAME_TIME])
        # Thrust
        # Note: Same reason as above. Need a 2-by-1 tensor.
        vertical_thrust_y = action[0]
        crosswind = action[1]
        side_counter_thrust = action[2]

        delta_state_vertical = BOOST_ACCEL * FRAME_TIME * t.tensor(
            [0., 0., 0., 0., 0., 0., 1.]) * vertical_thrust_y  # 1
        delta_state_crosswind_r = BOOST_ACCEL * FRAME_TIME * t.tensor(
            [0., 1., 0., 0., 0., 0., 0.]) * action * crosswind  # 2
        delta_state_crosswind_l = BOOST_ACCEL * FRAME_TIME * t.tensor(
            [0., 0., 0., -1., 0., 0., 0.]) * action * crosswind  # 3
        delta_state_side_thrust_l = BOOST_ACCEL * FRAME_TIME * t.tensor(
            [0., 0., -1., 0., 0., 0., 0.]) * action * side_counter_thrust  # 4
        delta_state_side_thrust_r = BOOST_ACCEL * FRAME_TIME * t.tensor(
            [0., 0., 0., 0., 1., 0., 0.]) * action * side_counter_thrust  # 5
        # Update velocity
        state = state + delta_state_vertical + delta_state_crosswind_r + delta_state_crosswind_l + delta_state_side_thrust_r + delta_state_side_thrust_l + delta_state_gravity  # drag part goes in here
        # Update state
        # Note: Same as above. Use operators on matrices/tensors as much as possible.
        # Do not use element-wise operators as they are considered inplace.
        step_mat = t.tensor([[0., 0., 0., 0., 0., 1., FRAME_TIME],  # 1
                             [0., 0., 0., 0., 0., 0., 1.],  # 1
                             [1., FRAME_TIME, 0., 0., 0., 0., 0.],  # 2
                             [0., 1., 0., 0., 0., 0., 0.],  # 2
                             [1., 0., 0., FRAME_TIME, 0., 0., 0.],  # 3
                             [0., 0., 0., 1., 0., 0., 0., 0.],  # 3
                             [1., 0., FRAME_TIME, 0., 0., 0., 0.],  # 4
                             [0., 0., 1., 0., 0., 0., 0.],  # 4
                             [1., 0., 0., 0., FRAME_TIME, 0., 0.],  # 5
                             [0., 0., 0., 0., 1., 0., 0.]])  # 5
        state = t.matmul(step_mat, state)

        return state

        # print(type(state))


# a deterministic controller
# Note:
# 0. You only need to change the network architecture in "__init__"
# 1. nn.Sigmoid outputs values from 0 to 1, nn.Tanh from -1 to 1
# 2. You have all the freedom to make the network wider (by increasing "dim_hidden") or
# deeper (by adding more lines to nn.Sequential)
# 3. Always start with something simple

class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to you
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            # You can add more layers here
            nn.Sigmoid()
        )

    def forward(self, state):
        action = self.network(state)
        return action


# the simulator that rolls out x(1), x(2), ..., x(T)
# Note:
# 0. Need to change "initialize_state" to optimize the controller over a distribution of initial states
# 1. self.action_trajectory and self.state_trajectory stores the action and state trajectories along time

class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():
        #  state 0   1   2   3   4   5   6
        state = [1., 0., 2., 5., 7., 5., 8.]  # need batch of initial states
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        return state[0] ** 2 + state[1] ** 2 + state[2] ** 2 + state[3] ** 2 + state[4] ** 2 + state[5] ** 2 + state[6] ** 2


# set up the optimizer
# Note:
# 0. LBFGS is a good choice if you don't have a large batch size (i.e., a lot of initial
#    states to consider simultaneously)
# 1. You can also try SGD and other momentum-based methods implemented in PyTorch
# 2. You will need to customize "visualize"
# 3. loss.backward is where the gradient is calculated (d_loss/d_variables)
# 4. self.optimizer.step(closure) is where gradient descent is done

class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))
            self.visualize()

    def visualize(self):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x, y)
        # plt.show()
        # if o == 40:
        #     data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        #     x = data[:, 0]
        #     y = data[:, 1]
        #     plt.plot(x, y)
        #     plt.show()


# Now it's time to run the code!

T = 100  # number of time steps
dim_input = 7  # state space dimensions
dim_hidden = 20  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
o = Optimize(s)  # define optimizer
o.train(80)  # solve the optimization problem
print(s)
print(type(s))
print(d)

print(c)
