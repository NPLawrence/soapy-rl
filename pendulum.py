import torch
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable, Objective
from neuromancer.loss import PenaltyLoss, AggregateLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL, pltPhase, plot_trajectories
import neuromancer.psl as psl
from neuromancer.dynamics import ode, integrators

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


from neuromancer.psl.signals import step, sines, periodic, noise, walk
from neuromancer.psl.base import ODE_NonAutonomous as ODE
from neuromancer.psl.base import cast_backend
from neuromancer.dynamics.ode import ODESystem
import inspect, sys

class Acrobot(ODESystem):
    """https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/acrobot.py"""
    def __init__(self, insize=5, outsize=4):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.m1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.m2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.l1 = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.l2 = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.g = torch.nn.Parameter(torch.tensor([9.81]), requires_grad=True)

        self.lc1 = self.l1/2.0
        self.lc2 = self.l2/2.0

        self.moi1 = (self.m1 * self.l1**2) / 3 # gym uses 1.0 with is incorrect...
        self.moi2 = (self.m2 * self.l2**2) / 3 # 1.0

    def single_sample_ode(self, x, u):

        theta1 = x[0]
        theta2 = x[1]
        dtheta1 = x[2]
        dtheta2 = x[3]

        d1 = self.m1 * self.lc1 **2 + self.m2 * (self.l1**2 + self.lc2 **2 + 2 * self.l1 * self.lc2  * torch.cos(theta2)) + self.moi1 + self.moi2
        d2 = self.m2 * (self.lc2**2 + self.l1 * self.lc2 * torch.cos(theta2)) + self.moi2
        phi2 = self.m2 * self.lc2 * self.g * torch.cos(theta1 + theta2 - torch.pi / 2.0)

        phi1 = (
            -self.m2 * self.l1 * self.lc2 * dtheta2**2 * torch.sin(theta2)
            - 2 * self.m2 * self.l1 * self.lc2 * dtheta2 * dtheta1 * torch.sin(theta2)
            + (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * torch.cos(theta1 - torch.pi / 2)
            + phi2
        )

        ddtheta2 = (
            u + d2 / d1 * phi1 - self.m2 * self.l1 * self.lc2 * dtheta1**2 * torch.sin(theta2) - phi2
        ) / (self.m2 * self.lc2**2 + self.moi2 - d2**2 / d1)

        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        dx = torch.hstack((dtheta1, dtheta2, ddtheta1, ddtheta2))

    
        return dx

    
    def ode_equations(self, x, u):
        
        return torch.vmap(self.single_sample_ode, randomness='different')(x,u)


class DoubleInvertedPendulum(ODESystem):
    """https://www3.math.tu-berlin.de/Vorlesungen/SS12/Kontrolltheorie/matlab/inverted_pendulum.pdf"""
    def __init__(self, insize=7, outsize=6):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.m0 = torch.nn.Parameter(torch.tensor([0.6]), requires_grad=True)
        self.m1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.m2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.l1 = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.l2 = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.g = torch.nn.Parameter(torch.tensor([9.80665]), requires_grad=True)

    def single_sample_ode(self, x, u):

        q, theta1, theta2 = x[0], x[1], x[2]
        qdot, theta1dot, theta2dot = x[3], x[4], x[5]

        M = torch.vstack((torch.hstack((self.m0 + self.m1 + self.m2, self.l1*(self.m1 + self.m2)*torch.cos(theta1), self.m2*self.l2*torch.cos(theta2))),
                          torch.hstack((self.l1*(self.m1 + self.m2)*torch.cos(theta1), self.l1**2 * (self.m1 + self.m2), self.l1*self.l2*self.m2*torch.cos(theta1-theta2))),
                          torch.hstack((self.l2*self.m2*torch.cos(theta2), self.l1*self.l2*self.m2*torch.cos(theta1-theta2), self.l2**2 * self.m2))))
        
        f = torch.hstack((self.l1*(self.m1 + self.m2)*theta1dot**2 * torch.sin(theta1) + self.m2*self.l2*theta2dot**2 * torch.sin(theta2),
                          -self.l1*self.l2*self.m2*theta2dot**2 * torch.sin(theta1 - theta2) + self.g*(self.m1 + self.m2)*self.l1*torch.sin(theta1),
                          self.l1*self.l2*self.m2*theta1dot**2 * torch.sin(theta1-theta2) + self.g*self.l2*self.m2*torch.sin(theta2))) 
        input = torch.hstack((u, torch.zeros_like(u), torch.zeros_like(u)))
        disturbance = 0.0*torch.randn(3)

        sol = torch.linalg.solve(M, f + input + disturbance, left=True)

        dx = torch.hstack((qdot, theta1dot, theta2dot, sol))
    
        return dx

    
    def ode_equations(self, x, u):
        
        return torch.vmap(self.single_sample_ode, randomness='different')(x,u)


class InvertedPendulum(ODESystem):
    """
    https://github.com/pnnl/neuromancer/blob/master/src/neuromancer/psl/nonautonomous.py
    Inverted Pendulum dynamics
    * states: :math:`x = [\theta \dot{\theta}]`;
        * :math:`\theta` is angle from upright equilibrium
    * input: u = input torque

    """
    def __init__(self, insize=3, outsize=2):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.b = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.m = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True) #0.15
        self.L = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True) #0.5
        self.g = torch.nn.Parameter(torch.tensor([9.81]), requires_grad=True) #friction

    def single_sample_ode(self, x, u):
        theta, theta_dot = x[0], x[1]
        y = torch.hstack((theta_dot,(self.m * self.g * self.L * torch.sin(theta) - self.b * theta_dot) / (self.m * self.L ** 2)))
        y[1] = y[1] + (u / (self.m * self.L ** 2))
        return y

    def ode_equations(self, x, u):
        
        return torch.vmap(self.single_sample_ode, randomness='different')(x,u)


class DoubleIntegrator(ODESystem):
    """
    Double integrator dynamics
    * states: :math:`x = [p, v]`
        * :math:`p` is position
        * :math:`v` is velocity
    * input: :math:`u` is force/acceleration command
    """
    def __init__(self, insize=3, outsize=2):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.m = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)

    def single_sample_ode(self, x, u):
        position, velocity = x[0], x[1]
        acceleration = (u - self.b * velocity) / self.m
        return torch.hstack((velocity, acceleration))

    def ode_equations(self, x, u):

        return torch.vmap(self.single_sample_ode, randomness='different')(x, u)
