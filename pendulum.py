import numpy as np
from scipy.integrate import odeint 

class DoublePendulum():
    """_summary_: class for the double pendulum
    """
    def __init__(self, m1 = 2, m2 = 1, l1 = 1.4, l2 = 1, g = 9.8, number_of_points = 1500):
        """_summary_: initialize the double pendulum class with physical parameters

        Args:
            m1 (float, optional): first mass. Defaults to 2.
            m2 (float, optional): second mass. Defaults to 1.
            l1 (float, optional): length of first bar. Defaults to 1.5.
            l2 (int, optional): length of second bar. Defaults to 1.
            g (float, optional): gravity. Defaults to 9.8.
        """
        
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.number_of_points = number_of_points
        
        print('The double pendulum is initialized with the following physical parameters:')
        self.get_physical_parameters()
        
    def update_physical_parameters(self, m1 = None, m2 = None, l1 = None, l2 = None, g = None, number_of_points = None):
        """_summary_: update the physical parameters of the double pendulum

        Args:
            m1 (float, optional): first mass. Defaults to None.
            m2 (float, optional): second mass. Defaults to None.
            l1 (float, optional): length of first bar. Defaults to None.
            l2 (int, optional): length of second bar. Defaults to None.
            g (float, optional): gravity. Defaults to None.
        """
        if m1 is not None:
            self.m1 = m1
        if m2 is not None:
            self.m2 = m2
        if l1 is not None:
            self.l1 = l1
        if l2 is not None:
            self.l2 = l2
        if g is not None:
            self.g = g
        if number_of_points is not None:
            self.number_of_points = number_of_points
        
    def get_physical_parameters(self):
        print(
            "m1 = ", self.m1, "kg",
            "m2 = ", self.m2, "kg",
            "l1 = ", self.l1, "m",
            "l2 = ", self.l2, "m",
            "g = ", self.g, "m/s^2",
            "number of points = ", self.number_of_points,
        )
        
        return self.m1, self.m2, self.l1, self.l2, self.g, self.number_of_points
        
    def initial_condition(self, 
                          theta1 = -10, 
                          theta2 = 10, 
                          theta1_dot = 0, 
                          theta2_dot = 0):
        
        """_summary_: initialize the initial conditions of the double pendulum (2 angles and 2 angular velocities)

        Args:
            theta1 (float, optional): _description_. Defaults to 0.
            theta2 (float, optional): _description_. Defaults to 0.
            theta1_dot (float, optional): _description_. Defaults to 0.
            theta2_dot (float, optional): _description_. Defaults to 0.
        """
        self.theta1 = np.pi*theta1/180
        self.theta2 = np.pi*theta2/180
        self.theta1_dot = theta1_dot
        self.theta2_dot = theta2_dot
        
        self.u0 = [theta1, theta1_dot, theta2, theta2_dot]


    def double_pendulum_step(self,u,t,m1,m2,L1,L2,g):
        """_summary_: ODE function for the double pendulum

        Args:
            u (_type_): current state
            t (_type_): current time

        Returns:
            np.array: state at the next time step t + dt 
        """
        # du = derivatives
        # u = variables
        # p = parameters
        # t = time variable
        
        du = np.zeros(4)
        c = np.cos(u[0]-u[2])  # intermediate variables
        s = np.sin(u[0]-u[2])  # intermediate variables

        du[0] = u[1]   # d(theta 1)
        du[1] = ( m2*g*np.sin(u[2])*c - m2*s*(L1*c*u[1]**2 + L2*u[3]**2) - (m1+m2)*g*np.sin(u[0]) ) /( L1 *(m1+m2*s**2) )
        du[2] = u[3]   # d(theta 2)   
        du[3] = ((m1+m2)*(L1*u[1]**2*s - g*np.sin(u[2]) + g*np.sin(u[0])*c) + m2*L2*u[3]**2*s*c) / (L2 * (m1 + m2*s**2))
        
        return du
    
    def compute_trajectory(self, 
                           simulation_time = 25.0, 
                           number_of_points = 1500, 
                           theta1 = 0, 
                           theta2 = 0, 
                           theta1_dot = 0, 
                           theta2_dot = 0):
        
        self.initial_condition(theta1, theta2, theta1_dot, theta2_dot)
        
        #t is the different time steps
        self.t = np.linspace(0, simulation_time, number_of_points)
        self.trajectory = odeint(self.double_pendulum_step, self.u0, self.t, args=(self.m1, self.m2, self.l1, self.l2, self.g))
        
        print("Trajectory computed")
        
        return self.trajectory
    
    def get_trajectories(self):
        theta1_trajectory = self.trajectory[0, :]
        theta1_dot_trajectory = self.trajectory[1, :]
        theta2_trajectory = self.trajectory[2, :]
        theta2_dot_trajectory = self.trajectory[3, :]
    

if __name__ == "__main__":
    double_pendulum = DoublePendulum()
    m1,m2,L1,L2,g = double_pendulum.get_physical_parameters()
    trajectory = double_pendulum.compute_trajectory()
