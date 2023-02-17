import matplotlib.pyplot as plt
from matplotlib import animation, rc
import numpy as np

def make_anim(double_pendulum, data2):
    L1 = double_pendulum.l1
    L2 = double_pendulum.l2
    Nt = double_pendulum.number_of_points
    t = double_pendulum.t
    
    def init2():
        line21.set_data([], [])
        line22.set_data([], [])
        line23.set_data([], [])
        line24.set_data([], [])
        line25.set_data([], [])
        time_string.set_text('')
        return  line23,line24, line25, line21, line22, time_string

    def animate2(i):
        # Motion trail sizes. Defined in terms of indices. Length will vary with the time step, dt. E.g. 5 indices will span a lower distance if the time step is reduced.
        trail1 = 6              # length of motion trail of weight 1 
        trail2 = 8              # length of motion trail of weight 2

        dt = t[2]-t[1]          # time step

        line21.set_data(xp1[i:max(1,i-trail1):-1], yp1[i:max(1,i-trail1):-1])   # marker + line of first weight
        line22.set_data(xp2[i:max(1,i-trail2):-1], yp2[i:max(1,i-trail2):-1])   # marker + line of the second weight

        line23.set_data([xp1[i], xp2[i]], [yp1[i], yp2[i]])       # line connecting weight 2 to weight 1
        line24.set_data([xp1[i], 0], [yp1[i],0])                # line connecting origin to weight 1

        line25.set_data([0, 0], [0, 0])
        time_string.set_text(time_template % (i*dt))
        return  line23, line24,line25,line21, line22, time_string

    fig2 = plt.figure()
    ax2 = plt.axes(xlim=(-L1-L2-0.5, L1+L2+0.5), ylim=(-2.5, 1.5))
    #line, = ax.plot([], [], lw=2,,markersize = 9, markerfacecolor = "#FDB813",markeredgecolor ="#FD7813")
    line21, = ax2.plot([], [], 'o-',color = '#d2eeff',markersize = 12, markerfacecolor = '#0077BE',lw=2, markevery=10000, markeredgecolor = 'k')   # line for Earth
    line22, = ax2.plot([], [], 'o-',color = '#ffebd8',markersize = 12, markerfacecolor = '#f66338',lw=2, markevery=10000, markeredgecolor = 'k')   # line for Jupiter
    line23, = ax2.plot([], [], color='k', linestyle='-', linewidth=2)
    line24, = ax2.plot([], [], color='k', linestyle='-', linewidth=2)
    line25, = ax2.plot([], [], 'o', color='k', markersize = 10)
    time_template = 'Time = %.1f s'
    time_string = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)
    ax2.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
    ax2.get_yaxis().set_ticks([])    # enable this to hide y axis ticks

    u0 = data2[:,0]     # theta_1 
    u1 = data2[:,1]     # omega 1
    u2 = data2[:,2]     # theta_2 
    u3 = data2[:,3]     # omega_2 
    
    xp1 = L1*np.sin(u0);          # First Pendulum
    yp1 = -L1*np.cos(u0);

    xp2 = xp1 + L2*np.sin(u2);     # Second Pendulum
    yp2 = yp1 - L2*np.cos(u2);

    anim2 = animation.FuncAnimation(fig2, animate2, init_func=init2,
                                   frames=Nt, interval=1000*(t[2]-t[1])*0.8, blit=True)
    return anim2