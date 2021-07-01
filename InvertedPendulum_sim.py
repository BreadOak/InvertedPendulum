from scipy.integrate import solve_ivp
import scipy.integrate as integrate
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import os
import math
import numpy as np

def InvertedPendulum(t,x):
	D = m*L**2*(M + m*(1-np.cos(x[2])**2))
	dx_dt1 = x[1]
	dx_dt2 = (1/D)*(-m**2*L**2*g*np.cos(x[2])*np.sin(x[2]) + m*L**2*(m*L*x[3]**2*np.sin(x[2]) - d*x[1])) + m*L**2*(1/D)*u
	dx_dt3 = x[3]
	dx_dt4 = (1/D)*((M + m)*m*g*L*np.sin(x[2]) - m*L*np.cos(x[2])*(m*L*x[3]**2*np.sin(x[2]) - d*x[1])) - m*L*np.cos(x[2])*(1/D)*u
	return np.array([dx_dt1, dx_dt2, dx_dt3, dx_dt4])   

st = float(0)     # Start time(s)
et = float(100)   # End time(s)
ts = float(0.1)   # Time step(s)
g  = float(-9.81) # gravity(m/s^2)
L  = float(2)     # Length of pendulum(m)
d  = float(1)     # 
m  = float(1)     # Mass of pendulum(kg)
M  = float(10)    # Mass of cart(kg)

u = 0     # control input

x_state1_ini = 0
x_state2_ini = 0
x_state3_ini = math.pi + 0.01
x_state4_ini = 0
x_state_ini = np.array([x_state1_ini, x_state2_ini, x_state3_ini, x_state4_ini])

t_span = [st, et+ts]
t = np.arange(st, et+ts, ts)
sim_points = len(t)
l = np.arange(0, sim_points, 1)

x_state = solve_ivp(InvertedPendulum, t_span, x_state_ini, t_eval = t)
x   = x_state.y[0,:] #(m)
dx  = x_state.y[1,:] #(m/s)
th  = x_state.y[2,:] #(rad)
dth = x_state.y[3,:] #(rad/s)
y = 0.5              #(m)

## simulation
fig = plt.figure()
for point in l:
	plt.plot( L*math.sin(th[point]) + x[point], -L*math.cos(th[point]) + y, 'bo' )
	plt.plot( [x[point], L*math.sin(th[point]) + x[point]], [y, -L*math.cos(th[point]) + y] )
	plt.plot( [x[point]-1,x[point]+1], [y+0.5, y+0.5],'r')
	plt.plot( [x[point]-1,x[point]+1], [y-0.5, y-0.5],'r')
	plt.plot( [x[point]-1,x[point]-1], [y-0.5, y+0.5],'r')
	plt.plot( [x[point]+1,x[point]+1], [y-0.5, y+0.5],'r')
	plt.xlim(-10,10)
	plt.ylim(-1,10)
	plt.xlabel('x-direction')
	plt.ylabel('y-direction')
	plt.pause(0.01)	 
	fig.clear()
plt.draw()