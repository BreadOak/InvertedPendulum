from scipy.integrate import solve_ivp
from scipy import linalg
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

def LQR_Gain(A,B,Q,R):
	P = linalg.solve_continuous_are(A, B, Q, R)
	K = np.dot( linalg.inv(R) , (np.dot( B.T , P )) )
	eigVals, eigVecs = linalg.eig(A - np.dot(B , K))
	return K

Start_time = float(0)   
End_time = float(20)    
Sampling_Frequency = float(10) 
N = End_time*Sampling_Frequency
st = 1/Sampling_Frequency # Sampling time
Sampling_time = [0, st]
t = np.arange(Start_time, st, st/100)
l = np.arange(0, int((End_time-Start_time)/st), 1)

g  = float(-9.81) # gravity(m/s^2)
L  = float(2)     # Length of pendulum(m)
d  = float(1)     # 
m  = float(1)     # Mass of pendulum(kg)
M  = float(10)    # Mass of cart(kg)

y = 0.5   # Hight of cart's center(m)
b = 1     # pendulum up (b=1)

A = np.array([[0,          1,                  0, 0],
			  [0,       -d/M,            b*m*g/M, 0],
			  [0,          0,                  0, 1],
			  [0, -b*d/(M*L), -b*(M + m)*g/(M*L), 0]])

B = np.array([          [0],
			        [1 / M],
			            [0],
			  [b*1 / (M*L)]])

Q = np.array([[1, 0, 0, 0],
			  [0, 1, 0, 0],
			  [0, 0, 1, 0],
			  [0, 0, 0, 1]])

R = np.array([ [0.0001] ])

## Initial state
x_state1_ini = 0               #(m)
x_state2_ini = 0               #(m/s)
x_state3_ini = math.pi + 0.5   #(rad)
x_state4_ini = 0               #(rad/s)
x_state_ini = np.array([ x_state1_ini, x_state2_ini, x_state3_ini, x_state4_ini ])

## Desired state
x_state1_des = 5         #(m)
x_state2_des = 0         #(m/s)
x_state3_des = math.pi   #(rad)
x_state4_des = 0         #(rad/s)
x_state_des = np.array([ x_state1_des, x_state2_des, x_state3_des, x_state4_des ])

x_data   = []
dx_data  = []
th_data  = []
dth_data = []

K = LQR_Gain(A,B,Q,R)

for i in l:
	x_error = x_state_ini - x_state_des
	u = -np.dot(K, x_error)
	x_state = solve_ivp(InvertedPendulum, Sampling_time, x_state_ini)
	x   = x_state.y[0,:] #(m)
	dx  = x_state.y[1,:] #(m/s)
	th  = x_state.y[2,:] #(rad)
	dth = x_state.y[3,:] #(rad/s)
	x   = x[-1] 
	dx  = dx[-1] 
	th  = th[-1] 
	dth = dth[-1] 

	# save data
	x_data.append(x)
	dx_data.append(dx)
	th_data.append(th)
	dth_data.append(dth)

	x_state = [x, dx, th, dth]
	x_state_ini = x_state

## Simulation 
x  = x_data
th = th_data
fig = plt.figure()
for point in l:
	plt.plot( L*math.sin(th[point]) + x[point], -L*math.cos(th[point]) + y, 'bo' )
	plt.plot( [x[point], L*math.sin(th[point]) + x[point]], [y, -L*math.cos(th[point]) + y] )
	plt.plot( [x[point]-1,x[point]+1], [y+0.5, y+0.5],'r')
	plt.plot( [x[point]-1,x[point]+1], [y-0.5, y-0.5],'r')
	plt.plot( [x[point]-1,x[point]-1], [y-0.5, y+0.5],'r')
	plt.plot( [x[point]+1,x[point]+1], [y-0.5, y+0.5],'r')
	plt.xlim(-20,20)
	plt.ylim(-1,20)
	plt.xlabel('x-direction')
	plt.ylabel('y-direction')
	plt.pause(0.01)	 
	fig.clear()
plt.draw()