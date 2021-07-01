from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete
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

def MPC_Gain(Ad,Bd,Cd,Dd,Nc,Np):
	# Augmented state-space model
	m1, n1 = Cd.shape
	n1, n_in = Bd.shape
	Ae = np.identity(n1+m1)
	Ae[0:n1, 0:n1] = Ad
	Ae[n1:n1+m1, 0:n1] = np.dot(Cd, Ad) 
	Be = np.zeros((n1+m1, n_in))
	Be[0:n1, :] = Bd
	Be[n1:n1+m1, :] = np.dot(Cd, Bd)
	Ce = np.zeros((m1,n1+m1))
	Ce[:, n1:n1+m1] = np.identity(m1)

	h_row, h_col = Ce.shape
	F_row, F_col = np.dot(Ce, Ae).shape
	Phi_row, Phi_col = np.dot(Ce, Be).shape

	n = n1 + m1
	h = np.zeros((h_row*Np, h_col))
	F = np.zeros((F_row*Np, F_col))

	h[0:h_row, :] = Ce
	F[0:F_row, :] = np.dot(Ce, Ae)

	for kk in range(1, Np):
		hk = kk*h_row
		Fk = kk*F_row
		h[hk:hk+h_row , :] = np.dot(h[hk-h_row:hk , :], Ae)
		F[Fk:Fk+F_row , :] = np.dot(F[Fk-F_row:Fk , :], Ae)

	v = np.zeros((Phi_row*Np, Phi_col)) 

	for nn in range(0, Np):
		vn = nn*Phi_row
		v[vn:vn+Phi_row , :] = np.dot( h[vn:vn+Phi_row , :] ,Be)  

	Phi = np.zeros((Phi_row*Np, Phi_col*Nc)) # Declare the dimension of Phi
	Phi[:, 0:Phi_col] = v                    # First column of Phi

	for i in range(1, Nc):
		ic = i*Phi_col
		ir = i*Phi_row
		Phi[:, ic:ic+Phi_col] = np.vstack([ np.zeros((ir, Phi_col)), v[0:Phi_row*Np-ir, :] ])

	BarRs = np.zeros((h_row*Np,h_row))
	for j in range(0, Np):
		Bj = j*h_row
		BarRs[Bj:Bj+h_row , :] = np.identity(h_row)

	Phi_Phi = np.dot(Phi.T, Phi)
	Phi_F = np.dot(Phi.T, F)	
	Phi_R = np.dot(Phi.T, BarRs)

	return Phi_Phi, Phi_F, Phi_R

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
u = 0     # Control input

A = np.array([[0,          1,                  0, 0],
			  [0,       -d/M,            b*m*g/M, 0],
			  [0,          0,                  0, 1],
			  [0, -b*d/(M*L), -b*(M + m)*g/(M*L), 0]])

B = np.array([          [0],
			        [1 / M],
			            [0],
			  [b*1 / (M*L)]])

C = np.array([[1, 0, 0, 0],
			  [0, 0, 1, 0] ])

D = np.array([ [0] ])

# Transform a continuous to a discrete state-space model
Ad, Bd, Cd, Dd, dt = cont2discrete((A, B, C, D), st)

Nc = 5
Np = 100
rw = 0.5

Phi_Phi, Phi_F, Phi_R = MPC_Gain(Ad,Bd,Cd,Dd,Nc,Np)
BarR = rw*np.identity(Nc)

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

## Initial value
xm0 = np.array([ [x_state_ini[0]], [x_state_ini[1]], [x_state_ini[2]], [x_state_ini[3]] ])
xm  = np.dot(Ad, xm0) + np.dot(Bd, u)
Y   = np.dot(Cd, xm)
Xf  = np.vstack([xm-xm0 , Y])
r   = np.vstack([ x_state_des[0], x_state_des[2] ])

for i in l:
	Delta_U = np.dot( np.linalg.pinv(Phi_Phi+BarR) , (np.dot(Phi_R,r) - np.dot(Phi_F,Xf)) )
	delta_u = Delta_U[0,0] # Single input 
	u = u + delta_u

	prev_xm = xm
	x_state_ini = np.concatenate(xm.T).tolist()
	x_state = solve_ivp(InvertedPendulum, Sampling_time, x_state_ini)
	x   = x_state.y[0,:]   #(m)
	dx  = x_state.y[1,:]   #(m/s)
	th  = x_state.y[2,:]   #(rad)
	dth = x_state.y[3,:]   #(rad/s)
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
	xm = np.array([x_state]).T
	Y = np.dot(Cd, xm)
	Xf = np.vstack([ xm - prev_xm, Y ])

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