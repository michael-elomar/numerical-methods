import numpy as np
import matplotlib.pyplot as plt


def simulated_annealing(f,x,T,k_max):
    def temp(k,alpha,t0):
        return alpha**k*t0
    def newCandidate(LB,UB,temp,x):
        # temp is the current simulated temperature
        lb = max(x - temp*(UB-LB),LB)
        ub = min(x + temp*(UB-LB),UB)
        return lb + (ub-lb)*np.random.rand(1)
    y = f(x)
    x_best,y_best = x,y
    k = 0
    while k <= k_max:
        current_temp = temp(k,0.9,T)

        new_x = newCandidate(-2,2,current_temp,x)
        new_y = f(new_x)
        delta_y = new_y - y
        if  delta_y <= 0 or np.random.rand(1) < np.exp(-delta_y / current_temp):
            x = new_x
            y = new_y
        if y < y_best:
            x_best = x
            y_best = y
        k += 0.5
    return x_best,y_best

"""

# this section here is to test the SA algorithm on a function with many local extrema
# uncomment and run to see the demonstration

def f(x):
    return np.sin(x**2 - x) - np.cos(5*x - 2)**2

def df(x):
    return (2*x - 1)*np.cos(x**2 - x) + 5*np.sin(10*x - 4)

x = np.linspace(-2,2,1000)
y = f(x)

plt.figure(0,figsize=(10,5))
plt.grid(True)
plt.plot(x,y,color = 'tab:blue')

x0 = x[np.random.randint(0,1000)]
plt.scatter(x0,f(x0),color = 'tab:red')
    
x_best,y_best = simulated_annealing(f,x0,1000,1000)

x_opt = x0
for k in range(1000):
    x_opt -= 0.01*df(x_opt)
plt.scatter(x_best,y_best,color = 'tab:green',label = 'simulated annealing')
plt.scatter(x_opt,f(x_opt),color = 'tab:purple',label = 'gradient descent')

plt.show()
"""
