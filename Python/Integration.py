#! /usr/bin/env python

# imports of external packages to use in our code
import sys
import numpy as np
import math
import sympy as sp
from sympy import Symbol, integrate, exp, oo
from scipy.integrate import quad
from scipy import integrate


def f(x):
    return x**4+x**3+x**2+x

def trapezoid(f,a,b,N=50):
    x = np.linspace(a,b,N+1) # N+1 points make N subintervals
    y = f(x)
    y_right = y[1:] # right endpoints
    y_left = y[:-1] # left endpoints
    dx = (b - a)/N
    T = (dx/2) * np.sum(y_right + y_left)
    return T

#exact
i, err = quad(f,0,1)
print(i)

#trapazoid
t = trapezoid(lambda x : x**4+x**3+x**2+x,0,1,1)
print(t)

#gauss
f = lambda x: x**4+x**3+x**2+x
g, err = integrate.quadrature(f, 0, 1)
print(g)


# Integration limits for the Trapezoidal rule
a = 0; b = 1
# define x as a symbol to be used by sympy
x = Symbol('x')
# find result from sympy
exact = i
# set up the arrays for plotting the relative error
n = np.zeros(40); Trapez = np.zeros(4); LagGauss = np.zeros(4);
# find the relative error as function of integration points
for i in range(1, 3, 1):
    npts = 10**i
    n[i] = npts
    Trapez[i] = abs((trapezoid(f,a,b,N=1)-exact)/exact)

print("Integration points=", n[0], n[1])
print("Trapezoidal relative error=", Trapez[0], Trapez[1])

