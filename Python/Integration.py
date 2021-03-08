#! /usr/bin/env python

# imports of external packages to use in our code
import sys
import numpy as np
import math
import sympy as sp
from scipy.integrate import quad
from scipy import integrate

#integrates exactly
def f(x):
    return x**4+x**3+x**2+x

def trapazoid(f,a,b,N=50):
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
t = trapazoid(lambda x : x**4+x**3+x**2+x,0,1,100)
print(t)

#gauss
f = lambda x: x**4+x**3+x**2+x
g, err = integrate.quadrature(f, 0, 1)
print(g)
