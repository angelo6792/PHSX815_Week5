#! /usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
#################
# Random class
#################
# class that can generate random numbers
class Random:
    """A random number generator class"""

    # initialization method for Random class
    def __init__(self, seed = 5555):
        self.seed = seed
        self.m_v = np.uint64(4101842887655102017)
        self.m_w = np.uint64(1)
        self.m_u = np.uint64(1)
        
        self.m_u = np.uint64(self.seed) ^ self.m_v
        self.int64()
        self.m_v = self.m_u
        self.int64()
        self.m_w = self.m_v
        self.int64()

    # function returns a random 64 bit integer
    def int64(self):
        with np.errstate(over='ignore'):
            self.m_u = np.uint64(self.m_u * np.uint64(2862933555777941757) + np.uint64(7046029254386353087))
        self.m_v ^= self.m_v >> np.uint64(17)
        self.m_v ^= self.m_v << np.uint64(31)
        self.m_v ^= self.m_v >> np.uint64(8)
        self.m_w = np.uint64(np.uint64(4294957665)*(self.m_w & np.uint64(0xffffffff))) + np.uint64((self.m_w >> np.uint64(32)))
        x = np.uint64(self.m_u ^ (self.m_u << np.uint64(21)))
        x ^= x >> np.uint64(35)
        x ^= x << np.uint64(4)
        with np.errstate(over='ignore'):
            return (x + self.m_v)^self.m_w

    # function returns a random floating point number between (0, 1) (uniform)
    def rand(self):
        return 5.42101086242752217E-20 * self.int64()

    # function returns a random double (0 to infty) according to an exponential distribution
    def Exponential(self, beta=1.):
      # make sure beta is consistent with an exponential
      if beta <= 0.:
        beta = 1.

      R = self.rand();

      while R <= 0.:
        R = self.rand()

      X = -math.log(R)/beta

      return x



#loop Rayleigh through an empty array to create distribution
random = Random()
n=100000
a = []
for x in range(0,n):
   a.append(random.Exponential())

#plot array
n, bins, patches = plt.hist(a, 50, density=True, facecolor='g', alpha=0.75)
plt.xlabel('x', fontsize = 16, color = "blue")
plt.ylabel('Probability', fontsize = 16, color = "red")
plt.title('Exponential distribution')
plt.grid(True)

plt.show()
