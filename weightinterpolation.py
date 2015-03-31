from math import exp
import numpy as np
from scipy.optimize import curve_fit

NUMVERTICES = 10


# Return len(points) number of corresponding weights that are exponentially distributed
def interpolateWeights(numpoints, maxweight):
    # params: independent var, p1, p2...
    def expfn(x,a):
        return np.exp(a*x)-1
    end = NUMVERTICES
    xs = [0, 1]
    ys = [0, maxweight]
    popt, pcov = curve_fit(expfn, xs, ys)
    
    # generate the weights, return them
    pl = numpoints-1
    weights = [0]
    for i in range(1,pl+1):
        weights.append(exp(float(i)/pl*popt[0])-1)
    return weights
