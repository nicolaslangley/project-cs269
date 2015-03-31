from memory_profiler import profile
import numpy as np
from scipy.interpolate import griddata

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

@profile
def test_fun():
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
    points = np.random.rand(1000, 2)
    values = func(points[:,0], points[:,1])
    grid = griddata(points, values, (grid_x, grid_y), method='cubic')

if __name__ == '__main__':
    test_fun()

