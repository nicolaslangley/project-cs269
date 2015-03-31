import numpy as np
from numpy import genfromtxt
import scipy as sci
from scipy import ndimage, linalg, signal, interpolate
import matplotlib.pyplot as plt
import time
# This is for memory analysis
import psutil
from memory_profiler import profile


# Function for converting to grayscale
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def acm(im, x, y, alpha, beta, gamma, kappa, wl, we, wt, iterations):
    ''' 
        Function for implementing traditional snakes
        NOTE: Input Image should be scaled to double
    '''
    # Parameters
    N = iterations
    # Convert to graycale if required
    if len(im.shape) > 2:
        smth = rgb2gray(im)
    else:
        smth = im

    # Change the snake starting position values
    x = np.array(x)
    y = np.array(y)

    # Calculate size of im
    [row, col] = smth.shape

    # Compute the external forces
    eline = smth
    # This only works in NumPy != 1.9.0
    [gx, gy] = np.gradient(smth, 1)
    temp_val = (gx * gx + gy * gy)
    eedge = -1 * np.sqrt((gx * gx) + (gy * gy))
    
    # Masks for derivatives
    m1 = np.mat(np.array([-1.,1.]))
    m2 = np.mat(np.array([[-1.],[1.]]))
    m3 = np.mat(np.array([1.,-2.,1.]))
    m4 = np.mat(np.array([[1.],[-2.],[1.]]))
    m5 = np.mat(np.array([[1.,-1.],[-1.,1.]]))

    # Image convolutions
    cx = ndimage.filters.convolve(smth, m1) 
    cy = ndimage.filters.convolve(smth, m2) 
    cxx = signal.convolve(smth, m3, 'same') 
    cyy = signal.convolve(smth, m4, 'same') 
    cxy = ndimage.filters.convolve(smth, m5) 

    eterm = np.zeros((row,col)) 
    for i in range(row):
        for j in range(col):
            # eterm from paper
            num_part1 = cyy[i,j]*cx[i,j]*cx[i,j]
            num_part2 = 2*cxy[i,j]*cx[i,j]*cy[i,j]
            num_part3 = cxx[i,j]*cy[i,j]*cy[i,j]
            denom = np.power(1 + cx[i,j]*cx[i,j] + cy[i,j]*cy[i,j],1.5)
            eterm[i,j] = (num_part1 - num_part2 + num_part3) / denom

    # E_ext as a weighted sum of the line, edge and termination functionals 
    eext = (wl * eline + we * eedge - wt * eterm)
    # Compute the gradient of E_ext
    [fy, fx] = np.gradient(eext, 1)

    # Initialize the snake
    x = x.conj().transpose()
    y = y.conj().transpose()

    [m, n] = np.mat(x).shape
    [mm, nn] = np.mat(fx).shape
    
    # Internal Energy penta diagonal matrix A
    # NOTE: m in Matlab corresponds to n here - indices are reversed
    A = np.zeros((n,n))
    b = np.array([(2 * alpha + 6 * beta), -(alpha + 4 * beta), beta])
    brow = np.zeros((1,n))
    brow[0,0:3] = brow[0,0:3] + b
    brow[0,(n-2):n] = brow[0,(n-2):n] + np.array([beta, -(alpha + 4 * beta)])
    for i in range(n):
        A[i,:] = brow
        brow = np.roll(brow,1)
   
    # Compute A inverse using LU decomposition
    [P, L, U] = linalg.lu(A + gamma * np.eye(n))
    Ainv = np.mat(linalg.inv(U)) * np.mat(linalg.inv(L))

    # Setup the interpolation for the snake positions
    x_range = np.arange(1, fx.shape[0] + 1)
    y_range = np.arange(1, fx.shape[1] + 1)
    fx_interp_func = interpolate.interp2d(x_range, y_range, fx)
    fy_interp_func = interpolate.interp2d(x_range, y_range, fy)

    print (psutil.virtual_memory())
    # Setup the plotting
    plt.ion()
    # Perform N iterations on the snake
    for i in range(N):
        fx_interp_result = np.array([fx_interp_func(x[i], y[i])[0] for j in range(x.shape[0])])
        fy_interp_result = np.array([fy_interp_func(x[i], y[i])[0] for j in range(x.shape[0])])
        # Apply the internal forces along with their weights
        ssx = gamma * x - kappa * fx_interp_result
        ssy = gamma * y - kappa * fy_interp_result

        # Calculate the new position of snake 
        x = Ainv * np.mat(ssx).transpose()
        y = Ainv * np.mat(ssy).transpose()

        # Convert positions into required format 
        x = np.array([x[j].tolist()[0][0] for j in range(x.shape[0])])
        y = np.array([y[j].tolist()[0][0] for j in range(y.shape[0])])
        print 'Memory usage in iteration ' + str(i+1)
        print (psutil.virtual_memory().used - initial_memory)
        
        # Display the snake in new position
        #plt.clf()
        #plt.imshow(image, cmap=plt.cm.gray)
        #plt.plot(x,y,'r-')
        #plt.draw()
        #time.sleep(0.01)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.plot(x, y, 'r-')
    plt.show()

# Define a global variables 
# NOTE: We may want to use a dict instead - for memory purposes
global x_input, y_input, n, fig, image, ax
x_input = []
y_input = []
n = 0
fig = None 
image = None
ax = None
# Function for getting starting values for snake based on point input
# NOTE: Uses a global image 
def get_snake_points():
    # Display the image as a plot
    plt.imshow(image, cmap=plt.cm.gray)
    global fig, ax
    fig = plt.gcf()
    ax = plt.gca()
    handler = EventHandler()
    plt.show()

class EventHandler:
    def __init__(self):
        fig.canvas.mpl_connect('button_press_event', self.onclick)
    
    # Callback function for determining snake points
    def onclick(self, event):
        if event.inaxes!=ax:
            return
        global n, x_input, y_input
        print n, x_input, y_input
        x_input.append(event.xdata)
        y_input.append(event.ydata)
        #plt.close()
        plt.plot(x_input,y_input, 'or')
        plt.draw()
        # TODO: Display these points on the image
        n += 1
        if event.button == 3:
            run_snakes(x_input,y_input)

def run_snakes(xs, ys):
    print "Running snakes..."
    # Set initial snake parameters
    alpha = 0.40
    beta = 0.20
    gamma = 1.00
    kappa = 0.15
    # Set the weights
    wl = 0.30
    we = 0.40
    wt = 0.70
    # Set the number of snake iterations
    iterations = 7
    print ('Alpha: {0} Beta: {1} Gamma: {2} Kappa: {3} wl: {4} we: {5} wt: {6}').format(alpha, beta, gamma, kappa, wl, we, wt)
    # Run the snakes algorithm
    # NOTE: image is a global image
    acm(image, xs, ys, alpha, beta, gamma, kappa, wl, we, wt, iterations)

if __name__ == '__main__':
    global initial_memory
    initial_memory = psutil.virtual_memory().used
    # Load test image and convert to double values
    image = ndimage.imread('nikelogo.png') / 255.
    xs = genfromtxt('xs_points1.csv', delimiter=',')
    ys = genfromtxt('ys_points2.csv', delimiter=',')
    # Determine the initial snake points
    # get_snake_points()
    run_snakes(xs,ys)

