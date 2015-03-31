''' 
Viterbi Algorithm for Image Segmentation
'''
import random
import pylab
from scipy import misc
from scipy import ndimage
import pypmc
import numpy as np
from math import sqrt, pi
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import math
import os
import sys
from scipy import interpolate
from pypmc.tools._probability_densities import unnormalized_log_pdf_gauss
from weightinterpolation import *

##### 
# PARAMETERS
#####
ENERGYDELTATHRESH = 100
NUMITERATIONS = 20
fname = 'nikelogo.png'
alpha = 100.0
beta = 0.1
NUMVERTICES = 45
NORMLENGTH = 5
NORMSPACING = 3
GAUSSIANMEAN = 6
INITIALRADIUS = 100
k_s = []

#Constants for Importance Sampling
SIGMA_S = 0.5
TOTAL_NUMBER_OF_SAMPLE = 250.0
DELTA_V_MIN = 0.5
DELTA_V_MAX = 8
initial_sigma = 0.5
c1 = 1/math.sqrt(2*math.pi)

image = misc.imread(fname)


originalimage = ndimage.filters.gaussian_filter(image, GAUSSIANMEAN)
image = ndimage.filters.gaussian_filter(image, GAUSSIANMEAN)

def replaceDuplicates(vertices):
    #print "vertices for removing dupes", vertices, len(vertices)
    seen = {}
    for i in range(0,len(vertices)):
        v = vertices[i]
        if v not in seen:
            seen[v] = 1
        else:
            if v != vertices[(i+1)%len(vertices)]:
                vertices[i] = midpoint(v,vertices[(i+1)%len(vertices)])
            else:
                vertices[i] = midpoint(v,vertices[(i+2)%len(vertices)])
    return vertices
    

def unitvector(head,tail):
    dx = head[0]-tail[0]; dy = head[1]-tail[1];
    return (dx/2,dy/2)
    
def midpoint(v1,v2):
    return (int((v1[0]+v2[0])/2),int((v1[1]+v2[1])/2))
def previousVertexIndex(i):
    if i-1>=0:
        return i-1
    else:
        return NUMVERTICES-1
def nextVertexIndex(i):
    if i+1<NUMVERTICES:
        return i+1
    else:
        return 0
def createCirclePoints(center, numvertices,radius):
    points = []
    pi = math.pi
    for i in range(0,numvertices):
        points.append( (int(center[0]+radius*math.cos(2*i*pi/numvertices)),int(center[1]+radius*math.sin(2*i*pi/numvertices)) ))
    return points   
def intensity(x,y):
    if x < 0:
        x = 0
    if x >= width:
        x = width-1
    if y < 0:
        y = 0
    if y >= height:
        y = height-1
    return originalimage[x][y]
def externalE(p):
    x = p[0]; y = p[1];
    def i(x,y):
        return intensity(x,y)
    e = -math.sqrt((i(x+2,y)-i(x-2,y))**2+(i(x,y+2)-i(x,y-2))**2+(i(x+1,y+2)-i(x-1,y-2))**2 +(i(x+2,y+2)-i(x-2,y-2))**2+(i(x+1,y+1)-i(x-1,y-1))**2+(i(x+2,y+1)-i(x-2,y-1))**2          )  
    return e*(i(x,y)**2/255.0**2)
def internalE(previous, current, after):
    return ((current[0]-previous[0])**2+(current[1]-previous[1])**2)\
    +((after[0]-2*current[0]+previous[0])**2+(after[1]-2*current[1]+previous[1])**2)
def length(v):
    return math.sqrt(v[0]**2+v[1]**2)

def importance_sampling(vertices):
    # Create vectors from vertices and find the angle between them.
    print vertices
    angles = []
    for i in range(0, len(vertices)):
        if(i == 0):
            v_1 = np.array(vertices[len(vertices) - 1])
            v_2 = np.array(vertices[i + 1])
        elif(i == len(vertices) - 1):
            v_1 = np.array(vertices[i - 1])
            v_2 = np.array(vertices[0])
        else:
            v_1 = np.array(vertices[i - 1])
            v_2 = np.array(vertices[i + 1])
        v = np.array(vertices[i])
        vec1 = v_1 - v
        vec2 = v_2 - v
        angles.append(0.1 + pi - np.arccos(np.dot(vec1, vec2) / math.ceil(length(vec1) * length(vec2))))
    return angles

def weight_points(angles):
    weight = []
    for i in range(0, len(angles)):
        # x = (angles[i] * angles[i]) - 4
        weight.append(1 / (1 + math.exp(angles[i])))
    return weight  
    


height = len(image[0])
width = len(image)
center = (width/2, height/2)
vertices = createCirclePoints(center, NUMVERTICES,INITIALRADIUS)
iterationnum = 0


while iterationnum < NUMITERATIONS:
    # ITERATION OF WHOLE ALGORITHM
    iterationnum += 1
    image = misc.imread(fname)
    image = ndimage.filters.gaussian_filter(image, GAUSSIANMEAN)
    NUMVERTICES = len(vertices)
    print "Number of vertices at BEGINNING OF ITERATION: ",NUMVERTICES

    # Initialize normals
    normals = [[0 for x in range(NORMLENGTH)] for x in range(NUMVERTICES)]

    for v in range(0,NUMVERTICES):
        image[vertices[v][0]][vertices[v][1]] = 0
        for i in range(0,NORMLENGTH):
            deltax = center[0]-vertices[v][0]; deltay = center[1]-vertices[v][1];
            magnitude = math.sqrt(deltax**2+deltay**2)
            direction = (deltax/magnitude,deltay/magnitude)
            normals[v][i] = (int(vertices[v][0]+NORMSPACING*i*direction[0]),int(vertices[v][1]+NORMSPACING*i*direction[1]))
            image[normals[v][i][0]][normals[v][i][1]] = 0

    plt.imshow(image, cmap = cm.Greys_r)
    plt.show()
    image = misc.imread(fname)
    image = ndimage.filters.gaussian_filter(image, GAUSSIANMEAN)

    # Fill the Dynamic Programming table
    trellisEnergies = [[[0 for x in range(NORMLENGTH)] for x in range(NORMLENGTH)] for x in range(NUMVERTICES)]
    optimalpath = []

    for i in range(0,NORMLENGTH):
        for j in range(0,NORMLENGTH):
            trellisEnergies[0][i][j] = 0

    for i in range(0,NUMVERTICES):
        lowestenergypoint = None
        lowestenergy = 999999999999999999
        for point2index in range(0,NORMLENGTH):
            k = (i+1)%NUMVERTICES
            j = (i+2)%NUMVERTICES
            point2 = normals[k][point2index]
            for point3index in range(0,NORMLENGTH):
                point3 = normals[j][point3index]
                energies = []
                for point1index in range(0,NORMLENGTH):
                    point1 = normals[i][point1index]
                    energies.append(trellisEnergies[i][point2index][point1index]+alpha*externalE(point2)+beta*internalE(point1,point2,point3))
                    #print energies, energies.index(min(energies))
                trellisEnergies[k][point2index][point3index] = min(energies)
                if min(energies) < lowestenergy:
                    lowestenergy = min(energies)
                    lowestenergypoint = normals[i][point2index]
        image[lowestenergypoint[0]][lowestenergypoint[1]] = 200
        optimalpath.append((lowestenergypoint[0],lowestenergypoint[1]))
            

    optimalpath = replaceDuplicates(optimalpath)
    
    curv_angle = importance_sampling(optimalpath)
    weight_arr = weight_points(curv_angle)

    vertices = optimalpath
    newvertices = []
    for v in range(0,len(vertices)):
        pointweightpairs = []
        beginvertexweight = (vertices[v],weight_arr[v])
        endvertexweight = (vertices[nextVertexIndex(v)],weight_arr[nextVertexIndex(v)])
        pointweightpairs.append(beginvertexweight)
        pointweightpairs.append(endvertexweight)
    
        # Return  input number of corresponding weights that are exponentially distributed from 0 to vertex v's weight
        # num points should be half the number of points along v_i -> v_i+1 i<n, else v_n -> v_1
        halfnumsegmentpoints = int(sqrt( float((vertices[v][0] - midpoint(vertices[v],vertices[nextVertexIndex(v)])[0])**2 + (vertices[v][1] - midpoint(vertices[v],vertices[nextVertexIndex(v)])[1])**2))/2)
        weights = interpolateWeights(halfnumsegmentpoints, weight_arr[v]); weights.reverse();
        weights2 = interpolateWeights(halfnumsegmentpoints, weight_arr[nextVertexIndex(v)]); weights2.reverse();
        vdir = unitvector(vertices[nextVertexIndex(v)],vertices[v])
        vdirx = vdir[0]; vdiry = vdir[1];
        for i in range(0,halfnumsegmentpoints):
            pointweightpairs.append((  (int(vertices[v][0]+float(i)/halfnumsegmentpoints*vdirx),int(vertices[v][1]+float(i)/halfnumsegmentpoints*vdiry)),weights[i]))
            pointweightpairs.append((  (int(vertices[nextVertexIndex(v)][0]-float(i)/halfnumsegmentpoints*vdirx),int(vertices[nextVertexIndex(v)][1]-float(i)/halfnumsegmentpoints*vdiry)),weights2[i]))
            p1 = pointweightpairs[len(pointweightpairs)-1][0]; p2 = pointweightpairs[len(pointweightpairs)-2][0]; 
            image[p1[0]][p1[1]] = 0
            image[p2[0]][p2[1]] = 0
        # Random choice
        total = 0
        for p,w in pointweightpairs:
            total += w
        target = random.random()*total
        sumsofar = 0
        for p,w in pointweightpairs:
            if target > sumsofar and target < sumsofar + w:
                newvertices.append(p)
                break
            sumsofar += w

    vertices = newvertices


    plt.imshow(image, cmap = cm.Greys_r)
    plt.show()
    

    
