# -*- coding: Latin-1 -*-
# TP optim : maximisation of the area
# par l'algorithme PSO
# Peio Loubiere & Rachid Chelouah pour l'EISTI
# septembre 2017
# usa : python yourMethod.py
from scipy import *
from math import *
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import sys
import pyclipper
from pyclipper import scale_to_clipper, scale_from_clipper
import sim_annealing
import particle_swarm
from functools import partial
from functools import reduce

# Visualization of the parcel
fig = plt.figure()
canv = fig.add_subplot(1,1,1)

# ************ Parameters of metaheuristics *************
dd = 10
da = 0.1
# ***********************************************************

# ***************** Problem settings ******************
#  Different proposals of parcels:
# polygon = ((10,10),(10,400),(400,400),(400,10))
# polygon = ((10,10),(10,300),(250,300),(350,130),(200,10))
# polygon = ((50,150),(200,50),(350,150),(350,300),(250,300),(200,250),(150,350),(100,250),(100,200))
# polygon = ((50,50),(50,400),(220,310),(220,170),(330,170),(330,480),(450,480),(450,50))

# ***********************************************************

# Transform the  polygon in list for display.
def poly2list(polygon):
    polygonfig = list(polygon)
    polygonfig.append(polygonfig[0])
    return polygonfig

def make_patch(poly):
    poly = poly2list(poly)
    codes = [Path.MOVETO]
    for i in range(len(poly)-2):
      codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    path = Path(poly, codes)
    patch = patches.PathPatch(path, facecolor='orange', alpha=0.4, lw=2)
    return patch


# Display window
def draw(polygone,sol):
    global canv, codes
    canv.clear()
    canv.set_xlim(0,500)
    canv.set_ylim(0,500)
    canv.add_patch(make_patch(polygone))
    canv.add_patch(make_patch(pos2rect(sol)))
    # Title display (rectangle area)
    plt.title("area : {}".format(round(area(sol),2)))

    plt.draw()
    plt.pause(0.1)

def partialDraw(sol):
    global canv, codes
    canv.add_patch(make_patch(pos2rect(sol)))
    plt.title("area : {}".format(round(area(sol),2)))

def drawNew(polygone):
    global canv, codes
    canv.clear()
    canv.set_xlim(0,500)
    canv.set_ylim(0,500)
    canv.add_patch(make_patch(polygone))
    plt.draw()
    plt.pause(0.1)

def flush():
    plt.draw()
    plt.pause(0.1)


# Collect bounding box bounds around the parcel
def getBounds(polygon):
    xmin = reduce(lambda acc, e: min(acc,e[0]),polygon[1:],polygon[0][0])
    xmax = reduce(lambda acc, e: max(acc,e[0]),polygon[1:],polygon[0][0])
    ymin = reduce(lambda acc, e: min(acc,e[1]),polygon[1:],polygon[0][1])
    ymax = reduce(lambda acc, e: max(acc,e[1]),polygon[1:],polygon[0][1])
    return xmin, xmax, ymin, ymax

# Transformation of a problem solution into a rectangle for clipping
# Return the rectangle (A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4))
def pos2rect(sol):
    x, y, angle, width, length = sol
    alpha = atan(width / length)
    vec = sqrt(width**2 + length**2)
    a = (x + vec*cos(angle+alpha), y + vec*sin(angle+alpha))
    b = (x + vec*cos(angle-alpha), y + vec*sin(angle-alpha))
    c = (x + vec*cos(pi+angle+alpha), y + vec*sin(pi+angle+alpha))
    d = (x + vec*cos(pi+angle-alpha), y + vec*sin(pi+angle-alpha))
    return a, b, c, d

# Distance between two points (x1,y1), (x2,y2)
def distance(p1,p2):
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

# Area of the rectangle (A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4))
# 	= distance AB * distance BC
def area(sol):
    pa, pb, pc, pd = pos2rect(sol)
    return distance(pa, pb) * distance(pb, pc)


def almost_there(p1, p2):
    return distance(p1, p2) < 0.1
# Clipping
# Predicate that verifies that the rectangle is in the polygon
# Test if
# 	- there is an intersection (!=[]) between the figures and
#	- both lists with the same length
# 	- all the points of the rectangle belong to the result of clipping
# If error (~flat angle), return false
def verifConstraint(sol, polygon):
    pc = pyclipper.Pyclipper()
    rect = pos2rect(sol)
    pc.AddPath(scale_to_clipper(polygon), pyclipper.PT_CLIP, True)
    pc.AddPath(scale_to_clipper(rect), pyclipper.PT_SUBJECT, True)
    solution = scale_from_clipper(pc.Execute(pyclipper.CT_INTERSECTION,
                    pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD))
    if (len(solution) < 1): return False
    if len(solution[0]) != 4:
        return False
    for point in rect:
        flag = False
        for point2 in solution[0]:
            if almost_there(point, point2):
                flag = True
        if flag == False:
            return False
    return True

# Creates a feasible particle (solution)
# ua particle is described by your metaheuristics :
# 	- pos : solution list of variables
#	- eval :  rectangle area
#	- ... : other components of the solution
def initOne(polygon):
    res = None
    xmin, xmax, ymin, ymax = getBounds(polygon)
    diam = min(xmax-xmin, ymax-ymin)
    while res == None:
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        alpha = random.uniform(0, 2*pi)
        width = random.uniform(0, diam/2)
        length = random.uniform(0, diam/2)
        res = x, y, alpha, width, length
        if not verifConstraint(res, polygon): res = None
    draw(polygon, res)
    return res

counter = 200
def fluctuation(polygon, sol):
    global counter
    counter -= 1
    if counter < 0:
        draw(polygon, sol)
        counter = 1000
    res = None
    tries = 10
    while res == None and tries > 0:
        tries -= 1
        x, y, angle, width, length = sol
        x += random.uniform(-dd, +dd)
        y += random.uniform(-dd, +dd)
        angle += random.uniform(-da, +da)
        width += random.uniform(-dd, +dd)
        length += random.uniform(-dd, +dd)
        res = x, y, angle, width, length
        if not verifConstraint(res, polygon): res = None
    return res if tries > 0 else sol

params = {}

# ------------------------------------------------------------

# polygon = ((10,10),(10,400),(400,400),(400,10))
# params = {"T0":1000, "MaxIter":5000}

# polygon = ((10,10),(10,300),(250,300),(350,130),(200,10))
# params = {"T0":5000, "MaxIter":5000}

# polygon = ((50,150),(200,50),(350,150),(350,300),(250,300),(200,250),(150,350),(100,250),(100,200))
# params = {"T0":5000, "Alpha":0.9997, "IterMax":8000}

polygon = ((50,50),(50,400),(220,310),(220,170),(330,170),(330,480),(450,480),(450,50))
params = {"T0":5000, "Alpha":0.9999, "IterMax":10000}

# ------------------------------------------------------------

task = partial(initOne, polygon), partial(fluctuation, polygon), lambda x:-area(x)

params["psi"] = 0.5
params["cmax"] = 1.3
particle_task = partial(initOne, polygon), partial(lambda x, y :verifConstraint(y,x), polygon), lambda x:-area(x), partialDraw, partial(drawNew, polygon), flush

# SIMULATED ANNEALING
# solution = sim_annealing.solve(*task, **params)

# PARTICLE SWARM
particle_swarm.solve(*particle_task, **params)
