# -*- coding: Latin-1 -*-
# TP optim : minimizing a function
# par l'algorithme PSO
# Peio Loubiere & Rachid Chelouah pour l'EISTI
# septembre 2017
# usage : python PSO.py
from scipy import *
from math import *
from matplotlib.pyplot import *
from functools import *
import sys


# Init of the population
def initPop(nb, init, sysEnergy):
    pop = [init() for i in range(nb)]
    return [{"pos":el, "bestpos":el, "energy":sysEnergy(el),
        "vit":[0]*len(el), "bestenergy":sysEnergy(el)} for el in pop]

# Returns the best particle depends on the metaheuristic
def best(p1, p2, sysEnergy):
    return p1 if sysEnergy(p1["pos"]) < sysEnergy(p2["pos"]) else p2

# Return a copy of the best particle of the population
def getBest(population, sysEnergy):
    return dict(reduce(lambda acc, e: best(acc, e, sysEnergy),population[1:],population[0]))

# Figure des graphes de :
#   - l'ensemble des energies des fluctuations retenues
#   - la meilleure energie
def drawStats(Htemps, Hbest):
    # afFILEhage des courbes d'evolution
    fig2 = figure(2)
    subplot(1,1,1)
    semilogy(Htemps, [-el for el in Hbest])
    title('Evolution of the best distance')
    xlabel('Time')
    ylabel('Distance')
    show()

# Update information for the particles of the population (swarm)
def update(particle,bestParticle):
    nv = dict(particle)
    if(particle["energy"] < particle["bestenergy"]):
        nv['bestpos'] = particle["pos"][:]
        nv['bestenergy'] = particle["energy"]
    nv['bestvois'] = bestParticle["bestpos"][:]
    return nv

#TODO
def limiting(position, velocity, validate):
    newpos = [p+v for p, v in zip(position, velocity)]
    if not validate(newpos):
        velocity = [-vel for vel in velocity]
        return position, velocity
    return newpos, velocity

#TODO
# Calculate the velocity and move a paticule
def move(particle, cmax, psi, validate, sysEnergy):
    nv = dict(particle)
    dim = len(particle["pos"])
    velocity = [0]*dim
    for i in range(dim):
        velocity[i] = (particle["vit"][i]*psi + \
        cmax*random.uniform()*(particle["bestpos"][i] - particle["pos"][i]) + \
        cmax*random.uniform()*(particle["bestvois"][i] - particle["pos"][i]))
    position = list(particle["pos"])
    position, velocity = limiting(position, velocity, validate)
    nv["vit"] = velocity
    nv["pos"] = position
    nv["energy"] = sysEnergy(position)
    return nv


def solve(init, validate, sysEnergy, drawThing, drawNew, flush, **params):
    Nb_cycles = params.get("Nb_cycles", 500)
    Nb_particle = params.get("Nb_particle", 20)
    psi = params.get("psi", 0.7)
    cmax = params.get("cmax", 1.47)

    Htemps = []       # temps
    Hbest = []        # distance

    # initialization of the population
    swarm = initPop(Nb_particle, init, sysEnergy)
    # initialization of the best solution
    best = getBest(swarm, sysEnergy)
    best_cycle = best

    for i in range(Nb_cycles):
        #Update informations
        swarm = [update(e,best_cycle) for e in swarm]
        # velocity calculations and displacement
        swarm = [move(e, cmax, psi, validate, sysEnergy) for e in swarm]
        # Update of the best solution
        best_cycle = getBest(swarm, sysEnergy)
        if (best_cycle["bestenergy"] < best["bestenergy"]):
            best = best_cycle
            # draw(best['pos'], best['fit'])

        # historization of data
        if i % 10 == 0:
            Htemps.append(i)
            Hbest.append(best['bestenergy'])

        # swarm display
        if i % 20 == 0:
            drawNew()
            for el in swarm:
                drawThing(el["pos"])
            flush()

    # END, displaying results
    Htemps.append(i)
    Hbest.append(best['bestenergy'])
    print("Best area is ", -best['bestenergy'])
    drawStats(Htemps, Hbest)
    return best
