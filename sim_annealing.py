from scipy import *
from math import *
import matplotlib.pyplot as plt
import sys
from functools import reduce

import time
time.millis = lambda: int(round(time.time() * 1000))

def drawStats(Htime, Henergy, Hbest, HT):
    # display des courbes d'evolution
    fig2 = plt.figure(2, figsize=(4, 6))
    plt.subplot(3,1,1)
    plt.semilogy(Htime, [-el for el in Henergy])
    plt.title("Evolution of the total energy of the system")
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.subplot(3,1,2)
    plt.semilogy(Htime, [-el for el in Hbest])
    plt.title('Evolution of the best energy')
    plt.xlabel('time')
    plt.ylabel('Best energy')
    plt.subplot(3,1,3)
    plt.semilogy(Htime, HT)
    plt.title('Evolution of the temperature of the system')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.show()

def metropolis(x_new, x_old, sysEnergy, system):
    energy_new = sysEnergy(x_new)
    energy_old = sysEnergy(x_old)
    delta = energy_new - energy_old
    # print(delta)
    if delta <= 0: # if improving,
        if energy_new <= system['best_energy']: # comparison to the best, if better, save and refresh the figure
            system['best_energy'] = energy_new
            system['best_point'] = x_new
        return (x_new, energy_new) # the fluctuation is retained, returns the neighbor
    else:
        if random.uniform() > exp(-delta/system['T']): # the fluctuation is not retained according to the proba
            return (x_old, energy_old)              # initial path
        else:
            return (x_new, energy_new)              # the fluctuation is retained, returns the neighbor

def solve(init, fluctuation, sysEnergy, **params):
    T0 = params.get('T0', 150) # initial temperature
    Tmin = params.get('Tmin', 1e-2) # final temperature
    tau = params.get('tau', 1e4) # constant for temperature decay
    Alpha = params.get('Alpha', 0.9995) # constant for geometric decay
    Step = params.get('Step', 9) # number of iterations on a temperature level
    IterMax = params.get('IterMax', 15000) # max number of iterations of the algorithm

    # initializing history lists for the final graph
    Henergy = []     # energy
    Htime = []       # time
    HT = []           # temperature
    Hbest = []        # distance

    # ######################################### INITIALIZING THE ALGORITHM ####### #####################

    x = init()
    energy = sysEnergy(x)
    system = dict()
    system['best_point'] = x
    system['best_energy'] = energy

    # main loop of the annealing algorithm
    t = 0
    system['T'] = T0
    iterStep = Step

    # ############################################ PRINCIPAL LOOP OF THE ALGORITHM ###### ######################
    millis = time.millis()
    # Convergence loop on criteria of number of iteration (to test the parameters)
    for i in range(IterMax):
    # Convergence loop on temperature criterion
    #while T> Tmin:
         # cooling law enforcement
        while (iterStep > 0):
          # choice of two random cities
            new = fluctuation(x)
            energy_new = sysEnergy(new)
            # application of the Metropolis criterion to determine the persisted fulctuation
            (x, energy) = metropolis(new, x, sysEnergy, system)
            iterStep -= 1

        # cooling law enforcement
        t += 1
        # rules of temperature decreases
        # T = T0*exp(-t/tau)
        system['T'] = system['T']*Alpha
        iterStep = Step

        #historization of data
        if t % 2 == 0:
            Henergy.append(energy)
            Htime.append(t)
            HT.append(system.get('T'))
            Hbest.append(system.get('best_energy'))

    new_millis = time.millis()
    print("Time of execution: {}ms.".format((new_millis-millis)))
    ############################################## END OF ALGORITHM - DISPLAY RESULTS ### #########################
    drawStats(Htime, Henergy, Hbest, HT)

    print("Best point is ", system['best_point'])
    print("Area is ", -system['best_energy'])

    return (system, (Htime, Henergy, Hbest, HT))
