import numpy as np
import pandas as pd
import sys
import copy
import random

from rank_preferences import *
from correlations import *
from weighting_methods import *
from spotis import SPOTIS


class DE_algorithm():
    def __init__(self):
        self.varMin = sys.float_info.epsilon
        self.varMax = 1
        self.maxIt = 200
        self.nPop = 60
        self.beta_min = 0.2
        self.beta_max = 0.8
        self.pCR = 0.4


    def __call__(self, pd_weights, X_train, y_train, types, bounds):
        self.varSize = np.shape(X_train)[1]
        return DE_algorithm._de_algorithm(self, pd_weights, X_train, y_train, types, bounds)


    def _FitnessFunction(self, matrix, weights, types, bounds, rank_ref):
        spotis = SPOTIS()
        pref = spotis(matrix, weights, types, bounds)
        rank = rank_preferences(pref, reverse = False)
        return spearman(rank, rank_ref)


    def _generate_population(self, X_train, y_train, types, bounds):

        # Initialize population
        class Empty_individual:
            Position = None
            Fitness = None

        class Best_sol:
            Position = None
            Fitness = -(np.inf) # goal-maximizing function

        # Generate population
        BestSol = Best_sol()
        NewSol = Empty_individual()

        pop=[]
        for i in range(self.nPop):
            pop.append(Empty_individual())
        for i in range(self.nPop):
            pop[i].Position = np.random.uniform(self.varMin, self.varMax, self.varSize)
            
            # pop[i].Position represent weights vector
            pop[i].Position = pop[i].Position / np.sum(pop[i].Position)
            pop[i].Fitness = self._FitnessFunction(X_train, pop[i].Position, types, bounds, y_train)
            
            if (pop[i].Fitness >= BestSol.Fitness): # goal-maximizing function
                BestSol = copy.deepcopy(pop[i])

        return pop, BestSol, NewSol


    def _de_algorithm(self, pd_weights, X_train, y_train, types, bounds):
        cols = ['C' + str(y) for y in range(1, X_train.shape[1] + 1)]
        pd_weights = pd.DataFrame(index = cols)

        pop, BestSol, NewSol = self._generate_population(X_train, y_train, types, bounds)
        
        BestFitness = np.zeros(self.maxIt)
        MeanFitness = np.zeros(self.maxIt)
        # DE Main Loop
        for it in range(self.maxIt):
            mean_fitness_sum = 0
            for i in range(self.nPop):
                x = copy.deepcopy(pop[i].Position)
                v_pop = np.arange(self.nPop)
                v_pop = np.delete(v_pop, i)
                A = random.sample(list(v_pop), 3)
                
                a = A[0]
                b = A[1]
                c = A[2]
                
                # Mutation
                beta = np.random.uniform(self.beta_min, self.beta_max, self.varSize)
                # DE/rand/1 strategy
                # v = pop[a].Position+beta*(pop[b].Position-pop[c].Position)
                # DE/best/1/ strategy
                v = BestSol.Position+beta*(pop[a].Position-pop[b].Position)
                v[v < self.varMin] = self.varMin
                v[v > self.varMax] = self.varMax

                # Crossover
                u = np.zeros(self.varSize)
                aj = np.random.randint(0, self.varSize)
                for j in range(self.varSize):
                    if j == aj or np.random.rand() <= self.pCR:
                        u[j] = copy.deepcopy(v[j])
                    else:
                        u[j] = copy.deepcopy(x[j])
                
                NewSol.Position = copy.deepcopy(u)
                # NewSol.Position represents weights vector
                NewSol.Position = NewSol.Position / np.sum(NewSol.Position)
                NewSol.Fitness = self._FitnessFunction(X_train, NewSol.Position, types, bounds, y_train)
                mean_fitness_sum += NewSol.Fitness

                # Selection
                if NewSol.Fitness >= pop[i].Fitness: # goal-maximizing function
                    pop[i] = copy.deepcopy(NewSol)
                    
                    if pop[i].Fitness >= BestSol.Fitness: # goal-maximizing function
                        BestSol = copy.deepcopy(pop[i])

            # Update Best Fitness Individual
            BestFitness[it] = copy.deepcopy(BestSol.Fitness)
            BestPosition = copy.deepcopy(BestSol.Position)

            MeanFitness[it] = mean_fitness_sum / self.nPop
            
            # Show Iteration Information
            print('Iteration: ', it, ': Best Fitness = ', BestFitness[it])
            pd_weights[str(it)] = BestPosition

        return pd_weights, BestPosition, BestFitness, MeanFitness