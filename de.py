import numpy as np
import pandas as pd
import sys
import copy
import random

from rank_preferences import *
from correlations import *
from weighting_methods import *
from spotis import SPOTIS

from visualization import *


class DE_algorithm():
    def __init__(self,
    varMin = sys.float_info.epsilon,
    varMax = 1.0,
    maxIt = 200,
    nPop = 60,
    beta_min = 0.2,
    beta_max = 0.8,
    pCR = 0.4):

        """
        Create DE object with initialization of setting parametres of DE

        Parameters
        ----------
            varMin : float
                Lower bound of weights values
            varMax : float
                Upper bound of weights values
            maxIt : int
                Maximum number of iterations
            nPop : int
                Number of individuals in population
            beta_min : float
                Lower bound of range for random F parameter for mutation
            beta_max : float
                Upper bound of range for random F parameter for mutation
            pCR : float
                Crossover probability
        """
        self.varMin = varMin
        self.varMax = varMax
        self.maxIt = maxIt
        self.nPop = nPop
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pCR = pCR


    def __call__(self, X_train, y_train, types, bounds, verbose = True):
        """
        Determine criteria weights using DE algorithm with the goal (fitness) function using 
        SPOTIS method and Spearman rank coefficient

        Parameters
        ----------
            X_train : ndarray
                Decision matrix containing training dataset of alternatives and their performances corresponding to the criteria
            y_train: ndarray
                Ranking of training decision matrix which is the targer variable
            types : ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.
            bounds : ndarray
                Bounds contain minimum and maximum values of each criterion. Minimum and maximum cannot be the same.

        Returns
        -------
            ndarray
                Values of best solution representing criteria weights
            ndarray
                Best values of fitness function in each iteration required for visualization of fitness function.
            ndarray
                Mean values of fitness function in each iteration required for visualization of fitness function.
        """
        self.varSize = np.shape(X_train)[1]
        self.verbose = verbose
        return DE_algorithm._de_algorithm(self, X_train, y_train, types, bounds)


    def _fitnessFunction(self, matrix, weights, types, bounds, rank_ref):
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

        pop = [Empty_individual() for i in range(self.nPop)]
        for i in range(self.nPop):
            pop[i].Position = np.random.uniform(self.varMin, self.varMax, self.varSize)
            
            # pop[i].Position represent weights vector
            pop[i].Position = pop[i].Position / np.sum(pop[i].Position)
            pop[i].Fitness = self._fitnessFunction(X_train, pop[i].Position, types, bounds, y_train)
            
            if (pop[i].Fitness >= BestSol.Fitness): # goal-maximizing function
                BestSol = copy.deepcopy(pop[i])

        return pop, BestSol, NewSol

    def _crossover(self, u, v, aj):
        u[aj] = v[aj]
        R = np.random.rand(len(u))
        u[R <= self.pCR] = v[R <= self.pCR]
        return u


    @staticmethod
    def _de_algorithm(self, X_train, y_train, types, bounds):

        # Generate population with individuals
        pop, BestSol, NewSol = self._generate_population(X_train, y_train, types, bounds)
        
        BestFitness = np.zeros(self.maxIt)
        MeanFitness = np.zeros(self.maxIt)
        # DE Main Loop
        for it in range(self.maxIt):
            mean_fitness_sum = 0
            for i in range(self.nPop):
                x = copy.deepcopy(pop[i].Position)

                # Mutation
                v_pop = np.arange(self.nPop)
                v_pop = np.delete(v_pop, i)
                A = random.sample(list(v_pop), 3)
                
                beta = np.random.uniform(self.beta_min, self.beta_max, self.varSize)
                # DE/rand/1 strategy
                # v = pop[A[0]].Position+beta*(pop[A[1]].Position-pop[A[2]].Position)
                # DE/best/1/ strategy
                v = BestSol.Position+beta*(pop[A[0]].Position-pop[A[1]].Position)
                v[v < self.varMin] = self.varMin
                v[v > self.varMax] = self.varMax

                # Crossover
                u = copy.deepcopy(x)
                aj = np.random.randint(0, self.varSize)
                u = self._crossover(u, v, aj)

                NewSol.Position = copy.deepcopy(u)
                # NewSol.Position represents weights vector
                NewSol.Position = NewSol.Position / np.sum(NewSol.Position)
                NewSol.Fitness = self._fitnessFunction(X_train, NewSol.Position, types, bounds, y_train)
                mean_fitness_sum += NewSol.Fitness

                # Selection
                if NewSol.Fitness >= pop[i].Fitness: # goal-maximizing function
                    pop[i] = copy.deepcopy(NewSol)
                    
                    if pop[i].Fitness >= BestSol.Fitness: # goal-maximizing function
                        BestSol = copy.deepcopy(pop[i])

            # Update Best Fitness Individual
            BestPosition = copy.deepcopy(BestSol.Position)

            # Save the best and mean fitness value for iteration for visualization
            BestFitness[it] = copy.deepcopy(BestSol.Fitness)
            MeanFitness[it] = mean_fitness_sum / self.nPop
            
            # Show Iteration Information
            if self.verbose:
                print('Iteration: ', it, ': Best Fitness = ', BestFitness[it])

        return BestPosition, BestFitness, MeanFitness