import numpy as np
import pandas as pd
import copy

from rank_preferences import *
from correlations import *
from weighting_methods import *
from spotis import SPOTIS
from de import DE_algorithm
from visualization import *


def main():
    # load dataset
    filename = 'input/mobile_phones2000.csv'
    data = pd.read_csv(filename)
    types = data.iloc[len(data) - 1, :].to_numpy()
    df_data = data.iloc[:200, :]
    whole_matrix = df_data.to_numpy()

    # determine bounds of alternatives performances for SPOTIS
    bounds_min = np.amin(whole_matrix, axis = 0)
    bounds_max = np.amax(whole_matrix, axis = 0)
    bounds = np.vstack((bounds_min, bounds_max))

    # load train and test datasets
    train_df = pd.read_csv('input/train.csv', index_col = 'Ai')
    X_train = train_df.iloc[:len(train_df) - 1, :-1].to_numpy()
    y_train = train_df.iloc[:len(train_df) - 1, -1].to_numpy()

    test_df = pd.read_csv('input/test.csv', index_col = 'Ai')
    X_test = test_df.iloc[:len(test_df) - 1, :-1].to_numpy()
    y_test = test_df.iloc[:len(test_df) - 1, -1].to_numpy()

    # real weights
    train_weights = entropy_weighting(X_train)
    cols = [r'$C_{' + str(y) + '}$' for y in range(1, data.shape[1] + 1)]
    pd_weights = pd.DataFrame(index = cols)

    # DE algorithm
    de_algorithm = DE_algorithm()
    pd_weights, BestPosition, BestFitness, MeanFitness = de_algorithm(pd_weights, X_train, y_train, types, bounds)

    # Results
    # Weights
    weights = pd.DataFrame(index = cols)
    weights['Real weights'] = train_weights
    weights['DE weights'] = BestPosition
    weights = weights.rename_axis('Cj')
    weights.to_csv('output/best_weights_de.csv')

    print('Weights correlation: ', pearson_coeff(train_weights, BestPosition))
    plot_weights(weights)

    # Fitness
    fitness_best = pd.DataFrame()
    fitness_best['Best fitness value'] = BestFitness
    fitness_best.to_csv('output/best_fitness.csv')

    fitness_mean = pd.DataFrame()
    fitness_mean['Mean fitness value'] = MeanFitness
    fitness_mean.to_csv('output/mean_fitness.csv')
    plot_fitness(BestFitness, MeanFitness)

    # Ranking
    spotis = SPOTIS()
    pref = spotis(X_test, BestPosition, types, bounds)
    y_pred = rank_preferences(pref, reverse = False)
    print('Rankings consistency: ', spearman(y_test, y_pred))

    results = pd.DataFrame(index = test_df.index[:-1])
    results['Real rank'] = y_test
    results['Predicted rank'] = y_pred
    results.to_csv('output/results_de.csv')
    
    plot_rankings(results)
    
    

if __name__ == '__main__':
    main()