import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns

from rank_preferences import *
from correlations import *
from weighting_methods import *
from spotis import SPOTIS
from de import DE_algorithm
from visualization import *

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr


def main():

    # load dataset
    
    filename = 'input/mobile_phones2000.csv'

    data = pd.read_csv(filename)
    types = data.iloc[len(data) - 1, :].to_numpy()
    df_data = data.iloc[:200, :]
    matrix = df_data.to_numpy()

    # determine bounds of alternatives performances for SPOTIS
    bounds_min = np.amin(matrix, axis = 0)
    bounds_max = np.amax(matrix, axis = 0)
    bounds = np.vstack((bounds_min, bounds_max))

    # split dataset on training and test dataset
    X_train_df, X_test_df = train_test_split(df_data, test_size=0.2, random_state=5)
    X_train_df = copy.deepcopy(X_train_df)
    X_test_df = copy.deepcopy(X_test_df)

    # new alternatives symbols
    list_alt_names_train = [r'$A_{' + str(i) + '}$' for i in range(1, len(X_train_df) + 1)]
    list_alt_names_test = [r'$A_{' + str(i) + '}$' for i in range(1, len(X_test_df) + 1)]

    X_train_df['Ai'] = list_alt_names_train
    X_test_df['Ai'] = list_alt_names_test

    X_train_df = X_train_df.set_index('Ai')
    X_test_df = X_test_df.set_index('Ai')

    X_train = X_train_df.to_numpy()
    X_test = X_test_df.to_numpy()

    # real weights
    train_weights = entropy_weighting(X_train)
    spotis = SPOTIS()
    pref_train = spotis(X_train, train_weights, types, bounds)
    y_train = rank_preferences(pref_train, reverse = False)

    pref_test = spotis(X_test, train_weights, types, bounds)
    y_test = rank_preferences(pref_test, reverse = False)

    cols = [r'$C_{' + str(y) + '}$' for y in range(1, X_train.shape[1] + 1)]
    pd_weights = pd.DataFrame(index = cols)

    # DE algorithm
    de_algorithm = DE_algorithm()
    pd_weights, BestPosition, BestFitness, MeanFitness = de_algorithm(pd_weights, X_train, y_train, types, bounds)

    # Results
    weights = pd.DataFrame(index = cols)
    weights['Entropy weights'] = train_weights
    weights['DE weights'] = BestPosition
    weights = weights.rename_axis('Cj')
    weights.to_csv('output/best_weights_de.csv')

    print('Weights correlation: ', pearson_coeff(train_weights, BestPosition))
    s, _ = pearsonr(train_weights, BestPosition)
    print('Check: ', s)

    spotis = SPOTIS()
    pref = spotis(X_test, BestPosition, types, bounds)
    rank = rank_preferences(pref, reverse = False)
    print('Consistency: ', spearman(rank, y_test))
    s, _ = spearmanr(rank, y_test)
    print('Check: ', s)

    results = pd.DataFrame(index = X_test_df.index)
    results['Real rank'] = y_test
    results['DE rank'] = rank
    results.to_csv('output/results_de.csv')

    fitness_best = pd.DataFrame()
    fitness_best['Best fitness value'] = BestFitness
    fitness_best.to_csv('output/best_fitness.csv')

    fitness_mean = pd.DataFrame()
    fitness_mean['Mean fitness value'] = MeanFitness
    fitness_mean.to_csv('output/mean_fitness.csv')

    # Results visualization
    plot_fitness(BestFitness, MeanFitness)
    plot_rankings(results)
    plot_weights(weights)
    

if __name__ == '__main__':
    main()