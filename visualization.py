import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_scatter(data, model_compare):
    sns.set_style("darkgrid")
    list_rank = np.arange(1, len(data) + 2, 4)
    list_alt_names = data.index
    for it, el in enumerate(model_compare):
        
        xx = [min(data[el[0]]), max(data[el[0]])]
        yy = [min(data[el[1]]), max(data[el[1]])]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(xx, yy, linestyle = '--', zorder = 1)

        ax.scatter(data[el[0]], data[el[1]], marker = 'o', color = 'royalblue', zorder = 2)
        for i, txt in enumerate(list_alt_names):
            ax.annotate(txt, (data[el[0]][i], data[el[1]][i]), fontsize = 14, style='italic',
                         verticalalignment='bottom', horizontalalignment='right')

        ax.set_xlabel(el[0], fontsize = 12)
        ax.set_ylabel(el[1], fontsize = 12)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xticks(list_rank)
        ax.set_yticks(list_rank)

        x_ticks = ax.xaxis.get_major_ticks()
        y_ticks = ax.yaxis.get_major_ticks()

        ax.set_xlim(-1, len(data) + 1)
        ax.set_ylim(0, len(data) + 1)

        ax.grid(True, linestyle = '--')
        ax.set_axisbelow(True)
    
        plt.tight_layout()
        plt.savefig('output/scatter_' + el[0] + '.pdf')
        plt.show()


def plot_fitness(BestFitness, MeanFitness):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    ax.plot(BestFitness, label = 'Best fitness value')
    ax.plot(MeanFitness, label = 'Mean fitness value')
    ax.set_xlabel('Iterations', fontsize = 12)
    ax.set_ylabel('Fitness value', fontsize = 12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(fontsize = 12)
    plt.tight_layout()
    plt.savefig('output/fitness.pdf')
    plt.show()


def plot_rankings(results):
    model_compare = []
    names = list(results.columns)
    model_compare = [[names[0], names[1]]]
    results = results.sort_values('Real rank')
    sns.set_style("darkgrid")
    plot_scatter(data = results, model_compare = model_compare)


def plot_weights(weights):
    sns.set_style("darkgrid")
    step = 1
    list_rank = np.arange(1, len(weights) + 1, step)
    
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.scatter(x = list_rank, y = weights['Real weights'].to_numpy(), label = 'Real weights')
    ax.scatter(x = list_rank, y = weights['DE weights'].to_numpy(), label = 'DE weights')
    
    ax.set_xlabel('Criteria', fontsize = 12)
    ax.set_ylabel('Weight value', fontsize = 12)
    ax.set_xticks(list_rank)

    ax.set_xticklabels(list(weights.index))
    ax.tick_params(axis = 'both', labelsize = 12)
    y_ticks = ax.yaxis.get_major_ticks()
    plt.legend()
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('output/weights_comparison.pdf')
    plt.show()