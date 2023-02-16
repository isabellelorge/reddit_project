
from scipy import stats, spatial
import numpy as np


def get_sim_dict(deg_adv, temporal, factual, other, embed_dict,  ref_vector=str, vector=None):
    '''
    Function to create dictionary of adverb cosine similarities with reference vector
    '''
    top_temp = temporal[-1]
    top_fact = factual[-1]
    top_oth = other[-1]

    bottom_temp = temporal[1]
    bottom_fact = factual[0]
    bottom_oth = other[1]


    sim_dict = {}
    for adv in deg_adv:
        if adv not in embed_dict:
            adv = adv.strip()
        if ref_vector == 'adj':
            sim_dict[adv] = np.mean([1 - spatial.distance.cosine(embed_dict[adv][i], vector) for i in range(len(embed_dict[adv]))])

        if ref_vector == 'diff':
            if adv in temporal:
                sim_dict[adv] = np.mean([1 - spatial.distance.cosine(embed_dict[adv][i], embed_dict[top_temp][i]-embed_dict[bottom_temp][i]) for i in range(len(embed_dict[adv]))])

            if adv in factual:
                sim_dict[adv] = np.mean([1 - spatial.distance.cosine(embed_dict[adv][i], embed_dict[top_fact][i]-embed_dict[bottom_fact][i]) for i in range(len(embed_dict[adv]))])

            if adv in other:
                sim_dict[adv] = np.mean([1 - spatial.distance.cosine(embed_dict[adv][i], embed_dict[top_oth][i]-embed_dict[bottom_oth][i]) for i in range(len(embed_dict[adv]))])
        if ref_vector == 'top':
            if adv in temporal:
                sim_dict[adv] = np.mean([1 - spatial.distance.cosine(embed_dict[adv][i], embed_dict[top_temp][i]) for i in range(len(embed_dict[adv]))])

            if adv in factual:
                sim_dict[adv] = np.mean([1 - spatial.distance.cosine(embed_dict[adv][i], embed_dict[top_fact][i]) for i in range(len(embed_dict[adv]))])

            if adv in other:
                sim_dict[adv] = np.mean([1 - spatial.distance.cosine(embed_dict[adv][i], embed_dict[top_oth][i]) for i in range(len(embed_dict[adv]))])
        
    return sim_dict


def get_correlations(sim_dict, category):
    '''
    Function to get correlations
    '''
    c = sorted({k:v for k, v in sim_dict.items() if k in category[:-1]}.items(), key = lambda k:k[1])
    print(c)
    print(stats.spearmanr(list(range(7)),[category[:-1].index(k[0]) for k in c]))
    print(stats.kendalltau(list(range(7)),[category[:-1].index(k[0]) for k in c]))
    
    
def pairwise_acc(temporal, factual, other, sim_dict):
    '''
    Function to get pairwise accuracies of intensity rankings from
    dictionary of cosine similarities with reference vector
    '''
    temporal_ties = [(3, 4), (4, 3), (5, 6), (6, 5)]
    factual_ties = [(0,1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0),
               (1,2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)]
    
    all_pairs = 0
    correct = 0
    for i in ['always', 'definitely', 'completely']:
        if i in sim_dict:
            sim_dict.pop(i)
    for i in sim_dict:
        for j in sim_dict:
            same_sc = False
            if i in temporal and j in temporal:
                same_sc = True
                i_idx = temporal.index(i)
                j_idx = temporal.index(j)
                if(i_idx, j_idx) not in temporal_ties:
                    all_pairs +=1
                else:
                    continue
            elif i in factual and j in factual:
                same_sc = True
                i_idx = factual.index(i)
                j_idx = factual.index(j)
                if (i_idx, j_idx) not in factual_ties:
                    all_pairs +=1
                else:
                    continue
            elif i in other and j in other:
                same_sc = True
                all_pairs +=1
                i_idx = other.index(i)
                j_idx = other.index(j)

            if same_sc == True:
                if (i_idx > j_idx) and (sim_dict[i] > sim_dict[j]):
                    correct +=1
                if (i_idx < j_idx) and (sim_dict[i] < sim_dict[j]):
                    correct+=1
    return correct/all_pairs
