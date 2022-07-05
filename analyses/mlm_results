import numpy as np

temporal = [' never', 'sometimes', 'usually', 'generally', 'often', 'already', 'frequently','always']
factual = ['maybe',  'perhaps', 'possibly', 'probably', 'really', 'actually', 'certainly', 'definitely']
other = ['slightly','hardly', 'basically', 'quite', 'pretty', 'very',  'seriously', 'completely'] 

def get_stats(accuracies_dict, MRR_dict):
    '''
    Function to print stats (need dictionary with acccuracies and MRR for each adverb)
    '''
    d = {}
    for v in accuracies_dict:
        d[v] ={'acc': 0, 'MRR': 0}
        d[v]['acc'] = accuracies_dict[v]/40
    for v in MRR_dict:
        d[v]['MRR'] = MRR_dict[v]/40
        
    print('OTHER')
    other_sorted = sorted({k:v for k, v in d.items() if k in other}.items(), key = lambda k:k[1]['acc'], reverse=True)
    print(other_sorted)
    
    print('FACTUAL')
    factual_sorted = sorted({k:v for k, v in d.items() if k in factual}.items(), key = lambda k:k[1]['acc'], reverse=True)
    print(factual_sorted)
    
    print('TEMPORAL')
    temporal_sorted = sorted({k:v for k, v in d.items() if k in temporal}.items(), key = lambda k:k[1]['acc'], reverse=True)
    print(temporal_sorted)
    
    print('acc_other', np.mean([i[1]['acc'] for i in other_sorted]))
    print('MRR_other', np.mean([i[1]['MRR'] for i in other_sorted]))

    print('acc_epistemic', np.mean([i[1]['acc'] for i in factual_sorted]))
    print('MRR_epistemic', np.mean([i[1]['MRR'] for i in factual_sorted]))

    print('acc_temporal', np.mean([i[1]['acc'] for i in temporal_sorted]))
    print('MRR_temporal', np.mean([i[1]['MRR'] for i in temporal_sorted]))