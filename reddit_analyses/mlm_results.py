import numpy as np
from ast import literal_eval

temporal = [' never', 'sometimes', 'usually', 'generally', 'often', 'already', 'frequently','always']
factual = ['maybe',  'perhaps', 'possibly', 'probably', 'really', 'actually', 'certainly', 'definitely']
other = ['slightly','hardly', 'basically', 'quite', 'pretty', 'very',  'seriously', 'completely'] 



def create_accuracy_dict(df_new, temporal, factual, other):
    for c in ['bert_b', 'bert_b_neutral', 'bert_l', 'bert_l_neutral',
         'roberta', 'roberta_neutral', 'gpt2', 'gpt2_neutral', 'adv_exp']:
        if type(df_new[c].iloc[0]) == str:
            df_new[c] = df_new[c].apply(lambda x: literal_eval(x))

    correct = {}
    above_not = {}
    MRR = {}
    for c in ['bert_b', 'bert_b_neutral', 'bert_l', 'bert_l_neutral',
            'roberta', 'roberta_neutral', 'gpt2', 'gpt2_neutral']:
        print(c)
        correct[c] = {}
        for cat in ['temporal', 'factual', 'other']:
            correct[c][cat] = 0
            correct[c][f'{cat}_adv'] = {}
        above_not[c] = {}
        for cat in ['temporal', 'factual', 'other']:
            above_not[c][cat] = 0
            above_not[c][f'{cat}_adv'] = {}
            
        MRR[c] = {}
        for cat in ['temporal', 'factual', 'other']:
            MRR[c][cat] = 0
            MRR[c][f'{cat}_adv'] = {}

        for idx, i in df_new[c].iteritems():
            adv = df_new['adv_exp'].iloc[idx][0]
            rank_target = i[1][0]
            rank_not = i[0][0]
            if rank_target < rank_not: #! smaller means higher
                if adv in temporal:
                    above_not[c]['temporal']+=1
                    if adv not in above_not[c]['temporal_adv']:
                        above_not[c]['temporal_adv'][adv] =1
                    else:
                        above_not[c]['temporal_adv'][adv] +=1
                       
                if adv in factual:
                    above_not[c]['factual']+=1
                    if adv not in above_not[c]['factual_adv']:
                        above_not[c]['factual_adv'][adv] =1
                    else:
                        above_not[c]['factual_adv'][adv] +=1
                if adv in other:
                    above_not[c]['other']+=1
                    if adv not in above_not[c]['other_adv']:
                        above_not[c]['other_adv'][adv] =1
                    else:
                        above_not[c]['other_adv'][adv] +=1
                    
            if rank_target == 0:
                if adv in temporal:
                    correct[c]['temporal']+=1
                    if adv not in correct[c]['temporal_adv']:
                        correct[c]['temporal_adv'][adv] =1
                    else:
                        correct[c]['temporal_adv'][adv] +=1
                if adv in factual:
                    correct[c]['factual']+=1
                    if adv not in correct[c]['factual_adv']:
                        correct[c]['factual_adv'][adv] =1
                    else:
                        correct[c]['factual_adv'][adv] +=1
                if adv in other:
                    correct[c]['other']+=1
                    if adv not in correct[c]['other_adv']:
                        correct[c]['other_adv'][adv] =1
                    else:
                        correct[c]['other_adv'][adv] +=1
                    
            if adv in temporal:
                MRR[c]['temporal']+= 1/(1+rank_target)
                if adv not in MRR[c]['temporal_adv']:
                    MRR[c]['temporal_adv'][adv] =1/(1+rank_target)
                else:
                    MRR[c]['temporal_adv'][adv] +=1/(1+rank_target)
            if adv in factual:
                MRR[c]['factual']+= 1/(1+rank_target)
                if adv not in MRR[c]['factual_adv']:
                    MRR[c]['factual_adv'][adv] =1/(1+rank_target)
                else:
                    MRR[c]['factual_adv'][adv] +=1/(1+rank_target)
            if adv in other:
                MRR[c]['other']+=1/(1+rank_target)
                if adv not in MRR[c]['other_adv']:
                    MRR[c]['other_adv'][adv] =1/(1+rank_target)
                else:
                    MRR[c]['other_adv'][adv] +=1/(1+rank_target)
    return correct, above_not, MRR


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