from ast import literal_eval
import pandas as pd 
import re
import requests
from nltk.corpus import wordnet as wn

import spacy

nlp = spacy.load("en_core_web_sm", disable = ['ner', 'parser', 'textcat'])


# corpora = dict(eng_us_2012=17, eng_us_2009=5, eng_us_2019=28,
#                eng_gb_2012=18, eng_gb_2009=6, eng_gb_2019=26,
#                chi_sim_2019=34, chi_sim_2012=23, chi_sim_2009=11,
#                eng_2012=15, eng_2009=0,
#                eng_fiction_2012=16, eng_fiction_2009=4, eng_1m_2009=1,
#                fre_2019=30, fre_2012=19, fre_2009=7,
#                ger_2019=31, ger_2012=20, ger_2009=8,
#                heb_2012=24,
#                heb_2009=9,
#                spa_2019=32, spa_2012=21, spa_2009=10,
#                rus_2019=36, rus_2012=25, rus_2009=12,
#                ita_2019=33, ita_2012=22)


def extract_unstacked_phrases_and_context(deg_adv, df):
    '''
    Function which takes a list of adverbs and a dataframe 
    and returns a dataframe with non-stacked expressions and their
    context (previous and current sentence)
    '''
    adv_exps = []
    adv_sentences = []
    phrases_df = pd.DataFrame()

    # for each adverb
    for adv in deg_adv:
        print(adv)

        # get subset df containing adverb
        adv_df = df[df['body'].str.contains(f'{adv} \w+\.', na=False)]

        # get the positions of adv in each comment
        positions = [i for i in enumerate(adv_df['body'].str.find(adv)) if i[1] != -1]

        # for each position, get adverbial expression and check that second word POS is ADJ
        for p in positions:
            adv_exp = re.findall(f'{adv} \w+\.', adv_df['body'].iloc[p[0]][p[1]:])[0]
            adj = re.sub('\.', '', adv_exp.split()[1])
            sentences = adv_df['body'].iloc[p[0]].split('.')
            l = [(i, s) for (i, s) in enumerate(sentences) if f'{adv} {adj}' in s]
            idx = l[0][0]
            sent = l[0][1]
            counter = idx
            prev_sent = ''
            while counter > 1:
                if sentences[counter-1] != '':
                    prev_sent = sentences[counter-1]
                    break
                else:
                    counter -=1
                
            context = re.sub(';|\n|&gt', '', prev_sent + '. ' + sent + '.')
            
            # check second element is adj and no stacked adv
            doc_exp = nlp(sent)

            if len(doc_exp) > 2:
                if doc_exp[-3].pos_ != 'ADV' and doc_exp[-1].pos_ == 'ADJ':
                        adv_exps.append((adv, adj))
                        adv_sentences.append(context)
            else:
                if doc_exp[-1].pos_ == 'ADJ':
                    adv_exps.append((adv, adj))
                    adv_sentences.append(context)
        phrases_df['adv_exp'] = adv_exps
        phrases_df['sentences'] = adv_sentences

        return phrases_df


def get_ngrams(query, corpus='eng_us_2019', startYear=2000, endYear=2019, smoothing=None, caseInsensitive=False,
                filename=None
                ):
    '''
    Function to get ngram frequencies
    returns DataFrame with frequency for each year and each query
    (need to be separated by commas in the string)
    '''
    params = dict(content=query, year_start=startYear, year_end=endYear,
                corpus=corpora[corpus], smoothing=smoothing,
                case_insensitive=caseInsensitive)
    if params['case_insensitive'] is False:
        params.pop('case_insensitive')
    if '?' in params['content']:
        params['content'] = params['content'].replace('?', '*')
    if '@' in params['content']:
        params['content'] = params['content'].replace('@', '=>')

    req = requests.get('http://books.google.com/ngrams/graph', params=params)
    res = re.findall('ngrams.data = .*\];', req.text)
    if not len(res)==1: print(req.text)
    assert(len(res)==1)

    if res:
        dataDict = literal_eval(res[0].replace(
            "ngrams.data = ", "").replace(";", ""))
        data = {qry['ngram']: qry['timeseries']
                for qry in dataDict}
        df = pd.DataFrame(data)
        df.insert(0, 'year', list(range(startYear, endYear + 1)))
    else:
        df = pd.DataFrame()
        
    return df


def create_sample_dataset(original_df, deg_adv, sample_size_per_token = 40, context_length = (10, 40)):
    '''
    Function to create a sample dataset of desired length from full dataset 
    '''
    sample_df = pd.DataFrame()

    for adv in deg_adv:
        print(adv)

        # drop duplicates
        df_adv = original_df[original_df['adv_exp'].str.contains(f"'{adv}'")]
        df_no_dup = df_adv.drop_duplicates(subset = 'adv_exp')

        # remove 0 frequencies
        no_zero = df_no_dup[df_no_dup['frequencies']!= 0]

        # restrict to context of desired length
        no_zero = no_zero[no_zero['sentences'].apply(lambda x: len(x.split()) >= context_length[0] and len(x.split()) <= context_length[1])]
        no_zero = no_zero.reset_index()

        # select most and least frequent expressions 
        max_idx = no_zero['frequencies'].idxmax()
        min_idx =  no_zero['frequencies'].idxmin()
        clean = no_zero.drop([max_idx, min_idx])

        # select other expressions at random
        random = clean.sample(n=sample_size_per_token-2, random_state=1)

        sample_df = sample_df.append(no_zero.iloc[min_idx])
        sample_df = sample_df.append(no_zero.iloc[max_idx])
        sample_df = sample_df.append(random)
    
    return sample_df


def create_contradiction_examples(original_sentences):
    '''
   Function to create synthetic sentences which involve a contradiction
    e.g., 'It is not alkaline. In fact, it is very acidic'. 
    using WordNet antonyms and synonyms 
    '''
    synth = []


    for s in original_sentences:
        adj = re.sub('\.', '', s.split()[-1])
        adv = s.split()[-2]
        syno = False
        anto = ''

        # for each synset of adj
        for ss in wn.synsets(adj):

            # if POS is ADJ
            if f'.a.' in ss.name():
                for l in ss.lemmas():

                    # different structure so it's felicitous
                    if adv == 'sometimes':
                        if syno == False and l.name() not in adj :
                            synth.append(f'I think it is {l.name()}. In fact, it is {adv} {adj}.')
                            syno = True
                        if anto == '' and l.antonyms():
                            anto = l.antonyms()[0].name()
                    else:
                        # try to find synomym
                        if syno == False and l.name() not in adj :
                            synth.append(f'It is {l.name()}. In fact, it is {adv} {adj}.')
                            syno = True
                        # otherwise use antonym
                        if anto == '' and l.antonyms():
                            anto = l.antonyms()[0].name()
                    
        # only use antonym if synonym not found
        if syno == False and anto != '':
            if adv == 'sometimes':
                synth.append(f'I do not think it is {anto}. In fact, it is {adv} {adj}.')
            else:
                synth.append(f'It is not {anto}. In fact, it is {adv} {adj}.')
        if syno == False and anto == '':
            synth.append('')

    return synth