'''
Script to decompress commments file, can also be used to loop through comments and do something with them
'''

# import argparse
import re
import json
import pandas as pd

import spacy

nlp = spacy.load("en_core_web_sm", disable = ['ner', 'parser', 'textcat'])

year = 2015
files = ['01', '02', '03', '04', '05', '06', '07', '09', '10', '11', '12']
source_dir = '/Users/isabellelorge/Desktop/reddit_data/zipped_files'
target_dir = '/Users/isabellelorge/Desktop/reddit_data/deg_adv'
final_df_path = ''

temporal = [' never', 'sometimes', 'usually', 'generally', 'often', 'already', 'frequently','always']
factual = ['maybe',  'perhaps', 'possibly', 'probably', 'really', 'actually', 'certainly', 'definitely']
other = ['slightly','hardly', 'basically', 'quite', 'pretty', 'very',  'seriously', 'completely'] 
deg_adv = temporal + factual + other



for f in files:
    comments_file = f'{source_dir}/comments_{year}-{f}.bz2'
    print(comments_file)

    # def main():

    #     # Read arguments
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--comments_file', default=None, type=str, required=True, help='Comments file')
    #     parser.add_argument('--subreddits_file', default=None, type=str, required=False, help='Subreddits file')
    #     parser.add_argument('--target_dir', default=None, type=str, required=False, help='Directory to store comments')
    #     args = parser.parse_args()

    # Read subreddits
    # if subreddits_file:
    #     with open(subreddits_file, 'r') as f:
    #         subreddits = set(f.read().strip().split('\n'))

    if target_dir:
        target_file = '{}/comments_extracted_{}.json'.format(
            target_dir, re.findall(r'\d{4}-\d{2}', comments_file)[0]
        )
        print(target_file)

    # Load comments in chunks, filter for adverbs through string search
    comments = list()
    for c in pd.read_json(comments_file, compression='bz2', lines=True, dtype=False, chunksize=10000):
    #     if subreddits_file:
    #         c = c[c.subreddit.isin(subreddits)]
    
#         c = c[c['body'].str.contains(' \w+\.|'.join(deg_adv))]
        c = c[c['body'].str.contains(f'{deg_adv[0]} \w+\.')]
        comments.append(c)
    
    # concat
    comments = pd.concat(comments, sort=True)
    print(len(comments))
    
    # for each adv, get df containing adv+word+.
    comments_adv = list()
    for adv in deg_adv:
        adv_df = comments[comments['body'].str.contains(f'{adv} \w+\.')]
        
        # get the positions of adv in each comment
        positions = [i for i in enumerate(adv_df['body'].str.find(adv)) if i[1] != -1]
        filtered_positions = []
        
        # for each position, get adverbial expression and check that second word POS is ADJ
        for p in positions:
            try:
                adv_exp = re.findall(f'{adv} \w+\.', adv_df['body'].iloc[p[0]][p[1]:])[0]
                doc_exp = nlp(adv_exp)
                if doc_exp[-2].pos_ == 'ADJ':
#                     print('EXPRESSION:', adv_exp)
#                     print('COMMENT:', adv_df['body'].iloc[p[0]])
                    filtered_positions.append(p[0])
            except Exception as e:
                continue
                
        adv_df_filt = adv_df.iloc[filtered_positions]
        comments_adv.append(adv_df_filt)
    
    # concat
    comments_adv = pd.concat(comments_adv, sort=True)
    print(len(comments_adv))
    
    # Store extracted comments
    comments_adv.to_json(
            target_file,
            orient='records',
            lines=True
        )

    # if __name__ == '__main__':
    # main()

dfs = []
files = ['01', '02', '03', '04', '05', '06', '07', '09', '10', '11', '12']

for f in files:
    print(f)
    comments_file = f'/Users/isabellelorge/Desktop/reddit_data/deg_adv/comments_extracted_{year}-{f}.json'

    comments = []

    with open(comments_file) as f:
        for line in f:
            comment = json.loads(line)
            comments.append(comment)

    df = pd.DataFrame(comments)
    dfs.append(df)

final_df = pd.concat(dfs, sort=True)
final_df = final_df.drop_duplicates(subset = ['body'])
final_df = final_df.reset_index(drop=True)
final_df.to_csv(final_df_path)