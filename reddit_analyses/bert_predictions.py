import numpy as np
import tensorflow as tf
from transformers import RobertaModel, BertModel, BertTokenizer, TFBertForMaskedLM, RobertaTokenizer, TFRobertaForMaskedLM, GPT2Tokenizer, TFGPT2LMHeadModel

tokenizer_large = BertTokenizer.from_pretrained('bert-large-cased')
model_large = TFBertForMaskedLM.from_pretrained('bert-large-cased')
model_large_embed = BertModel.from_pretrained('bert-large-cased')
tokenizer_base = BertTokenizer.from_pretrained('bert-base-cased')
model_base = TFBertForMaskedLM.from_pretrained('bert-base-cased')
model_base_embed = BertModel.from_pretrained('bert-base-cased')
tokenizer_berta = RobertaTokenizer.from_pretrained('roberta-large')
model_berta = TFRobertaForMaskedLM.from_pretrained('roberta-large')
model_berta_embed = RobertaModel.from_pretrained('roberta-large')
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = TFGPT2LMHeadModel.from_pretrained("gpt2")

def get_bert_token_embedding(sentence, word, model_name):
    if model_name == 'bert_large':
        word = [word.strip()]
        tokenizer = tokenizer_large
        model = model_large_embed
    if model_name == 'bert_base':
        word = [word.strip()]
        tokenizer = tokenizer_base
        model = model_base_embed
    if model_name == 'roberta':
        word = ' ' + word
        tokenizer = tokenizer_berta
        model = model_berta_embed

    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    word_token = tokenizer.encode(word)[1] 
    print(word_token)
    print(tokenizer.decode(word_token))
    word_index =  np.where(inputs['input_ids'].numpy()[0] == word_token)[0][0]
    word_embed = last_hidden_states.detach().numpy()[0][word_index]
    return word_embed

def get_top_k_predictions(input_string, target,  model_name, k=10):
    '''
    Function to get bert ranking and pred for not, for target and top k predictions
    '''
    
    if model_name == 'bert_large':
        tokenizer = tokenizer_large
        model = model_large
        mask_token = tokenizer.encode('[MASK]')[-2]
        adv_token = tokenizer.encode(target)[1] 
        neg_token = tokenizer.encode(['not'])[1] 
    if model_name == 'bert_base':
        tokenizer = tokenizer_base
        model = model_base
        mask_token = tokenizer.encode('[MASK]')[-2]
        adv_token = tokenizer.encode(target)[1] 
        neg_token = tokenizer.encode(['not'])[1] 
    if model_name =='roberta':
        tokenizer = tokenizer_berta
        model = model_berta
        mask_token = tokenizer.mask_token_id
        adv_token = tokenizer.encode(' ' + target)[1]
        neg_token = tokenizer.encode(' not')[1]
    if model_name == 'gpt2':
        tokenizer = tokenizer_gpt2
        model = model_gpt2
        adv_token = tokenizer.encode(' ' + target)[0]
        neg_token = tokenizer.encode(' not')[0]
        
    tokenized_inputs = tokenizer(input_string, return_tensors="tf")
    outputs = model(tokenized_inputs["input_ids"], return_dict = True)

    # get the index of adv in sentence
    if model_name == 'gpt2':
        adv_index = np.where(tokenized_inputs['input_ids'].numpy()[0] == adv_token)[0][0] - 1
    else:
        adv_index = np.where(tokenized_inputs['input_ids'].numpy()[0] == mask_token)[0][0]

    # get the top k logits for adv position in sentence
    top_k_indices = tf.math.top_k(outputs.logits, k).indices[0].numpy()[adv_index]

    # get softmax over all voc for adv position in sentence
    all_prob = tf.nn.softmax(outputs.logits[0][adv_index]).numpy()
    sorted_prob = tf.argsort(all_prob, direction='DESCENDING').numpy()

    # get top prob
    top_k_prob = all_prob [top_k_indices]

    # get words for top prob
    if model_name == 'roberta':
        decoded_top = [tokenizer.decode(i) for i in top_k_indices]
    else:
        decoded_top = tokenizer.decode(top_k_indices).split()

    # get prob for adv 
    adv_prob =  all_prob[adv_token]
    adv_rank = np.where(sorted_prob == adv_token)[0][0]

    # get prob for not 
    neg_prob = all_prob[neg_token]
    neg_rank = np.where(sorted_prob == neg_token)[0][0]
    
#     if model_name == 'gpt2':
#         adv_prob = adv_prob[0]
#         neg_prob = neg_prob[0]
    
    return (neg_rank, neg_prob), (adv_rank, adv_prob), list(zip(decoded_top, top_k_prob))

def get_negation_performance(df_sent, predictions_column):
    '''
    Get number of top 1 and top 10 'not' predictions,
    as well as number of times 'not' ranked above target
    '''
    top_1 = 0
    top_10 = 0
    above_original = 0

    for i, p in df_sent[predictions_column].iteritems():
        adv = df_sent['sentences'].iloc[i].split()[-2]
        if p:
            print(p[1])
            print(p[2])
            if p[2][0][0] == 'not':
                top_1 +=1
            if any(i[0] == 'not' for i in p[2]):
                top_10 +=1
            if adv.strip() not in ['never', 'hardly']:
                adv_idx = p[1][0]
                not_idx = p[0][0]
                if not_idx < adv_idx: # smaller idx means not is ranked above
                    above_original +=1
    
    return top_1, top_10, above_original