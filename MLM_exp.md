# Testing language models' ability to predict intensifiers

## Rationale 
Large language models trained to produce representations based purely on a distributional assumption (i.e., that words with similar contexts have similar meanings) tend to struggle with producing meaningful representations of words which have a high contextual overlap (e.g., antonyms or negation). We want to test the ability of these models to predict the presence of an intensifier in an original sentence (versus a negation). 

We wish to investigate:
1.  Is the language model is able to overcome frequency effects by using context to predict the correct adverb
2.  Does the prediction accuracy correlate with the intensity of the original adverb (i.e., adverbs with higher intensity whose semantics are further from negation are more accurately predicted)

3.  Does the accuracy differ depending on the type of intensifier, i.e., intensifiers with neutral semantics versus those which integrate a temporal or epistemic dimension. If so which ones are easier or harder?

## Method
### Data
We select 24 intensifier adverbs among the most frequent in the dataset: 8 intensifiers including a temporal dimension (never, sometimes, already, often, generally, usually, frequently, always) 8 intensifiers including an epistemic dimension (maybe, perhaps, possibly, probably, actually, really, certainly, definitely) and 8 neutral intensifiers (hardly, slightly, basically, quite, pretty, very, seriously, completely) spanning a range of intensities along their respective dimensions. 

We use data from the Reddit politosphere (Hofmann et al., 2021) dataset. We collect all data from 2015 (about 6GB uncompressed) and use Spacy dependencies to filter for phrases with syntactic form 'ADV+ADJ.' Filtering for sentences which end with the target expression allows us to control its position for human comparison (controlling context seen when they get to the mask) and potentially use autoregressive left-to-right language models for prediction. 

We collect all adjective types and tokens for each adverb. The diversity ranges from 40 different adjective combinations (for 'frequently') to over 3k for the most frequent adverbs ('very' and 'really'). We examine the bigram frequencies using the Google Ngram API and find them to be fairly normally distributed when transformed into base e logarithms, mostly ranging between -22 and -10 with a mean and median around -16. 

Taking our least frequent adverb as an upper bound, we select 40 different adjective combinations for each adverb, extracting a context of 1-2 sentences between 10 and 40 words each time. For each adverb, we take the most frequent and least frequent combinations (excluding spelling errors) and then select 38 combinations at random from the relevant sample. 

Examples
| Sentences | Negative | Target| 
| :---: | :---: | ---- |
|  They don't want to be armed.  The ones that would want guns are `probably scary`.| They don't want to be armed.  The ones that would want guns are `not scary`.| probably scary |
|  Regardless of what you may think of him, this is `really touching`. |  Regardless of what you may think of him, this is `not touching`.|really touching|


### Task
We choose a fairly conservative task which could easily be set up for human evaluation for comparison purposes. For each example, we ask whether the model would predict the original intensifier rather than a negation (which would in most cases have the opposite meaning). To evaluate this, we check whether the original adverb ranks above the negation in the predictions for this position. This is because many things in the context can influence the ranking of an adverb but we want to focus on whether the language model is able to predict that a rank zero on the scale, or the opposite of an intensifier should be very unlikely in many of these sentences. Another reason for using this task is that it can easily be transformed into a classification task for humans (whereas it would be impossible to obtain rankings of words across all vocabulary from humans). 

For this comparison to be possible, the position of the negation should be the same as that of the original adverb. Thus, we modify the structure of the few sentences where a negation would not have seemed naturalistic, e.g., 'I'm glad he didn't take the job. He would have been frequently exhausted' => 'I'm annoyed he took the job. He is frequently exhausted'. This generally produces sentences where a negation could naturalistically be inserted, e.g., 'I'm annoyed he took the job. He is not exhausted.'

We replace the target position with the [MASK] token and obtain from the BERT large and BERT base cased models the prediction probabilities and ranks for the original adverb, negation ('not') and the top 10 predictions for error analysis. We also obtain the same for a truncated 'neutral' context as baseline on prior probabilities, e.g., 'is frequently exhausted' (we do not add pronouns such as 'It', 'She' or 'He' as each of these is shown to exert bias on predictions, but adding the verb 'is' prompts BERT to recognise the syntactic structure and suggest appropriate completions).

## Results 

### `BERT large`
We calculate the accuracy (number of times the original adverb is ranked above negation), MRR (mean reciprocal rank of original adverb) and 'diff' (the average difference between rank of adverb and rank of negation; a positive difference means on average negation has lower rank, comes above in the ranking)

- average accuracy:  0.39375000000000004
- MRR:  0.16998984322597085
- average diff:  17.412499999999998

Per adverb:

`NEUTRAL`

average accuracy: 0.45

MRR: 0.22

diff: 9.29
- `completely`: 
{'acc': 0.525, 'MRR': 0.09381856332307728, 'diff': -9.5}

- `seriously`:
{'acc': 0.4, 'MRR': 0.09616571616212585, 'diff': 53.825}

- `very`: 
{'acc': 0.725, 'MRR': 0.5503681016980435, 'diff': -22.425}

- `pretty`: 
{'acc': 0.625, 'MRR': 0.19027027367998, 'diff': -16.8}

-  `quite`: 
{'acc': 0.5, 'MRR': 0.2594916616158157, 'diff': 14.375}

-  `basically`: 
{'acc': 0.55, 'MRR': 0.3949152405912109, 'diff': -4.7}

- `slightly`: 
{'acc': 0.25, 'MRR': 0.07400610369280741, 'diff': 15.175}

- `hardly`: 
{'acc': 0.025, 'MRR': 0.0840489540439977, 'diff': 44.375}


`EPISTEMIC`

acc: 0.39

MRR: 0.14

diff: 27.44

- `definitely`: {'acc': 0.25, 'MRR': 0.08789945887132253, 'diff': 55.05}

- `certainly`: {'acc': 0.225, 'MRR': 0.11673036764580177, 'diff': 18.675}

- `really`: {'acc': 0.725, 'MRR': 0.24820748440023116, 'diff': -21.85}

- `actually`: {'acc': 0.35, 'MRR': 0.18352471606611626, 'diff': 5.9}

- `probably`: {'acc': 0.4, 'MRR': 0.10214808630050005, 'diff': 2.175}

- `possibly`, {'acc': 0.5, 'MRR': 0.12651156825159232, 'diff': 32.7})

- `perhaps`: {'acc': 0.275, 'MRR': 0.06421658869360565, 'diff': 42.0}

- `maybe`: {'acc': 0.425, 'MRR': 0.16357214371018916, 'diff': 84.9}



`TEMPORAL`

acc: 0.34

MRR: 0.15

diff: 15.50

- `always`: {'acc': 0.425, 'MRR': 0.2478908550977482, 'diff': 8.7}

- `frequently`: {'acc': 0.225, 'MRR': 0.02832851722070597, 'diff': 96.3}
- `generally`: {'acc': 0.35, 'MRR': 0.10861424506381287, 'diff': -0.375}
- `usually`: {'acc': 0.35, 'MRR': 0.11513225070964965, 'diff': 4.85}
- `often`: {'acc': 0.375, 'MRR': 0.22101984882358305, 'diff': -2.475}
- `already`: {'acc': 0.25, 'MRR': 0.09974669209479751, 'diff': 20.175}
- `sometimes`: {'acc': 0.55, 'MRR': 0.18005722093943818, 'diff': -12.925}
- `never`: {'acc': 0.175, 'MRR': 0.24307157872714832, 'diff': 9.775}


### `BERT 
average accuracy:  0.09
average MRR:  0.09
average diff:  161.80

`NEUTRAL`
- acc: 0.17 (with context: +0.28)
- MRR: 0.18 (with context: +0.04)
- diff: 75.28125 (with context: -64)


`EPISTEMIC`
- acc: 0.03 (with context: +0.36)
- MRR: 0.03 (with context: +0.11)
- diff: 323.08 (with context: -294)


`TEMPORAL`
- acc: 0.06 (with context: +0.27)
- MRR: 0.06 (with context: +0.09)
- diff: 87.04 (with context: -72)

### Discussion

The accuracy of BERT large is on average 0.39, which means it generally fails to rank the original adverb above negation. The only adverbs for which the accuracy is above chance are the common 'very', 'pretty' and 'really'. However, the performance does benefit from context, as the accuracy improves by 30% when context is added. The epistemic category gets the highest boost with an improvement of 36% on average. 

While the accuracy is not perfectly correlated with intensity of adverbs, it does correlate to a certain extent, with the model having the most difficulty distinguishing original 'hardly' and 'never' from 'not', as expected, and intense adverbs ('very', 'really', 'always', 'completely') generally having higher accuracies. 

![Heatmap](heatmap.jpg)