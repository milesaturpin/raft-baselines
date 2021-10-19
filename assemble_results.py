import pandas as pd
import json

from pandas.io.pytables import AppendableFrameTable

files = [
#     "cohere_results/20211013_0448.json",
# "cohere_results/20211013_0451.json",
# "cohere_results/20211013_0727.json",
# "cohere_results/20211013_0732.json",
# "cohere_results/20211013_0745.json",
# "cohere_results/20211014_0005.json",
# "cohere_results/20211014_1728.json"
# "cohere_results/20211014_1755.json",
"cohere_results/20211014_1946.json"
]

results=[]

for fname in files:
    with open(fname, 'r') as f:
        results.append(json.load(f))
    f.close()

import datasets
configs = datasets.get_dataset_config_names("ought/raft")
configs.remove('banking_77')

dfs = []
for i, result in enumerate(results):

    result.update(result['classifier_kwargs'])
    # result['zero_shot'] = all([int(x) == 0 for x in list(result['num_examples'].values())])
    result['fname'] = files[i]
    del result['classifier_kwargs']
    del result['num_examples']
    
    zero_shot = True
    for config in configs:
        del result[config]['y_true']
        del result[config]['y_pred']
        zero_shot = zero_shot and result[config]['num_prompt_training_examples'] == 0
        # result[config] = {col: result[config][col] for col in ['config', 'acc', 'f1', 'micro_f1', 'num_prompt_training_examples']]
    result['zero_shot'] = zero_shot
    tmp = pd.DataFrame.from_dict(result)
    tmp.index.name = 'metric'
    dfs.append(tmp)

df = pd.concat(dfs).set_index('fname', append=True).sort_index(level=[0,1])
df = df.drop(index='config')

df = df.reset_index()
print(df)
df['banking_77'] = None
df['data'] = 'train'
df['avg'] = None

for col in ['max_tokens','do_semantic_selection', 'description', 'scoring_method']:
    if col not in df.columns:
        df[col] = None
df = df[['model_type', 
'metric',
'fname',
'scoring_method',
'description',
'data',
'do_semantic_selection',
'max_tokens',
'zero_shot',
'avg',
 'ade_corpus_v2', 
 'banking_77',
       'neurips_impact_statement_risks', 
       'one_stop_english', 
       'overruling',
       'semiconductor_org_types', 
       'systematic_review_inclusion', 
'tai_safety_research',
'terms_of_service', 
       'tweet_eval_hate',
       'twitter_complaints', 
       ]]
print(df)
print(df.columns)
df.to_csv('experiment4.csv')
    

