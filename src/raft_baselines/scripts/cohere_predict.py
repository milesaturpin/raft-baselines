import datasets
from raft_baselines.classifiers import CohereClassifier, TransformersCausalLMClassifier
from sklearn.metrics import accuracy_score, f1_score

# prompt="""Label the sentence based on whether it is related to an adverse drug effect (ADE). Details are described below:
# Drugs: Names of drugs and chemicals that include brand names, trivial names, abbreviations and systematic names were annotated. Mentions of drugs or chemicals should strictly be in a therapeutic context. This category does not include the names of metabolites, reaction byproducts, or hospital chemicals (e.g. surgical equipment disinfectants).
# Adverse effect: Mentions of adverse effects include signs, symptoms, diseases, disorders, acquired abnormalities, deficiencies, organ damage or death that strictly occur as a consequence of drug intake.
# Possible labels:
# 1. ADE-related
# 2. not ADE-related

# Sentence: The mechanism by which sunitinib induces gynaecomastia is thought to be associated with an unknown direct action on breast hormonal receptors.
# Label:"""
# co.choose_best(model='baseline-shrimp',query=prompt, options=['ADE-related', 'not ADE-related'], mode='APPEND_OPTION')


class Pcolour:
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def colorize(text, color):
    return f"{color}{text}{Pcolour.END}"
def blue(*args):
    return colorize(" ".join(map(str, list(args))), Pcolour.BLUE)
def magenta(*args):
    return colorize(" ".join(map(str, list(args))), Pcolour.MAGENTA)
def red(*args):
    return colorize(" ".join(map(str, list(args))), Pcolour.RED)

config_kwargs = dict(
    # random_seed = 42,
    random_seed = 9385,
    # random_seed = 4387,
    zero_shot = False,
    should_print_prompt = True,

    # model = 'baseline-seal'
    # model = 'baseline-shark'
    # model = 'baseline-orca'
    # model_type = 'distilgpt2'
    # model_type = 'gpt2-xl'
    # model_type = 'EleutherAI/gpt-neo-2.7B'
    model = 'baseline-shrimp'
    # model = 'baseline-shrimp'
)

# scoring method, do semantic selection,

classifier_kwargs = dict(
    # n_test = 5,
    # scoring_method = 'first_token_likelihood' if not zero_shot else 'avg_log_likelihood',  # first_token_likelihood or avg_log_likelihood
    scoring_method = 'first_token_likelihood',  # first_token_likelihood or avg_log_likelihood
    # scoring_method = 'avg_log_likelihood',  # first_token_likelihood or avg_log_likelihood
    engine = config_kwargs['model'],
    # model_type = config_kwargs['model_type'],
    do_semantic_selection = False,
    max_tokens = 1024 if config_kwargs['model'] in ['baseline-shrimp', 'baseline-otter', 'baseline-seal'] else 2048
    # max_tokens = 1024 if config_kwargs['model_type'] in ['baseline-shrimp', 'baseline-otter', 'baseline-seal'] else 2048
    )


# if zero_shot:
#     print('USING AVG LIKELIHOOD BC ZERO SHOT')

configs = datasets.get_dataset_config_names("ought/raft")
configs.remove('banking_77')

# configs = ['twitter_complaints']

# configs = ['ade_corpus_v2']
# configs = ['banking_77']

NUM_EXAMPLES = {
    "ade_corpus_v2": 25,
    # "ade_corpus_v2": 4,
    "banking_77": 10,
    "terms_of_service": 5,
    "tai_safety_research": 5,
    "neurips_impact_statement_risks": 25,
    "overruling": 25,
    "systematic_review_inclusion": 10,
    "one_stop_english": 3, # TODO: normally 5 but was giving too long errors
    "tweet_eval_hate": 50,
    "twitter_complaints": 25,
    "semiconductor_org_types": 5,
}

# # TODO: uncomment for Cohere
if classifier_kwargs['max_tokens'] == 1024:
    NUM_EXAMPLES = {k: v//2 for k, v in NUM_EXAMPLES.items()}

# train = datasets.load_dataset(
#     "ought/raft", "neurips_impact_statement_risks", split="train"
# )

# import ipdb; ipdb.set_trace()

# acc_scores={}
# f1_scores={}
results = {}
results['classifier_kwargs'] = classifier_kwargs
results.update({'num_examples': NUM_EXAMPLES})

for config in configs:
    results[config] = {}
    print('\n', magenta(config),'\n')
    train_dataset = datasets.load_dataset("ought/raft", config, split="train")
    print(magenta(str(train_dataset.features['Label'])))
    for clas in train_dataset.features['Label'].names:
        print(clas, train_dataset.features['Label'].str2int(clas))
    extra_kwargs = {
        "config": config,
        "num_prompt_training_examples": NUM_EXAMPLES[config] if not config_kwargs['zero_shot'] else 0,
    }
    if config == "banking_77":
        extra_kwargs["add_prefixes"] = True

    # initialize with a dict
    results[config].update(extra_kwargs)

    # classifier = TransformersCausalLMClassifier(
    #     train_dataset, #config=train_dataset, 
    #     **classifier_kwargs, **extra_kwargs
    # )
    # TODO change
    classifier = CohereClassifier(
        train_dataset, #config=train_dataset, 
        **classifier_kwargs, **extra_kwargs
    )


    correct = 0
    n = 0
    probs = []
    y_pred = []
    y_true = []
    results[config]['num_failed'] = 0

    def compute_and_store_metrics(y_true, y_pred):
        global results
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=range(1, train_dataset.features['Label'].num_classes), average='macro')
        micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=range(1, train_dataset.features['Label'].num_classes), average='micro')
        results[config]['acc'] = acc
        results[config]['f1'] = f1
        results[config]['micro_f1'] = micro_f1
        print(red('Accuracy:', acc))
        print(red('F1 (macro):', f1))
        print(red('Micro F1:', micro_f1))

    def predict(example):
        global correct, n, probs, y_pred, y_true
        # import ipdb; ipdb.set_trace()
        label_true = example["Label"]
        output_probs = classifier.classify(example, random_seed=config_kwargs['random_seed'], should_print_prompt=config_kwargs['should_print_prompt'])
        output = max(output_probs.items(), key=lambda kv_pair: kv_pair[1])
        probs.append(output_probs)

        label_pred = train_dataset.features["Label"].str2int(output[0])
        example["Label Prediction"] = label_pred
        print(magenta('Prediction:'), output[0])
        print(magenta('True:'), train_dataset.features["Label"].int2str(label_true))

        y_pred.append(label_pred)
        y_true.append(label_true)
        compute_and_store_metrics(y_true=y_true, y_pred=y_pred)

        # import ipdb; ipdb.set_trace()
        return example

    for i, example in enumerate(train_dataset):
        try:
            predict(example)
        except Exception as e:
            print(e)
            results[config]['num_failed'] += 1

        # if i ==3:
        #     break

    # train_dataset.map(predict, load_from_cache_file=False)

    # overwrite for final time
    compute_and_store_metrics(y_true=y_true, y_pred=y_pred)
    results[config].update({'y_true': y_true, 'y_pred':y_pred})

print('Accuracies')
print({config: results[config]['acc'] for config in configs})
print('F1 scores')
print({config: results[config]['f1'] for config in configs})
print('micro F1 scores')
print({config: results[config]['micro_f1'] for config in configs})

import json

from datetime import datetime
date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
with open(f'cohere_results/{date}.json', 'w') as f:
    json.dump(results, f)

