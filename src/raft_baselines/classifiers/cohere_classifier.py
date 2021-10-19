from typing import Dict, Optional, List, Mapping

import numpy as np
import datasets
import cohere
import time


from raft_baselines.classifiers.in_context_classifier import InContextClassifier
# from raft_baselines.utils.gpt3_utils import (
#     complete,
#     search,
# )
from raft_baselines.utils.tokenizers import CohereTokenizer

apikey = 'maG5fNuvc4bBqnI0MMgzz5hWQi8xOIoeNvI3gxJi'
co = cohere.CohereClient(apikey)

# COHERE_MAX_TOKENS = 1024
# COHERE_MAX_TOKENS = 2048
# COHERE_MAX_TOKENS = 1800
# tokenizer = TransformersTokenizer("gpt2")

def choose_best(model ,query, options, mode='APPEND_OPTION'):
    success = False
    retries = 0
    while not success:
        try:
            response = co.choose_best(model=model,query=query, options=options, mode=mode)
            success = True
        except Exception as e:
            print(f"Exception in Cohere Platform: {e}")
            retries += 1
            if retries > 3:
                raise Exception("Max retries reached")
                break
            else:
                print("retrying")
                time.sleep(retries * 3)
    return response


class CohereClassifier(InContextClassifier):

    def __init__(
        self,
        *args,
        engine: str = "baseline-shrimp",
        scoring_method: str = "first_token_likelihood",
        max_tokens: int = 2048,
        # search_engine: str = "ada",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            tokenizer=CohereTokenizer(),
            # max_tokens=COHERE_MAX_TOKENS,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.scoring_method = scoring_method
        self.engine: str = engine
        # self.search_engine: str = search_engine

    def semantically_select_training_examples(self):
        raise NotImplementedError()

    # def does_token_match_class(self, token: str, clas: str) -> bool:
    #     # prepend a space to the class label
    #     # because we always expect a leading space in the first token
    #     # returned from the OpenAI API, given our prompt format
    #     clas_str = (f" {clas}" if not self.add_prefixes else f" {self.classes.index(clas) + 1}")

    #     clas_first_token_id: int = self.tokenizer(clas_str)["input_ids"][0]
    #     token_id: int = self.tokenizer(token)["input_ids"][0]

    #     # Compare token ids rather than the raw tokens
    #     # because GPT2TokenizerFast represents some special characters
    #     # differently from the GPT-3 API
    #     # (e.g. the space at the beginning of the token is " " according to the API,
    #     # but "Ġ" according to the tokenizer.
    #     # Standardizing to token ids is one easy way to smooth over that difference.
    #     return clas_first_token_id == token_id

    # def _classify_prompt(
    #     self,
    #     prompt: str,
    # ) -> Dict[str, float]:
    #     # raw_p = self._get_raw_probabilities(prompt)
    #     # sum_p = np.sum(raw_p)
    #     # if sum_p > 0:
    #     #     normalized_p = np.array(raw_p) / np.sum(raw_p)
    #     # else:
    #     #     normalized_p = np.full(len(self.classes), 1 / len(self.classes))
    #     # class_probs = {}
    #     # for i, clas in enumerate(self.classes):
    #     #     class_probs[clas] = normalized_p[i]
    #     import ipdb; ipdb.set_trace()
        
    #     return response

    def _get_raw_probabilities(
        self,
        prompt: str,
    ) -> List[float]:
        # import ipdb; ipdb.set_trace()


        classes = [" " + clas for clas in self.classes]
        response = choose_best(model=self.engine ,query=prompt, options=classes, mode='APPEND_OPTION')
        # print(response.scores)
        # print(response.token_log_likelihoods)
        # print([x[0] for x in response.token_log_likelihoods])
        if self.scoring_method == 'first_token_likelihood':
            print([x[0] for x in response.token_log_likelihoods])
            raw_p = np.exp(np.array([x[0] for x in response.token_log_likelihoods]))
        elif self.scoring_method == 'avg_log_likelihood':
            print(response.scores)
            raw_p = np.exp(np.array(response.scores))
        else:
            raise NotImplementedError()
        return raw_p