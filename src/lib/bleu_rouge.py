import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

class bleu_rouge:
    """
    A class for comparing bleu and rouge scores of generated text with referenced text.
    It also generates samples of prompt and their next sentence as reference
    from a corpus.
    ----------------------------------------------------------------------
    """    

    def get_prompt_reference(self, corpus, num_ref=10, seed=2022):
        """Get samples of prompt and their next sentence as reference

        Args:
        corpus (list): corpus consisting of list of sentences to sample the prompts from
        num_ref (int): number of prompts and refs to sample (defaults to 10)
        seed (int): seed for replicability
        
        Corpus should be a list of sentences. To separate a document from another, 
        insert an empty string between them.

        Returns:
        dict containing sampled prompts and their reference (next line)
        """
        rng = np.random.default_rng(seed)
        # random.seed(seed)
        refs = set()

        while len(refs) < num_ref:
            idx = rng.integers(0,len(corpus)-1) # minus 1 as last sent has no next sent
            sample = corpus[idx]
            next_sent = corpus[idx+1]
            if sample and next_sent: # check sample and next line is not empty
                refs.add((sample, next_sent))

        prompt_ref = {}
        for k,v in refs:
            prompt_ref[k] = v
        
        self.num_ref = num_ref
        self.prompt_ref = prompt_ref

        return prompt_ref


    def compute_bleu(self, prompt, generated_text, verbose=True):
        """Computes the cumulative n-gram bleu score up to 4-gram. The average is also
        returned.

        Args:
        prompt (str): prompt used to generate text. this should be from the sampled prompts
        generated_text (str): the generated text using the prompt
        verbose (bool): prints out the tokenized references and generated text 
        
        Returns:
        dict containing scores for BLEU-1 to BLEU-4 and the average of them
        """

        ref = self.prompt_ref.get(prompt)
        if not ref:
            error_text = """PROMPT does NOT exist in sampled prompts.
            Run get_prompt_reference() to get prompt samples and check
            bleu_rogue.prompt_ref for the set of prompts and references"""
            raise AttributeError(error_text)

        ref = [ref.split(' ')] # ref is list of tokens in list of ref
        generated_text = generated_text.split(' ')
        if verbose:
            print('Reference is {}'.format(ref))
            print('Generated text is {}'.format(generated_text))

        bleu1 = sentence_bleu(ref, generated_text, weights=(1, 0, 0, 0))
        bleu2 = sentence_bleu(ref, generated_text, weights=(0.5, 0.5, 0, 0))
        bleu3 = sentence_bleu(ref, generated_text, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = sentence_bleu(ref, generated_text, weights=(0.25, 0.25, 0.25, 0.25))
        bleus = [bleu1, bleu2, bleu3, bleu4]
        bleus = [bleu for bleu in bleus if not bleu < 0.0001] # remove bleu-n that are very small due to no overlap
        avg_bleu = sum(bleus) / len(bleus)

        self.bleu_scores = {'BLEU-1':bleu1, 'BLEU-2':bleu2, 'BLEU-3':bleu3, 'BLEU-4':bleu4, 'Avg':avg_bleu}

        return {'BLEU-1':bleu1, 'BLEU-2':bleu2, 'BLEU-3':bleu3, 'BLEU-4':bleu4, 'Avg':avg_bleu}

    