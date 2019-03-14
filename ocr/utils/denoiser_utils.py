import gluonnlp as nlp
import leven
import mxnet as mx
import numpy as np

from ocr.utils.encoder_decoder import decode_char

class SequenceGenerator:
    
    def __init__(self, sampler, language_model, vocab, ctx_nlp, tokenizer=nlp.data.SacreMosesTokenizer(), detokenizer=nlp.data.SacreMosesDetokenizer()):
        self.sampler = sampler
        self.language_model = language_model
        self.ctx_nlp = ctx_nlp
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer

    def generate_sequences(self, inputs, begin_states, sentence):
        samples, scores, valid_lengths = self.sampler(inputs, begin_states)
        samples = samples[0].asnumpy()
        scores = scores[0].asnumpy()
        valid_lengths = valid_lengths[0].asnumpy()
        max_score = -10e20

        # Heuristic #1
        #If the sentence is correct, let's not try to change it 
        sentence_tokenized = [i.replace("&quot;", '"').replace("&apos;","'").replace("&amp;", "&") for i in self.tokenizer(sentence)]
        sentence_correct = True
        for token in sentence_tokenized:
            if (token not in self.vocab or self.vocab[token] > 400000) and token.lower() not in ["don't", "doesn't", "can't", "won't", "ain't", "couldn't", "i'd", "you'd", "he's", "she's", "it's", "i've", "you've", "she'd"]:
                sentence_correct = False
                break
        if sentence_correct:
            return sentence

        # Heuristic #2
        # We want sentence that have the most in-vocabulary words
        # and we penalize sentences that have out of vocabulary words 
        # that do not start with a capital letter
        for i, sample in enumerate(samples):
            tokens = decode_char(sample[:valid_lengths[i]])
            tokens = [i.replace("&quot;", '"').replace("&apos;","'").replace("&amp;", "&") for i in self.tokenizer(tokens)]
            score = 0

            for t in tokens:
                # Boosting names
                if (t in self.vocab and self.vocab[t] < 450000) or (len(t) > 0 and t.istitle()):
                    score += 0
                else:
                    score -= 1
                score -= 0
            if score == max_score:
                max_score = score
                best_tokens.append(tokens)
            elif score > max_score:
                max_score = score
                best_tokens = [tokens]

        # Heurisitic #3
        # Smallest edit distance
        # We then take the sentence with the lowest edit distance
        # From the predicted original sentence
        best_dist = 1000
        output_tokens = best_tokens[0]
        best_tokens_ = []
        for tokens in best_tokens:
            dist = leven.levenshtein(sentence, ' '.join(self.detokenizer(tokens)))
            if dist < best_dist:
                best_dist = dist
                best_tokens_ =[tokens]
            elif dist == best_dist:
                best_tokens_.append(tokens)

        # Heuristic #4
        # We take the sentence with the smallest number of tokens 
        # to avoid split up composed words
        min_len = 10e20
        for tokens in best_tokens_:
            if len(tokens) < min_len:
                min_len = len(tokens)
                best_tokens__ = [tokens]
            elif len(tokens) == min_len:
                best_tokens__.append(tokens)

        # Heuristic #5 
        # Lowest ppl
        # If we still have ties we take the sentence with the lowest
        # Perplexity score according to the language model
        best_ppl = 10e20            
        for tokens in best_tokens__:
            if len(tokens) > 1:
                inputs = self.vocab[tokens]
                hidden = self.language_model.begin_state(batch_size=1, func=mx.nd.zeros, ctx=self.ctx_nlp)
                output, _ = self.language_model(mx.nd.array(inputs).expand_dims(axis=1).as_in_context(self.ctx_nlp), hidden)
                output = output.softmax()
                l = 0
                for i in range(1, len(inputs)):
                    l += -output[i-1][0][inputs[i]].log()
                ppl = (l/len(inputs)).exp()
                if ppl < best_ppl:
                    output_tokens = tokens
                    best_ppl = ppl
        output = ' '.join(self.detokenizer(output_tokens))


        # Heuristic #6
        # Sometimes there are artefact at the end of the corrected sentence
        # We cut the end of the sentence
        if len(output) > len(sentence) + 10:
            output = output[:len(sentence)+2]
        return output