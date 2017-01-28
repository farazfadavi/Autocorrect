import math
import collections
from termcolor import colored


class LaplaceBigramLanguageModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.bigramCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        for sentence in corpus.corpus:
            prev = sentence.data[0].word
            for datum in sentence.data[1:]:
                token = (prev, datum.word)
                prev = datum.word
                self.bigramCounts[token] = self.bigramCounts[token] + 1
                self.total += 1

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        score = 0.0
        prev = sentence[0]
        for word in sentence[1:]:
            token = (prev, word)
            prev = word
            count = self.bigramCounts[token]
            score += math.log(count + 1)
            score -= math.log(self.total + len(self.bigramCounts))
        return score
