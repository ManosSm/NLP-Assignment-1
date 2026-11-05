from tokenizers import Tokenizer
import numpy as np

class Encoder:

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        raise NotImplementedError("Implement me in a sub-class")

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        raise NotImplementedError("Implement me in a sub-class")

class BagOfWords(Encoder):

    def __init__(self, vocab: list, tokenizer: Tokenizer):
        """
        This encoder has one dimension for each symbol in 'vocab'. 
        """
        self._vocab = tuple(vocab)
        self.tokenizer = tokenizer
    
    def __len__(self):
        """Number of dimensions"""
        return len(self._vocab)

    def __call__(self, x):
        """
        Our encoder employs BPE crude tokeniser and counts 
        the number of appearances in x of the words in its vocabulary.
        """
        tokens = self.tokenizer.encode(x).tokens
        return np.array([tokens.count(w) for w in self._vocab])