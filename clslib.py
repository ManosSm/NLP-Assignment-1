from tokenizers import Tokenizer
import numpy as np

class NamedClasses:
    """
    This is a helper class that wraps around a list of class names and provides 
    functions to map from names (str) to integers.
    """

    def __init__(self, names: list):
        self.names = tuple(names)        
        self._str2int = {name: k for k, name in enumerate(names)}
    
    def index(self, name: str):
        """Convert name to 0-based integer (0-based because Python lists are 0-based)"""
        return self._str2int[name]  # if the name is not recognised, this will raise a ValueError

    def name(self, index: int):
        """Convert an 0-based integer to a class name"""
        return self.names[index]  # if the index is out of bounds, this will raise a IndexError        

    def __len__(self):
        """Size of label space"""
        return len(self.names)

    def __iter__(self):
        """Iterate over names"""
        return iter(self.names)

    def items(self):
        """Iterates over pairs of the kind (index, name) for all categories in the space"""
        return enumerate(self.names)

    def __str__(self):
        return ", ".join(f"{i}: {n}" for i, n in self.items())

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.names)})"
        

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