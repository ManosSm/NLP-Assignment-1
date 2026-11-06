from tokenizers import Tokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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

class TokenIndicator(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding 
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1

        return vector


class TFIDF(Encoder):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda text: tokenizer.encode(text).tokens,
            vocabulary=tokenizer.get_vocab(),
            lowercase=False,
            token_pattern=None
        )

    def __len__(self):
        return len(self.tokenizer.get_vocab())

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def __call__(self, text):
        return self.vectorizer.transform([text]).toarray()[0]

    def save(self, path):
        np.save(path, self.vectorizer.idf_)

    def load(self, path):
        self.vectorizer.idf_ = np.load(path)
        return self