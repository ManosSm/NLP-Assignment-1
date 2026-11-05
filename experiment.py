from pathlib import Path
from collections import OrderedDict
from tqdm.auto import tqdm
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import random
import numpy as np
import itertools

import smlp

def train_bpe_tokenizer(corpus, vocab_size, min_frequency, pretokenizer=Metaspace(), unk_token="[UNK]"):
    """
    corpus: a list of texts (each a string)
    vocab_size: number of distinct BPE tokens
    min_frequency: we only merge token pairs that are frequent enough
    pretokenizer: a simple tokenizer that we use before BPE tokenization
    
        You probably don't need to change the pretokenizer, unless you are working with
        languages that really don't use any form of blank space 
        (then you could, for example, pretokenize at character level)
    """
    # Create an empty BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token=unk_token)) #unk_token used for characters that cannot be encoded

    # First split the corpus into words
    tokenizer.pre_tokenizer = pretokenizer

    # Tell BPE how large the vocabulary should be
    trainer = BpeTrainer(
        vocab_size=vocab_size, # The size of the vocabulary to learn.
        min_frequency=min_frequency, # Only perform a merge if the pair occurs at least min_frequency times in the training corpus.
        special_tokens=[unk_token]
    )

    # Learn the vocabulary and merge rules
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    return tokenizer

def cross_validation_split(pairs, k, seed=0):
    """
    Shuffles a list of labelled texts and splits the result into k folds.
    
    pairs: lists of tuples (text, label)
    k: number of folds
    seed: for reproducible shuffling
    """
    
    # Shuffle the pairs
    random.seed(seed)
    pairs = list(pairs)
    ids = list(range(len(pairs)))
    random.shuffle(ids)

    # Split without coercing (text, int) tuples into a string array
    ids_parts = np.array_split(np.array(ids, dtype=object), k)
    
    # Create a dictionary to store the training and validation sets for each fold
    cross_validation_sets = {}
    for i in range(k):
        validation_set = [pairs[j] for j in ids_parts[i]] 
        training_ids = list(itertools.chain.from_iterable(part for k, part in enumerate(ids_parts) if k != i))
        training_set = [pairs[j] for j in training_ids]
        cross_validation_sets[i] = {
            "training": training_set, 
            "validation": validation_set, 
            'training_ids': training_ids,   # these are not often needed, but we return it just in case
            'validation_ids': ids_parts[i]  # these are not often needed, but we return it just in case
        }

    return cross_validation_sets

def cross_validation_experiment(data_splits, encoder: 'Encoder', model_option:str, save_to="tmp", smlp_args=dict()):
    """
    Trains and evaluates k models using a k-fold data split for cross validation. 
    This function returns a dictionary with 
        - models: the paths to the k models trained (the model objects aren't returned, they are stored in disk)
        - predictions: the predictions by each of the k models
        - accuracies: the accuracy of each trained model on its corresponding validation partition
    
    data_splits: the training/validation folds
    encoder: an instance of Encoder
    model_option: for now this only supports 'logistic'
        which tells this function to use train_logistic_model (defined earlier in this notebook)
    save_to: folder where to store models
    smlp_args: dict with training arguments for smlp.train_neural_network_model
    """

    models, predictions, accuracies = OrderedDict(), OrderedDict(), OrderedDict()
    
    for fold in tqdm(data_splits.keys()):
        training_set = data_splits[fold]["training"]
        validation_set = data_splits[fold]["validation"]

        fold_folder = Path(f"{save_to}/fold-{fold}")
        fold_folder.mkdir(parents=True, exist_ok=True) 

        #TODO add for other models
        if model_option == 'mlp':            
            metrics, best_epoch, predictions[fold] = smlp.train_neural_network_model(
                training_set, validation_set, encoder, save_to=f'{fold_folder}/model', **smlp_args
            )
            models[fold] = f'{fold_folder}/model'
            accuracies[fold] = metrics[best_epoch]['val_metrics']['accuracy']            
        # elif model_option == 'logistic':                        
        #     llmodel = loglin.fit(training_set, encoder, save_to=f'{fold_folder}/model', **loglin_args) 
        #     models[fold] = f'{fold_folder}/model'
            
        #     predictions[fold] = loglin.predict([text for text, gold_label in validation_set], llmodel, encoder)    
        #     gold_labels_validation = [gold_label for text, gold_label in validation_set]
        #     accuracies[fold] = accuracy_score(gold_labels_validation, predictions[fold])
        else:
            raise ValueError(f"Unkonwn model_option={model_option}")

    return {'models': models, 'predictions': predictions, 'accuracies': accuracies}