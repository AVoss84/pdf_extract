import nltk, re, string, gensim, spacy, fasttext        # for type hints
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from typing import (Dict, List, Text, Optional, Any, Callable, Union)
from gensim.models import FastText, Phrases, phrases, TfidfModel
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
from gensim import corpora
#from gensim.models.word2vec import Word2Vec
from pprint import pprint
from gensim.parsing.preprocessing import stem_text, strip_multiple_whitespaces, strip_short, strip_non_alphanum, strip_punctuation, strip_numeric
from copy import deepcopy
from tqdm.auto import tqdm
import pandas as pd
from pdf_extract.services import file
from pdf_extract.config import global_config as glob

nltk.download('punkt')
nltk.download('stopwords')

class clean_text(BaseEstimator, TransformerMixin):

    def __init__(self, verbose : bool = True, language : str = 'english', stem : bool = False, lemma : bool = False, **kwargs):
        """Utility class for text preprocessing, i.e. cleaning + lemmatization or stemming

        Args:
            verbose (bool, optional): Show user info. Defaults to True.
            language (str, optional): used language. Defaults to 'german'.
            stem (bool, optional): Should stemming be used? Defaults to False.
            lemma (bool, optional): Should lemmatization be used? Defaults to False.
        """
        self.verbose = verbose
        self.kwargs = kwargs
        self.stemming = stem
        self.lemma = lemma
        self.language = language
        assert (self.lemma & self.stemming) != True, 'Use either lemmatization or stemming!'
        self.stop_words = set(stopwords.words(self.language)) 
        if self.verbose: print(f'Using {self.language} language.'); print(f'Using {len(self.stop_words)} stop words.') 
        # English stopwords:
        self.own_stopwords = file.TXTService(verbose=False, root_path=glob.UC_CODE_DIR + '/pdf_extract/config', path='stopwords_eng.txt').doRead()
        self.stop_words = self._add_stopwords(self.own_stopwords)
        # German stopwords:
        self.own_stopwords = file.TXTService(verbose=False, root_path=glob.UC_CODE_DIR + '/pdf_extract/config', path='stopwords_ger.txt').doRead()
        self.stop_words = self._add_stopwords(self.own_stopwords)
        if self.verbose: print(f'Adding custom stop words...') 

        if 'without_stopwords' in list(self.kwargs.keys()):
            self.stop_words = self._remove_stopwords(self.kwargs.get('without_stopwords', ''))
                
        if 'with_stopwords' in list(self.kwargs.keys()):
            self.stop_words = self._add_stopwords(self.kwargs.get('with_stopwords', '')) 
            
        if self.stemming:
            self.stemmer = SnowballStemmer(self.language); print("Loading nltk stemmer.")
            
        if self.lemma and (self.language == 'english'):
            self.nlp = spacy.load('en_core_web_lg'); print("Loading spaCy embeddings for lemmatization.")
        if self.lemma and (self.language == 'german'):
            self.nlp = spacy.load('de_core_news_lg'); print("Loading spaCy embeddings for lemmatization.")
            
        #self.umlaut = file.YAMLservice(root_path = glob.UC_CODE_DIR + '/claims_topics/config', path = 'preproc_txt.yaml').doRead()

    def _add_stopwords(self, new_stopwords : Union[List, None])-> set:
        """
        Change stopword list. Include into stopword list.
        Args:
            new_stopwords (list): _description_
        Returns:
            set: updated stopwords
        """
        if new_stopwords:
            old = self.stop_words.copy()
            self.stop_words = self.stop_words.union(set(new_stopwords))
            if self.verbose: print(f"Added {len(self.stop_words)-len(old)} stopword(s).")
            return self.stop_words

    def _remove_stopwords(self, without_stopwords : Union[List, None])-> set:
        """
        Change stopword list. Exclude from stopwords
        Args:
            without_stopwords (list): _description_
        Returns:
            set: updated stopwords
        """
        if without_stopwords and self.stop_words:
            old = self.stop_words.copy()
            self.stop_words = self.stop_words.difference(set(without_stopwords))
            if self.verbose: print(f"Removed {len(old)-len(self.stop_words)} stopword(s).")
            return self.stop_words

    def untokenize(self, text: List[str])-> str:
        """Revert tokenization: list of strings -> string"""
        return " ".join([w for w in text])

    def count_stopwords(self):
        print(f'{len(self.stop_words)} used.')
 
    def remove_whitespace(self, text : str)-> str:
        """Remove whitespaces"""
        return  " ".join(text.split())

    def remove_punctuation(self, text: str)-> str:  
       """Remove punctuation"""  
       return [re.sub(f"[{re.escape(punctuation)}]", " ", token) for token in text]

    def remove_numbers(self, text: str)-> str:    
        """Remove numbers
        Args:
            text (str): _description_
        Returns:
            str: _description_
        """
        return [re.sub(r"\b[0-9]+\b\s*", "", token) for token in text]

    def remove_stopwords(self, text : str)-> str:
        """Remove stopwords
        Args:
            text (str): _description_
        Returns:
            str: _description_
        """
        return [token for token in text if token not in self.stop_words]

    def remove_digits(self, text: str)-> str: 
        """Remove digits instead of any number, e.g. keep dates"""
        return [token for token in text if not token.isdigit()]

    def remove_non_alphabetic(self, text: str)-> str: 
        """Remove non-alphabetic characters"""
        return [token for token in text if token.isalpha()]
    
    def remove_spec_char_punct(self, text: str)-> str: 
        """Remove all special characters and punctuation"""
        return [re.sub(r"[^A-Za-z0-9\s]+", "", token) for token in text]

    def remove_short_tokens(self, text: str, token_length : int = 2)-> str: 
        """Remove short tokens"""
        return [token for token in text if len(token) > token_length]

    def remove_punct(self, text: str)-> str:
        """Remove punctuations"""
        tokenizer = RegexpTokenizer(r"\w+")
        lst = tokenizer.tokenize(' '.join(text))
        # table = str.maketrans('', '', string.punctuation)          # punctuation
        # lst = [w.translate(table) for w in text]     # Umlaute
        return lst

    def replace_umlaut(self, text : str) -> str:
        """Replace special German umlauts (vowel mutations) from text"""
        vowel_char_map = {ord(k): v for k,v in self.umlaut['umlaute'].items()}  # use unicode value of Umlaut
        return [token.translate(vowel_char_map) for token in text]

    def stem(self, text : str)-> str:
        """Apply nltk stemming"""
        return [self.stemmer.stem(w)  for w in text]
    
    def lemmatize(self, text : str)-> str:
        """Apply spaCy lemmatization"""
        text = self.untokenize(text)
        return [token.lemma_.lower() for token in self.nlp(text)]

    def fit(self, X : pd.DataFrame, y : pd.Series = None):
        return self    
    
    def transform(self, X : pd.Series, **param)-> pd.Series:    
        """Preprocess text
        Args:
            X (pd.Series): Input corpus
        Returns:
            pd.Series: transformed corpus
        """
        corpus = deepcopy(X)
        if self.verbose: print("Setting to lower cases.")
        corpus = corpus.str.lower()
        if self.verbose: print("Removing whitespaces.")
        corpus = corpus.apply(self.remove_whitespace)
        if self.verbose: print("Applying word tokenizer.")
        corpus = corpus.apply(lambda x: word_tokenize(x))
        if self.verbose: print("Removing custom stopwords.") 
        corpus = corpus.apply(self.remove_stopwords)
        if self.verbose: print("Removing punctuations.")
        corpus = corpus.apply(self.remove_punct)
        if self.verbose: print("Removing numbers.")
        corpus = corpus.apply(self.remove_numbers)
        if self.verbose: print("Removing digits.")
        corpus = corpus.apply(self.remove_digits)
        if self.verbose: print("Removing non-alphabetic characters.")
        corpus = corpus.apply(self.remove_non_alphabetic)
        #if self.verbose: print("Replacing German Umlaute.") 
        #corpus = corpus.apply(self.replace_umlaut)  
        #if self.verbose: print("Removing special character punctuations.")
        #corpus = corpus.apply(self.remove_spec_char_punct)
        if self.verbose: print("Removing short tokens.")
        corpus = corpus.apply(self.remove_short_tokens, token_length=3)
        if self.stemming: 
            if self.verbose: print("Applying stemming.") 
            corpus = corpus.apply(self.stem)          # German stemmer
        if self.lemma: 
            if self.verbose: print("Applying lemmatization.") 
            corpus = corpus.apply(self.lemmatize)  # makes preprocessing very slow though
        corpus = corpus.apply(self.untokenize)
        if self.verbose: print("Finished preprocessing!")
        return corpus #.to_frame(name="text") 
