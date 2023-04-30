import numpy as np
import pandas as pd
import pdfplumber
from typing import (Dict, List, Text, Optional, Any, Union, Tuple)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from copy import deepcopy
from pdf_extract.config import config  
from pdf_extract.services import file 


def set_formats(X: pd.DataFrame, 
                format_schema_map : Dict = {'categorical_cols' : object, 'text_cols' : str, 
                                            'numeric_cols': np.float32, 'date_cols': np.datetime64}, 
                verbose : bool = False)-> pd.DataFrame:

    """Set custom data formats as specified in input_output.yaml

    Args:
        X (pd.DataFrame): Raw input data
        format_schema_map (dictionary, optional): see input_output.yaml. Defaults to {'categorical_cols' : object, 'text_cols' : str, 'numeric_cols': np.float32, 'date_cols': np.datetime64}.
        verbose (bool, optional): print user info. Defaults to False.

    Returns:
        pd.DataFrame: newly formatted data
    
    # Run simple test:
    >>> df = set_formats(X = pd.DataFrame(['...some test']), verbose = True) 
    Data formats have been set.
    """
    schema_map = config.io['input']['schema_map']               # get schema mapping from i/o config

    for format_type, obj_type in format_schema_map.items(): 
        if format_type not in list(schema_map.keys()):
            continue

    if verbose: print("Data formats have been set.")  
    return X 


def train_test_split_extend(X: pd.DataFrame, y: Optional[pd.DataFrame]=None, test_size : List=[0.2, 0.1], **para)-> Tuple:
    """Split dataset (X,y) into train, test, validation set.

    Args:
        X (pd.DataFrame): Design matrix with features in columns
        y (Optional[pd.DataFrame], optional): In case of supervised learning task. Defaults to None.
        test_size (List, optional): Proportion of test set size and optionally validation set. Defaults to [].

    Returns:
        tuple: X,y dataframes according to folds
    """
    assert len(test_size) in [1,2], 'Specify test set size and optionally followed by validation set size.'

    if y is None: 
        if len(test_size) == 2:
          p_rest = round(sum(test_size),2) ; p_test = round(test_size[0]/p_rest,2) 
          X_train, X_rest = train_test_split(X, test_size = p_rest, **para)
          X_valid, X_test = train_test_split(X_rest, test_size=p_test, **para)
          assert (X.shape[0] == X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])
          return X_train, X_valid, X_test

        if len(test_size) == 1:
          p_test = test_size[0]
          X_train, X_test = train_test_split(X, test_size = p_test, **para)
          assert (X.shape[0] == X_train.shape[0] + X_test.shape[0])
          return X_train, X_test
    else:    
        if len(test_size) == 2:
          p_rest = round(sum(test_size),2) ; p_test = round(test_size[0]/p_rest,2)
          X_train, X_rest, y_train, y_rest = train_test_split(X, y, stratify = y, test_size = p_rest, **para)
          X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, stratify = y_rest, test_size=p_test, **para)
          assert (X.shape[0] == X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])
          return X_train, X_valid, X_test, y_train, y_valid, y_test

        if len(test_size) == 1:
          p_test = test_size[0]   
          X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = p_test, **para)
          assert (X.shape[0] == X_train.shape[0] + X_test.shape[0])
          return X_train, X_test, y_train, y_test
        

def extract_pdf_data(feed)-> Tuple[str, str]:
    """Extract data from pdf
    Args:
        feed (_type_): _description_
    Returns:
        string: raw text data
    """
    i, page_objects, text = 0, {}, ""
    with pdfplumber.open(feed) as pdf:
        while i < len(pdf.pages):
            page = pdf.pages[i]
            page_objects[str(i+1)] = page.extract_text(x_tolerance=1, y_tolerance=3) #.split('\n')
            text += page_objects[str(i+1)]
            i += 1
    return text, pdf.stream.name



class naiveBayes(BaseEstimator, ClassifierMixin):
    
    def __init__(self, eps = 1e-7, verbose = True, alpha = 1, beta0 = 1, beta1 = 1):
        self.verbose = verbose
        self.eps = eps     # not needed if you use bayesian version -> Laplace smooting
        self.alpha = alpha
        if verbose : print('-- Bernoulli Naive Bayes (via ML estimation)--')

    def logsumexp(self, x):
        """Log-Sum-Exp Trick"""
        c = x.max()
        return c + np.log(np.sum(np.exp(x - c)))
            
        
    def fit(self, X: np.array, y : np.array):
            """
            Input:
                corpus: a list of training documents
                labels: an m x 1 array with label of each document (e.g. claim descr.)
            
            # Example:
            >>> corpus = np.array([['I', 'like', 'cats'], ['I', 'like', 'dogs'], ['What', 'do', 'you'], ['I', 'like', 'cats']])
            >>> labels = np.array([2,6, 2, 2])
            >>> nb = utils.naiveBayes()
            -- Bernoulli Naive Bayes (via ML estimation)--
            >>> class_prior_prob, class_cond_prob = nb.fit(corpus, labels)
            Creating (token, label) -> frequency mappings for documents in corpus...
            >>> class_cond_prob
                        2    6
            you   0.333333  0.0
            What  0.333333  0.0
            dogs  0.000000  1.0
            do    0.333333  0.0
            like  0.666667  1.0
            cats  0.666667  0.0
            I     0.666667  1.0
            """
            corpus = deepcopy(X) ; labels = deepcopy(y)
            if self.verbose : print('Creating (token, label) -> frequency mappings for documents in corpus...')        
            N = corpus.shape[0] ; self.vocab, tokenized_corpus = set(),[]
            
            # Create vocab.:
            for i in corpus.tolist(): self.vocab.update(i)     # set will ignore existing tokens
            # Calculate class prior probab.    
            class_vals, self.N_c = np.unique(labels, return_counts=True)
            self.joint_distr = pd.DataFrame(0,index=list(self.vocab), columns=[str(i) for i in set(labels)])
            self.alphas = np.ones(len(class_vals))*self.alpha     # Dirichlet prior para
            alpha0 = sum(self.alphas)
            for z, (y, sentence) in enumerate(zip(labels, corpus)):
                #tokens = word_tokenize(sentence)
                #tokens = sentence.tolist()[0]
                #for word in tokens:
                for word in sentence:
                    self.joint_distr.loc[str(word),str(y)] += 1

            class_prior_prob = self.N_c / N        
            class_cond_prob = self.joint_distr.values / self.N_c
            class_prior_prob = pd.Series(class_prior_prob + self.eps, index=[str(i) for i in class_vals])
            class_cond_prob = pd.DataFrame(class_cond_prob, index=self.joint_distr.index, columns=self.joint_distr.columns)
            self.log_pd_theta = np.log(class_cond_prob + self.eps)
            self.log_pd_theta_compl = np.log(1 - class_cond_prob + self.eps)
            self.log_class_prior_prob = np.log(class_prior_prob)
            return class_prior_prob, class_cond_prob

        
    def predict(self, X: np.array): 
            """
            Predict: Calculate posterior predictive distribution and calculate MAP estimate
            see for example: Murphy, ML-a probab. perspective, eqn (3.66)
            """
            lpost = np.zeros((X.shape[0], len(self.log_pd_theta.columns)))
            corpus = deepcopy(X)
            for i, sentence in enumerate(corpus):
                for z, c in enumerate(self.log_pd_theta.columns):
                    lp = self.log_class_prior_prob[str(c)]
                    for j in self.vocab:
                        if j in sentence: 
                            lp = lp + self.log_pd_theta.loc[str(j), str(c)]
                        else:
                            lp = lp + self.log_pd_theta_compl.loc[str(j), str(c)]
                    lpost[i,z] = lp

            post = np.exp(lpost - self.logsumexp(lpost))
            yhat = np.argmax(lpost, axis=1)
            return lpost, yhat


class make_nb_feat(BaseEstimator, TransformerMixin):
  """
  Create Naive Bayes like document embeddings
  """
  def __init__(self, verbose : bool = True, **vect_param):
      """
      Example:
      #--------
      >>> corpus = np.array(['I like cats', 'I like dogs', 'What do you like?', 'This is fun!!!'])
      >>> y = np.array([2,6, 2, 2])
      >>> nb = utils.make_nb_feat().fit_transform(corpus,y)
      -- Creating Naive Bayes like document embeddings --
      >>> nb
           level2    level6
      0 -1.021651 -1.504077
      1 -2.120264 -0.810930
      2 -2.748872 -3.295837
      3 -1.021651 -1.504077
      """  
      self.verbose = verbose  
      if self.verbose : print('-- Creating Naive Bayes like document embeddings --')  
            
      self.pipeline = Pipeline([
               #('cleaner', utils.clean_text(verbose=False)),
               ('vectorizer', CountVectorizer(lowercase=True, #ngram_range=(2, 2),
                                   token_pattern = '(?u)(?:(?!\d)\w)+\\w+', 
                                    analyzer = 'word',  #char_wb
                                    tokenizer = None, 
                                    stop_words = None, #"english"
                                    **vect_param          
                                    )),  
               ('model', BernoulliNB(alpha=1))
            ])

  def fit(self, X, y):
        
      self.pipeline.fit(X, y)
      self.pipeline.named_steps['vectorizer'].get_stop_words()
      self.vocab_ = self.pipeline.named_steps['vectorizer'].get_feature_names_out()
      self.vectorizer = self.pipeline.named_steps['vectorizer']
      self.model = self.pipeline.named_steps['model']
      dt = self.vectorizer.transform(X)   # train set
      self.doc_term_mat_train = dt.toarray()
      self.log_cond_distr_train = pd.DataFrame(self.model.feature_log_prob_, index=[str(i) for i in self.model.classes_], columns=self.vocab_)
      self.joint_abs_freq_train = pd.DataFrame(self.model.feature_count_, index=[str(i) for i in self.model.classes_], columns=self.vocab_)
      return self

  def transform(self, X):

      dt = self.vectorizer.transform(X)
      self.doc_term_mat = dt.toarray()
      features_class = pd.DataFrame()
      for c in self.model.classes_:
            # log class cond. prob
            feat_c = np.sum(self.doc_term_mat * self.log_cond_distr_train.loc[str(c),:].values, axis = 1)   # broadcast
            # Joint abs. freq
            #feat_c = np.sum(self.doc_term_mat * self.joint_abs_freq_train.loc[str(c),:].values, axis = 1)   # broadcast
            features_class['level'+str(c)] = feat_c
      return features_class 


if __name__ == "__main__":
    import doctest
    doctest.testmod()