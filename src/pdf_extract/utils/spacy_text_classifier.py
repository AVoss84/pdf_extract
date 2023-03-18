from typing import Callable
import os, subprocess, spacy
from spacy.tokens import DocBin
from spacy.cli.train import train as train_model
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
#import configparser
from pdf_extract.config import global_config as glob
#from imp import reload

class SpacyClassifier(BaseEstimator, ClassifierMixin):
    """
    Train spaCy text classifier from scratch.
    """ 
    def __init__(self, verbose : bool = True):
        self.verbose = verbose
        self.nlp = spacy.load("de_core_news_lg") 
        if self.verbose: 
            print(f"Pretrained spaCy model loaded.")
    
    def __repr__(self):
            return f"SpacyClassifier(verbose = {self.verbose})"
    

    def fit(self, X: np.array = None, y : np.array = None, **params)-> 'SpacyClassifier':
        
        # In case config file has not been filled - do it now:
        #------------------------------------------------------
        if not hasattr(self, "config_file_name"): 
            self._fill_config_file()
        
        # Fit model based on config settings:
        #--------------------------------------
        try:
            train_model(
                config_path = f"{glob.UC_DATA_DIR}/{self.config_file_name}",
                output_path = f"{glob.UC_DATA_DIR}/output",
                overrides={"paths.train": f"{glob.UC_DATA_DIR}/train.spacy",
                           "paths.dev": f"{glob.UC_DATA_DIR}/valid.spacy"}, **params
                )
            if self.verbose: 
                print("\nTraining done!")
            
            # Finally use best model fit
            #----------------------------
            self.trained_nlp_ = spacy.load(f"{glob.UC_DATA_DIR}/output/model-best")    
        except Exception as ex:
            print(ex)
        return self

    
    def predict_proba(self, X: np.array)-> np.array:
        """Predict two-class posterior distribution

        Args:
            self (_type_): _description_
            float (_type_): _description_

        Returns:
            Class scores: _description_
        """
        query = X.tolist()
        scores = [self.trained_nlp_(text).cats for text in query]
        return pd.DataFrame(scores).values


    def predict(self, X: np.array)-> np.array:
        """Predict classes for query

        Args:
            X (pd.DataFrame): _description_

        Returns:
            np.array: _description_
        """
        return self.predict_proba(X).argmax(axis=1)


    def create_docs(self, X: np.array, y : np.array, target_file: str, save2disc : bool = True)-> Callable:
        """Create spacy compatible document object and save to disc

        Args:
            data (List[Tuple[str, str]]): _description_
            target_file (str): _description_
            cats (List[str]): _description_
            save2disc (bool, optional): _description_. Defaults to True.

        Returns:
            Callable: DocBin instance
        """
        classes_, _ = np.unique(y, return_inverse=True)
        data = list(zip(X, y))
        docs = DocBin()
        for doc, label in tqdm(self.nlp.pipe(data, as_tuples=True), total = len(data)):   
            for cat in classes_:
                doc.cats[str(cat)] = 1 if str(cat) == str(label) else 0
            #doc.cats["1"] = (str(label) == "1")*1
            docs.add(doc)
        if save2disc: 
            docs.to_disk(os.path.join(glob.UC_DATA_DIR, target_file))
            if self.verbose : 
                print(os.path.join(glob.UC_DATA_DIR, target_file))
        return docs


    def _fill_config_file(self, base_config_file_name : str = "base_config.cfg", config_file_name : str = "config.cfg"):
        """Fill train config file from base config file. 
           You can create a base config e.g. using spaCy's GUI: 
           https://spacy.io/usage/training#quickstart

        Args:
            base_config_file_name (str, optional): _description_. Defaults to "base_config.cfg".
            config_file_name (str, optional): _description_. Defaults to "config.cfg".
        """
        self.config_file_name = config_file_name
        if self.verbose: 
            print(f"Using {base_config_file_name} as base template.")

        cmd_init = 'python -m spacy init fill-config {}/{} {}/{}'.format(glob.UC_DATA_DIR, base_config_file_name, glob.UC_DATA_DIR, self.config_file_name)
        process = subprocess.Popen(cmd_init.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout_cmd, _ = process.communicate()
        if self.verbose : 
            print(stdout_cmd.decode("utf-8"))     # convert bytes to string for nicer printing 