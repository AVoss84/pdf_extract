import numpy as np
import pandas as pd
from typing import (Dict, List, Text, Optional, Any, Union)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
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


def train_test_split_extend(X, y: pd.DataFrame = None, test_size=[], **para):

    assert isinstance(test_size, list)
    assert isinstance(X, pd.DataFrame)

    if y is None: 
        if len(test_size) == 3:
          p_rest = round(1-test_size[0],2) ; p_test = test_size[2]/p_rest
          X_train, X_rest = train_test_split(X, test_size = p_rest, **para)
          X_valid, X_test = train_test_split(X_rest, test_size=p_test, **para)
          assert (X.shape[0] == X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])
          return X_train, X_valid, X_test

        if len(test_size) == 2:
          p_test = round(1-test_size[0],2) 
          X_train, X_test = train_test_split(X, test_size = p_test, **para)
          assert (X.shape[0] == X_train.shape[0] + X_test.shape[0])
          return X_train, X_test

    else:    
        #assert isinstance(y, pd.DataFrame)
        if len(test_size) == 3:
          p_rest = round(1-test_size[0],2) ; p_test = test_size[2]/p_rest
          X_train, X_rest, y_train, y_rest = train_test_split(X, y, stratify = y, test_size = p_rest, **para)
          X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, stratify = y_rest, test_size=p_test, **para)
          assert (X.shape[0] == X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])
          return X_train, X_valid, X_test, y_train, y_valid, y_test

        if len(test_size) == 2:
          p_test = round(1-test_size[0],2) ;
          X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = p_test, random_state=random_state,shuffle=shuffle)
          assert (X.shape[0] == X_train.shape[0] + X_test.shape[0])
          return X_train, X_test, y_train, y_test
        

if __name__ == "__main__":
    import doctest
    doctest.testmod()