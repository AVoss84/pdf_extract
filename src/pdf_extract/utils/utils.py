import numpy as np
import pandas as pd
import pdfplumber
from typing import (Dict, List, Text, Optional, Any, Union, Tuple)
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()