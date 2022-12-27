"""
Services for reading and writing from and to various file formats
"""

import pandas as pd
from imp import reload
import os, yaml
from typing import (Dict, List, Text, Optional, Any, Union)
from my_package.config import global_config as glob

class CSVService:
    def __init__(self, path : Optional[str] = "", delimiter : str = "\t", encoding : str = "UTF-8", schema_map : Optional[dict] = None, 
                 root_path : str = glob.UC_DATA_DIR, verbose : bool = False):
        """Generic read/write service for CSV files

        Args:
            path (str, optional): Filename. Defaults to "".
            delimiter (str, optional): see pd.read_csv. Defaults to "\t".
            encoding (str, optional): see pd.read_csv. Defaults to "UTF-8".
            schema_map (Optional[dict], optional): mapping scheme for renaming of columns, see pandas rename. Defaults to None.
            root_path (str, optional): root path where file is located. Defaults to glob.UC_DATA_DIR.
            verbose (bool, optional): should user information be displayed? Defaults to False.
        """
        self.path = os.path.join(root_path, path)
        self.delimiter = delimiter
        self.verbose=verbose
        self.encoding = encoding
        self.schema_map = schema_map

    def doRead(self, **kwargs) -> pd.DataFrame:
        """Read data from CSV

        Returns:
            pd.DataFrame: data converted to dataframe
        """
        df = pd.read_csv(filepath_or_buffer=self.path, encoding=self.encoding, delimiter=self.delimiter, **kwargs)
        if self.verbose: print(f"CSV Service Read from File: {str(self.path)}")
        if self.schema_map:
            df.rename(columns=self.schema_map, inplace=True)
        return df

    def doWrite(self, X : pd.DataFrame, **kwargs):
        """Write X to CSV file

        Args:
            X (pd.DataFrame): input data
        """
        X.to_csv(path_or_buf=self.path, encoding=self.encoding, sep=self.delimiter, **kwargs)
        if self.verbose: print(f"CSV Service Output to File: {str(self.path)}")


class XLSXService:
    def __init__(self, path : Optional[str] = "", sheetname : str = "Sheet1", root_path : str = glob.UC_DATA_DIR, schema_map : Optional[dict] = None, verbose : bool = False):
        """Generic read/write service for XLS files

        Args:
            path (str, optional): Filename. Defaults to "".
            sheetname (str, optional): see pd.read_excel. Defaults to "Sheet1".
            root_path (str, optional): root path where file is located_. Defaults to glob.UC_DATA_DIR.
            schema_map (Optional[dict], optional): mapping scheme for renaming of columns, see pandas rename. Defaults to None.
            verbose (bool, optional): should user information be displayed?. Defaults to False.
        """
        self.path = os.path.join(root_path, path)
        self.writer = pd.ExcelWriter(self.path)
        self.sheetname = sheetname
        self.verbose=verbose
        self.schema_map = schema_map
        
    def doRead(self, **kwargs) -> pd.DataFrame:
        """Read from XLS file

        Returns:
            pd.DataFrame: input data as dataframe
        """
        df = pd.read_excel(self.path, self.sheetname, **kwargs)
        if self.verbose: print(f"XLS Service Read from File: {str(self.path)}")
        if self.schema_map:
            df.rename(columns=self.schema_map, inplace=True)
        return df    
        
    def doWrite(self, X : pd.DataFrame, **kwargs):
        """Write to XLS file

        Args:
            X (pd.DataFrame): input data
        """
        X.to_excel(self.writer, self.sheetname, **kwargs)
        self.writer.save()
        if self.verbose: print(f"XLSX Service Output to File: {str(self.path)}")


class PickleService:
    def __init__(self, path : Optional[str] = "", root_path : str = glob.UC_DATA_DIR, schema_map : Optional[dict] = None, verbose : bool = False):
        """Generic read/write service for Pkl files

        Args:
            path (str, optional): Filename. Defaults to "".
            root_path (str, optional): root path where file is located. Defaults to glob.UC_DATA_DIR.
            schema_map (Optional[dict], optional): mapping scheme for renaming of columns, see pandas rename. Defaults to None.
            verbose (bool, optional): should user information be displayed?. Defaults to False.
        """
        self.path = os.path.join(root_path, path)
        self.schema_map = schema_map
        self.verbose=verbose

    def doRead(self, **kwargs)-> pd.DataFrame:
        """Read pkl files

        Returns:
            pd.DataFrame: input data
        """
        df = pd.read_pickle(self.path, **kwargs)
        if self.verbose : print(f"Pickle Service Read from file: {str(self.path)}")
        if self.schema_map: df.rename(columns = self.schema_map, inplace = True)
        return df

    def doWrite(self, X: pd.DataFrame, **kwargs)-> bool:
        """Write to PKL file

        Args:
            X (pd.DataFrame): input data

        Returns:
            bool: True if write process was successful and vice versa
        """
        try:
            X.to_pickle(path = self.path, compression = None)    
            if self.verbose : print(f"Pickle Service Output to File: {str(self.path)}")
            return True
        except Exception as e0:
            print(e0); return False        


class YAMLservice:
        def __init__(self, path : Optional[str] = "", root_path : str = glob.UC_CODE_DIR, verbose : bool = False):
            """Generic read/write service for YAML files

            Args:
                path (str, optional): Filename. Defaults to "".
                root_path (str, optional): root path where file is located. Defaults to glob.UC_CODE_DIR.
                verbose (bool, optional): should user information be displayed?. Defaults to False.
            """
            self.path = os.path.join(root_path, path)
            self.verbose = verbose 
        
        def doRead(self, **kwargs)-> Union[Dict, List]:  
            """Read from YAML file

            Returns:
                Union[Dict, List]: Read-in yaml file
            """
            with open(self.path, 'r') as stream:
                try:
                    my_yaml_load = yaml.load(stream, Loader=yaml.FullLoader, **kwargs)   
                    if self.verbose: print(f"Read: {self.path}")
                except yaml.YAMLError as exc:
                    print(exc) 
            return my_yaml_load
        
        def doWrite(self, X: pd.DataFrame, **kwargs)-> bool:
            """Write dictionary X to YAMl file

            Args:
                X (pd.DataFrame): Input data

            Returns:
                bool: True if write process was successful and vice versa
            """
            with open(self.path, 'w') as outfile:
                try:
                    yaml.dump(X, outfile, default_flow_style=False)
                    if self.verbose: print(f"Write to: {self.path}")
                    return True
                except yaml.YAMLError as exc:
                    print(exc); return False


class TXTService:
    def __init__(self, path : Optional[str] = "", encoding : str ="utf-8", root_path : Optional[str] = glob.UC_DATA_DIR, verbose : bool = True):
        """Generic read/write service for TXT-files

        Args:
            path (Optional[str], optional): Filename. Defaults to "".
            encoding (str, optional): see pd.read_csv. Defaults to "utf-8".
            root_path (Optional[str], optional): root path where file is located. Defaults to glob.UC_DATA_DIR.
            verbose (bool, optional): should user information be displayed?. Defaults to True.
        """
        self.path = os.path.join(root_path, path)
        self.encoding = encoding
        self.verbose = verbose

    def doRead(self, **kwargs) -> pd.DataFrame:
        """Read TXT files

        Returns:
            pd.DataFrame: Read input data
        """
        try:
            df = pd.read_csv(self.path, sep=" ", header=None, encoding = self.encoding, **kwargs)
            if self.verbose : print(f"TXT Service read from file: {str(self.path)}")    
        except Exception as e0:
            print(e0); df = None
        finally: 
            return df
        
    def doWrite(self, X : pd.DataFrame, **kwargs)-> bool:
        """Write to TXT files

        Args:
            X (pd.DataFrame): Input data

        Returns:
            bool: True if write process was successful and vice versa
        """
        try:
            X.to_csv(self.path, index=None, sep=' ', header=None, encoding = self.encoding, mode='w+', **kwargs)
            if self.verbose : print(f"TXT Service output to file: {str(self.path)}")  
            return True
        except Exception as e0:
            print(e0); return False

