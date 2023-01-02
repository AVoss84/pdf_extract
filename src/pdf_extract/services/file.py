"""
Services for reading and writing from and to various file formats
"""
import pandas as pd
from imp import reload
import os, yaml, json, toml, pickle
from typing import (Dict, List, Text, Optional, Any, Union)
from pdf_extract.config import global_config as glob

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
        self.verbose = verbose
        self.encoding = encoding
        self.schema_map = schema_map

    def doRead(self, **kwargs)-> pd.DataFrame:
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
    def __init__(self, path : Optional[str] = "", root_path : str = glob.UC_DATA_DIR, schema_map : Optional[dict] = None, is_df : bool = True, verbose : bool = True):
        """Generic read/write service for Pkl files
        Args:
            path (str, optional): Filename. Defaults to "".
            is_df (bool): Is pandas dataframe or not?
            root_path (str, optional): root path where file is located. Defaults to glob.UC_DATA_DIR.
            schema_map (Optional[dict], optional): mapping scheme for renaming of columns, see pandas rename. Defaults to None.
            verbose (bool, optional): should user information be displayed?. Defaults to True.
        """
        self.path = os.path.join(root_path, path)
        self.schema_map = schema_map
        self.verbose=verbose
        self.is_df = is_df
    
    def doRead(self, **kwargs)->pd.DataFrame:
        """Read pkl files
        Returns:
            pd.DataFrame: input data
        """
        try:
            if self.is_df:
                data = pd.read_pickle(self.path, **kwargs)
                if self.schema_map: self.df.rename(columns = self.schema_map, inplace = True)
            else:
                data = pickle.load(open(self.path, "rb"))
            if self.verbose : print(f"Pickle Service Read from file: {str(self.path)}")
            return data
        except Exception as e:
            print(e)

    def doWrite(self, X):
        """Write to PKL file
        Args:
            X (pd.DataFrame): input data
        """
        try:
            if self.is_df:
                X.to_pickle(path = self.path, compression = None)        # "gzip"
            else:
                pickle.dump(X, open(self.path, "wb"))
            if self.verbose : print(f"Pickle Service Read from file: {str(self.path)}")
        except Exception as e:
            print(e)


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
        
        def doWrite(self, X: pd.DataFrame, **kwargs):
            """Write dictionary X to YAMl file
            Args:
                X (pd.DataFrame): Input data
            """
            with open(self.path, 'w') as outfile:
                try:
                    yaml.dump(X, outfile, default_flow_style=False)
                    if self.verbose: print(f"Write to: {self.path}")
                except yaml.YAMLError as exc:
                    print(exc)


class TXTService:
    def __init__(self, path : Optional[str] = "", root_path : Optional[str] = glob.UC_DATA_DIR, verbose : bool = True):
        """Generic read/write service for TXT-files
        Args:
            path (Optional[str], optional): Filename. Defaults to "".
            root_path (Optional[str], optional): root path where file is located. Defaults to glob.UC_DATA_DIR.
            verbose (bool, optional): should user information be displayed?. Defaults to True.
        """
        self.path = os.path.join(root_path, path)
        self.verbose = verbose

    def doRead(self, **kwargs)-> List:
        """Read TXT files
        Returns:
            List: Input data
        """
        try:
            with open(self.path, **kwargs) as f:
                df = f.read().splitlines()
                #df = pd.read_csv(self.path, sep=" ", header=None, encoding = self.encoding, **kwargs)
            if self.verbose : print(f"TXT Service read from file: {str(self.path)}")    
        except Exception as e0:
            print(e0); df = None
        finally: 
            return df
        
    def doWrite(self, X : pd.DataFrame, **kwargs):
        """Write to TXT files.
        Args:
            X (List): Input data
        """
        try:
            #with open(self.path, 'w', **kwargs) as f:
            #    f.write('\n'.join(X))
            X.to_csv(self.path, index=None, sep="\t", header=None, mode='w+', **kwargs)   # sep=" "
            if self.verbose : print(f"TXT Service output to file: {str(self.path)}")  
        except Exception as e0:
            print(e0)


class JSONservice:
        def __init__(self, path : Optional[str] = "", root_path : str = '', verbose = True):
            
            self.path = os.path.join(root_path, path)
            self.verbose = verbose
        
        def doRead(self, **kwargs)-> dict:  
            """Read in JSON file from specified path
            Returns:
                dict: Output imported data
            """
            if os.stat(self.path).st_size == 0:         # if json not empty
                return dict()
            try:
                with open(self.path, 'r') as stream:
                    my_json_load = json.load(stream, **kwargs)                    
                if self.verbose: print(f'Read: {self.path}')
                return my_json_load    
            except Exception as exc:
                print(exc) 
            
        def doWrite(self, X: dict, **kwargs):
            """Write X to JSON file
            Args:
                X (dict): Input data
            """
            with open(self.path, 'w', encoding='utf-8') as outfile:
                try:
                    json.dump(X, outfile, ensure_ascii=False, indent=4, **kwargs)
                    if self.verbose: print(f'Write to: {self.path}')
                except Exception as exc:
                    print(exc) 
                   
class TOMLservice:
        def __init__(self, path : Optional[str] = "", root_path : str = glob.UC_CODE_DIR, 
                     verbose : bool = False):
            """Generic read/write service for TOML files.
            Args:
                path (str, optional): _description_. Defaults to "".
                root_path (str, optional): _description_. Defaults to glob.UC_CODE_DIR.
                verbose (bool, optional): _description_. Defaults to False.
            """
            self.root_path = root_path
            self.path = path
            self.verbose = verbose 

        def doRead(self, **kwargs)-> dict:  
            """Read from toml file.
            Returns:
                Dict: Imported toml file
            """
            with open(os.path.join(self.root_path, self.path), 'r') as stream:
                try:
                    toml_load = toml.load(stream, **kwargs)   
                    if self.verbose: print(f"Read: {self.root_path+self.path}")
                except Exception as exc:
                    print(exc) 
            return toml_load

        def doWrite(self, X: dict, **kwargs)-> bool:
            """Write dictionary X to TOML file.
            Args:
                X (Dict): Input dictionary
            """
            with open(os.path.join(self.root_path, self.path), 'w') as outfile:
                try:
                    toml.dump(X, outfile)
                    if self.verbose: print(f"Write to: {self.root_path+self.path}")
                except Exception as exc:
                    print(exc)