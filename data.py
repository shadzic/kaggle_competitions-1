# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 21:18:08 2014

@author: ivanux
"""

import os
import plateform 
import pandas as pd


class Data: 
    """
    Class representing our data.
    """
    
    def __init__(self, path = None):
        if path is None:
            try:
                self.path = self._default_path()
            except (OSError, IOError):
                pass
        else:
            self.path = path
        self.X , self.y = self._read_csv()        
    
    
    def _default_path(self):
        """
        Constructs the path of the csv file storing our data. By default, the train.csv file is stored 
        in the same folder as the data.py file.
        """
        if platform.system() == 'Linux':
            if os.path.exists(os.getcwd() +'/train.csv'):
                return os.getcwd()+'/train.csv'
            else:
                raise IOError("file not found")
        elif platform.system() == 'Windows':
            if os.path.exists(os.getcwd() +'/train.csv'):
                return os.getcwd() +'\\train.csv'
            else:
                raise IOError("file not found")
        else: 
            raise OSError("your OS is not supported")
    
    
    def _read_csv(self):
        """
        Read our .csv file storing our data and set X and y
        """
        self.dataframe = pd.read_csv(self.path, delimiter=',').dropna()
        n, p = self.dataframe.shape
        return self.dataframe.iloc[:,1:p], self.dataframe['label'] 
        


#class Digits(Data):
#    pass        
        
        
        
        
        
        
        
        
        
        
        
        
        
                