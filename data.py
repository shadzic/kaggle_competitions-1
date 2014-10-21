# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 21:18:08 2014

@author: ivanux
"""

import os
import platform 
import pandas as pd
import pylab as pl


DELIMITER = ','


class CsvData: 
    """
    Class representing csv data.
    """
    
    def __init__(self, path = None):
        if path is None:
            try:
                self.path = self._default_path()
            except IOError:
                pass
        else:
            self.path = path
        self.X , self.y = self._read_csv()        
    
    
    def _default_path(self):
        """
        Returns the path of the csv file storing our data. 
        By default, the train.csv file is stored in the same 
        folder as the data.py file.
        """
        if os.path.exists(os.path.join(os.getcwd(), 'train.csv')):
            return os.path.join(os.getcwd(), 'train.csv')
        else:
            raise IOError("file not found")

    
    
    def _read_csv(self):
        """
        Read our .csv file storing our data and set X and y
        """
        self.dataframe = pd.read_csv(self.path, delimiter = DELIMITER).dropna()
        n, p = self.dataframe.shape
        return self.dataframe.iloc[:,1:p], self.dataframe['label'] 
        


class Digits(CsvData):
    """
    Class representing our digits data. 
    """    
    
    def __init__(self, path = None): 
        CsvData.__init__(self, path)
    
    
    def plot_some_digits(self, some = 9):
        """
        Plots n random digits among the data. 
        """
        nine_first_digit = self.X.iloc[0:some,:]
        for i in range(nine_first_digit.shape[0]):  
            pl.subplot(3, 3, i + 1)
            digit = nine_first_digit.iloc[i,:].values.reshape((28,28))
            pl.imshow(digit, cmap=pl.cm.summer, interpolation='none') 
        
        
        
        
        
        
        
        
        
        
        
                