import numpy as np
import tensorflow as tf
import itertools
from collections import Counter
import csv

##Build vocabulary

def my_load_data_and_labels():
    """
    Costum load routine :D
    """
    texts = list(open('tekst_out2.csv').readlines())
    texts = [s.strip() for s in texts]
    x_text = texts
    codes = list(open('code_out2.csv').readlines())
    codes = [s.strip() for s in codes]
    y = codes
    return [x_text, y]



    print(x_text)

my_load_data_and_labels()