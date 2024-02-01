# -*- coding: utf-8 -*-

from pathlib import Path
import pickle

def load_flowline_geom():
    fp = r"/Users/drotto/src/USGS/glacier-diseq/features/flowline_geom.pickle"
    with open(fp, 'rb') as f:
        f_geom = pickle.load(f)
        
    return f_geom