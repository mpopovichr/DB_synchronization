__author__ = 'mpopovic'


import sqlite3 as lite
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import pandas.io.parsers as pp

import os

dbPath = '/data/biophys/etournay/DB/'

f = []
for (dirpath, dirnames, filenames) in os.walk(dbPath):
    f.extend(filenames)
f.sort()

dbName = f[5]
savePath= './data/'+dbName[:-7]+'/'


con = lite.connect(dbPath+dbName[:-7]+'/'+dbName)

data = psql.frame_query('SELECT * FROM cells;', con)

