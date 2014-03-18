__author__ = 'mpopovic'


import sqlite3 as lite
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import pandas.io.parsers as pp
import os

class Movie():
    def __init__(self, name):
        self.name = name


dbPath = '/data/biophys/etournay/DB/'

f = []
for (dirpath, dirnames, filenames) in os.walk(dbPath):
    f.extend(filenames)
f = list(set(f))

movies = {}
for dbName in f:
    print dbName
    movies[dbName] = Movie(dbName)
    movies[dbName].data = pd.DataFrame(columns=('time', 'elong_xx'))
    con = lite.connect(dbPath+dbName[:-7]+'/'+dbName)
    time_data = psql.frame_query('SELECT * FROM timepoints;', con)
    data = psql.frame_query('SELECT * FROM cells;', con)
    frame_list = pd.Series(data.frame).unique()
    for frame in frame_list:
        avg_elong = data[data.frame==frame]['elong_xx'].mean()
        time = time_data.frame[frame]
    df = pd.DataFrame([[]])


##SMOOTHING ELONG
for m in movies:
    kernel = np.ones(10)/10.
    m.smooth_elong_xx = np.convolve(m.elong_xx)

##ALIGNING TIME
for dbName in f:
    max_t = 0
    max_elong = 0
    for frm in movies[dbName].frames:
        if movies[dbName].elong_xx[frm] > max_elong:
            max_elong = movies[dbName].elong_xx[frm]
            max_t = movies[dbName].time[frm]
    for frm in movies[dbName].frames:
        movies[dbName].time[frm] -= max_t
        movies[dbName].elong_xx[frm] -= max_elong


WT_movies = ['WT_25deg_111102.sqlite', 'WT_25deg_111103.sqlite', 'WT_25deg_120531.sqlite']
plt.figure()
for dbName in WT_movies:
    plt.plot([movies[dbName].time[x] for x in movies[dbName].frames], [movies[dbName].elong_xx[x] for x in movies[dbName].frames])
plt.show()
