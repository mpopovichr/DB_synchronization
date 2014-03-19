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
    con = lite.connect(dbPath+dbName[:-7]+'/'+dbName)
    time_data = psql.frame_query('SELECT * FROM timepoints;', con)
    data = psql.frame_query('SELECT * FROM cells;', con)
    frame_list = pd.Series(data.frame).unique()
    avg_elong, avg_area, time = [], [], []
    for frame in frame_list:
        avg_elong.append(data[data.frame==frame]['elong_xx'].mean())
        time.append(time_data.frame[frame])
        avg_area.append(data[data.frame==frame]['area'].mean())
    df = pd.DataFrame(index=frame_list, columns = ['time', 'elong_xx','area'])
    df['time']  = time
    df['elong_xx'] = avg_elong
    df['area'] = avg_area
    movies[dbName].data = df
    con.close()

##CUMULATIVE ELONGATION
for m in movies.values():
    m.data['cum_elong_xx'] = m.data['elong_xx'].cumsum()

##SMOOTHING ELONG
for m in movies.values():
    kernel = np.ones(10)/10.
    m.data['smooth_elong_xx'] = np.convolve(m.data['elong_xx'], kernel, 'same')


WT_simple_movies = ['WT_25deg_111102.sqlite', 'WT_25deg_111103.sqlite', 'WT_25deg_120531.sqlite']
cold_movies = WT_simple_movies + ['MTcdc2_25deg_130905.sqlite']
hot_movies = ['HTcdc2_25-30deg_130927.sqlite', 'MTcdc2_25-30deg_130917.sqlite', 'MTcdc2_25-30deg_130919.sqlite', 'WT_25-30deg_130926.sqlite', 'HTcdc2_25-30deg_130925.sqlite', 'WT_25-30deg_130921.sqlite', 'HTcdc2_25-30deg_130924.sqlite', 'MTcdc2_25-30deg_130916.sqlite']
WT_movies = WT_simple_movies + ['WT_25-30deg_130926.sqlite', 'WT_25-30deg_130921.sqlite']
HT_movies = []
WT_movies, HT_movies, MT_movies = [],[],[]
for x in movies.keys():
    if x[:2] == 'WT':
        WT_movies.append(x)
    if x[:2] == 'HT':
        HT_movies.append(x)
    if x[:2] == 'MT':
        MT_movies.append(x)

# for m in [movies[x] for x in WT_simple_movies]:
# for m in [movies[x] for x in cold_movies]:
plt.figure()
for m in [movies[x] for x in WT_movies]:
    plt.plot(m.data['shifted_time'], m.data['elong_xx'], label =m.name)
plt.legend(loc = 4)
plt.show()

plt.figure()
# plt.plot(movies['WT_25deg_111102.sqlite'].data['elong_xx'], movies['WT_25deg_111103.sqlite'].data['elong_xx'][:-9])
plt.plot(movies['WT_25deg_111102.sqlite'].data['elong_xx'], movies['WT_25deg_120531.sqlite'].data['elong_xx'][:-9])
plt.show()

##ALIGNING TIME AND ELONGATION
for m in movies.values():
    max_frame = m.data['elong_xx'].argmax()
    max_t = m.data['time'][max_frame]
    max_elong = m.data['elong_xx'][max_frame]
    m.data['shifted_time'] = m.data['time'] - max_t
    m.data['shifted_elong_xx'] = m.data['elong_xx'] - max_elong




plt.figure()
for dbName in WT_movies:
    plt.plot([movies[dbName].time[x] for x in movies[dbName].frames], [movies[dbName].elong_xx[x] for x in movies[dbName].frames])
plt.show()
