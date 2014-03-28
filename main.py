__author__ = 'mpopovic'

import sqlite3 as lite
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import pandas.io.parsers as pp
import os

import scipy.interpolate as itp
from scipy import signal
from scipy import fftpack

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


## REMOVING OUTLIERS

for m in movies.values():
    x = np.array(m.data['time'])
    y = np.array(m.data['elong_xx'])
    x_old = x
    y_old = y
    elon_deriv = (y[1:]-y[:-1])/(x[1:]-x[:-1])
    t=20
    while t < len(x)-30:
        local_elon_deriv_mean = elon_deriv[t-30:t+30].mean()
        local_elon_deriv_stddev = elon_deriv[t-30:t+30].std()
        if ((y[t+1]-y[t])/(x[t+1]-x[t]) > local_elon_deriv_mean + local_elon_deriv_stddev) or ((y[t+1]-y[t])/(x[t+1]-x[t]) < local_elon_deriv_mean - local_elon_deriv_stddev):
            x = np.delete(x,t+1)
            y = np.delete(y,t+1)
            print 'DELETING FRAME: ', t
        else:
            t += 1
    m.itp_elon = itp.interp1d(x,y)

plt.figure()
for m in [movies[x] for x in WT_movies]:
    x = np.arange(0,m.data['time'].max(), 1000)
    y = m.itp_elon(x)
    plt.plot(x,y, label = m.name)
plt.legend(loc = 'best')
plt.show()

##ALIGNING MAXIMA
for m in movies.values():
    max_loc = np.argmax(m.data['elong_xx'])
    m.shift = m.data['time'][max_loc]-m.data['time'][0]
    m.data['shifted_time_align_max'] = m.data['time'] - m.data['time'][max_loc]

MT_movies
HT_movies[2]
WT_cold_movies = ['WT_25deg_111103.sqlite', 'WT_25deg_120531.sqlite', 'WT_25deg_111102.sqlite']
WT_hot_movies = ['WT_25-30deg_130926.sqlite', 'WT_25-30deg_130921.sqlite']
MT_hot_movies = ['MTcdc2_25-30deg_130917.sqlite', 'MTcdc2_25-30deg_130919.sqlite', 'MTcdc2_25-30deg_130916.sqlite']
MT_cold_movies =['MTcdc2_25deg_130905.sqlite']

plt.figure()
# for m in [movies[x] for x in WT_cold_movies]:
#     x = m.data['shifted_time_align_max']/3600.
#     y = m.data['elong_xx']/m.data['elong_xx'].max()
#     plt.plot(x,y, label = m.name, c='blue')
for m in [movies[x] for x in WT_hot_movies]:
    x = m.data['shifted_time_align_max']/3600.
    y = m.data['elong_xx']/m.data['elong_xx'].max()
    plt.plot(x,y, label = m.name, c = 'blue')
# for m in [movies[x] for x in MT_hot_movies]:
#     x = m.data['shifted_time_align_max']/3600.
#     y = m.data['elong_xx']/m.data['elong_xx'].max()
#     plt.plot(x,y, label = m.name, c='blue')
for m in [movies[x] for x in HT_movies]:
    x = m.data['shifted_time_align_max']/3600.
    y = m.data['elong_xx']/m.data['elong_xx'].max()
    plt.plot(x,y, label = m.name, c='red')
# m = movies[WT_hot_movies[1]]
# x = m.data['shifted_time_align_max']/3600.
# y = m.data['elong_xx']/m.data['elong_xx'].max()
# plt.plot(x,y, label = m.name, c='red')
# m = movies[HT_movies[0]]
# x = m.data['shifted_time_align_max']/3600.
# y = m.data['elong_xx']/m.data['elong_xx'].max()
# plt.plot(x,y, label = m.name, c='green')
# m = movies[HT_movies[2]]
# x = m.data['shifted_time_align_max']/3600.
# y = m.data['elong_xx']/m.data['elong_xx'].max()
# plt.plot(x,y, label = m.name, c='green')
# for m in [movies[x] for x in MT_cold_movies]:
#     x = m.data['shifted_time_align_max']/3600.#*0.85
#     y = m.data['elong_xx']/m.data['elong_xx'].max()
#     plt.plot(x,y, label = m.name, c='green')
plt.legend(loc = 'best')
plt.xlabel('time[h]')
plt.ylabel('Q_1')
plt.savefig('figures/WTHT_hot.png')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as it

## READING EXPERIMENTAL DATA
e_time, e_shear, e_elon = [], [], []
inp = open('/home/mpopovic/Documents/Work/Projects/drosophila_wing_analysis/111102/blade+hinge/triangleState.dat', 'r')
inp.readline()
for line in inp.readlines():
    dat = line.rstrip().split()
    e_time.append(float(dat[0]))
    e_elon.append(float(dat[3]))
inp.close()
e_time = [x-16.0 for x in e_time]
max_index = np.argmax(e_elon)
e_time = np.array(e_time) - e_time[0]
plt.figure()
plt.plot(e_time, e_elon)
plt.show()

master = movies['WT_25deg_111102.sqlite']
slave = movies['WT_25deg_111103.sqlite']

x_max = np.min([master.data['time'].max(), slave.data['time'].max()])
x_range = np.arange(0, x_max, 100)
y_master = master.itp_elon(x_range)
y_slave = slave.itp_elon(x_range)

test1 = np.ones(10)
test2 = np.ones(10)
norm = np.arange(len(test1))+1
norm =np.append(norm, np.arange(len(test1)-1,0,-1))
signal.correlate(test1, test2)/norm

norm = np.arange(len(y_master))+1
norm =np.append(norm, np.arange(len(y_master)-1,0,-1))
shft = (np.argmax(signal.correlate(y_master, y_slave)/norm)-len(y_slave))*100

x_corr = np.arange(-150, 150, 1)
y_corr_master = (y_master- y_master.mean())/y_master.std()
y_corr_slave = (y_slave- y_slave.mean())/y_slave.std()
len(signal.correlate(y_master, y_slave))
len(np.arange(-len(y_master), len(y_master)-1,1))
plt.figure()
plt.plot(np.arange(-len(y_master), len(y_master)-1,1), signal.correlate(y_master, y_slave)/norm)
plt.legend(loc= 'best')
plt.show()

plt.figure()
plt.plot(x_range, y_master, x_range , y_slave)
plt.legend(loc = 'best')
plt.show()

plt.figure()
for m in [movies[x] for x in hot_movies]:
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


import numpy as np

import rpy2.robjects.numpy2ri
from rpy2 import robjects
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr

# Set up our R namespaces
R = rpy2.robjects.r

DTW = importr('dtw')

x_max = np.min([master.data['time'].max(), slave.data['time'].max()])
x_range = np.arange(0, x_max, 100)
y_master = master.itp_elon(x_range)
y_slave = slave.itp_elon(x_range)
y_master_cs = y_master.cumsum()
y_slave_cs = y_slave.cumsum()
y_master_deriv =(y_master[1:]-y_master[:-1])/100.
y_slave_deriv =(y_slave[1:]-y_slave[:-1])/100.
