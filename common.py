#where I keep all the constants
from pathlib import Path
import itertools
import numpy as np

DUMMY = "dummy"
DATA_DIR = Path("data").absolute()

start_of_sea_channel_CANDAS_GC = 415
start_of_sea_channel_CANDAS_TF = 745
start_of_sea_channel_CASTOR = 1082
start_of_sea_channel_SAFE = 236 #(or 239 -> 233??) #Turns out all Aragon Photonics interrogators start at meter 60


## Picked bad channels below:

#CANDAS1 poster event
bad_channels_candas1_2nd_picked_ch_4426 = np.loadtxt('channel_selections/quality_picks_candas1_picked_0724_1805_ch_4426-5988.csv',delimiter=',',skiprows=1)
bad_channels_candas1_2nd_picked_ch_4426 = bad_channels_candas1_2nd_picked_ch_4426.astype(int)
bad_channels_candas1_2nd_picked_ch_4426.T[0] = bad_channels_candas1_2nd_picked_ch_4426.T[0] + 4426
bad_channels_candas1_2nd_picked_ch_2701 = np.loadtxt('channel_selections/quality_picks_candas1_0815_picked _jul24_ch_2701-4425.csv',delimiter=',',skiprows=1)
bad_channels_candas1_2nd_picked_ch_2701 = bad_channels_candas1_2nd_picked_ch_2701.astype(int)
bad_channels_candas1_2nd_picked_ch_2701.T[0] = bad_channels_candas1_2nd_picked_ch_2701.T[0] + 2701
bad_channels_candas1_2nd_picked_ch_2701 = bad_channels_candas1_2nd_picked_ch_2701[:-1563]
bad_channels_candas1_2nd_picked_ch_415 = np.loadtxt('channel_selections/quality_picks_candas1_0815_picked_jul_23_ch_415-2700.csv', delimiter=',', skiprows=1)
bad_channels_candas1_2nd_picked_ch_415 = bad_channels_candas1_2nd_picked_ch_415.astype(int)
bad_channels_candas1_2nd_picked_ch_415.T[0] = bad_channels_candas1_2nd_picked_ch_415.T[0] + 415
bad_channels_candas1_2nd_picked = np.concatenate((bad_channels_candas1_2nd_picked_ch_415,bad_channels_candas1_2nd_picked_ch_2701,bad_channels_candas1_2nd_picked_ch_4426), axis=0)

#CANDAS2 poster event
bad_channels_candas2 = np.loadtxt('channel_selections/quality_picks_candas2_picked_0730_1940.csv',delimiter=',',skiprows=1)
bad_channels_candas2 = bad_channels_candas2.astype(int)
bad_channels_candas2.T[0] = bad_channels_candas2.T[0] + start_of_sea_channel_CANDAS_GC
candas2_bad_channels = bad_channels_candas2[np.where(bad_channels_candas2.T[1] == 1)].T[0]

#Twin event 08/22 17:13 Gran Canaria
bad_channels_twin_gc = np.loadtxt('channel_selections/quality_picks_gc_twin_event_G20200822_1713_picked_0802_1839.csv',delimiter=',',skiprows=1)
bad_channels_twin_gc = bad_channels_twin_gc.astype(int)
bad_channels_twin_gc.T[0] = bad_channels_twin_gc.T[0] + start_of_sea_channel_CANDAS_GC
twin_gc_bad_channels = bad_channels_twin_gc[np.where(bad_channels_twin_gc.T[1] == 1)].T[0]

#Twin event 08/22 17:13 Tenerife
bad_channels_twin_tf = np.loadtxt('channel_selections/quality_picks_tf_twin_event_T20200822_1713_picked_0807_0337.csv',delimiter=',',skiprows=1)
bad_channels_twin_tf = bad_channels_twin_tf.astype(int)
bad_channels_twin_tf.T[0] = bad_channels_twin_tf.T[0] + start_of_sea_channel_CANDAS_TF
twin_tf_bad_channels = bad_channels_twin_tf[np.where(bad_channels_twin_tf.T[1] == 1)].T[0]

#Tenerife event 07/27 20:44 picked July 17th, just one of a twin
bad_channels_tf_1st_pick = np.loadtxt('channel_selections/quality_labels_0727_2044_pickedjul17.txt')
bad_channels_tf_1st_pick = bad_channels_tf_1st_pick.astype(int)

#CASTOR 07/20 event (Specifically 115624!!)
bad_channels_castor = np.loadtxt('channel_selections/quality_picks_castor2_20230720_picked_0813_2235.csv',delimiter=',',skiprows=1)
bad_channels_castor = bad_channels_castor.astype(int)
bad_channels_castor.T[0] = bad_channels_castor.T[0] + start_of_sea_channel_CASTOR
castor_bad_channels = bad_channels_castor[np.where(bad_channels_castor.T[1] == 1)].T[0]

# bad_channels_CANDAS_08_15 = list(itertools.chain(range(415,421),
#                                     range(430,472),
#                                     range(509,517),
#                                     range(719,721),
#                                     range(1159,1166),
#                                     range(1231,1298),
#                                     [2575,2590,2594],
#                                     range(2992,3002),
#                                     range(3009,3061),
#                                     range(3071,3121),
#                                     range(3206,3208),
#                                     range(3235,3243),
#                                     range(3292,3297),
#                                     [3307],
#                                     range(3861,3868),
#                                     range(3935,3941),
#                                     [4124,4137,4138,4392,4393],
#                                     range(4446,4466),
#                                     range(5301,5308),
#                                     range(5646,5729),
#                                     range(5864,5873)
#                                     ))


# bad_channels_CANDAS2_12_27 = list(itertools.chain(range(415,610),
#                                     range(685, 727),
#                                     [904,911,912,928],
#                                     range(1075,1159),
#                                     range(1218,1238),
#                                     range(1252,1256),
#                                     range(1278,1289),
#                                     range(1361,1367),
#                                     [1417,1457,1548,1549],
#                                     range(1776,1864),
#                                     range(2011,2018),
#                                     range(2039,2042),
#                                     range(2063,2110),
#                                     range(2158,2171),
#                                     [2197,2198,2199],
#                                     range(2257,2313),
#                                     range(2321,2324),
#                                     range(2369,2374),
#                                     range(2379,2382),
#                                     range(2387,2389),
#                                     range(2403,2406),
#                                     #range(2415,2476),
#                                     #range(2488,2591),
#                                     range(2515,2591),
#                                     range(2609,2615),
#                                     [2639],
#                                     range(2740,2746),
#                                     [2758,2777,2778],
#                                     range(2984,3115),
#                                     [3158,3170,3171,3184,3185],
#                                     range(3191,3364),
#                                     range(3917,3973),
#                                     range(4073,4078),
#                                     range(4136,4173),
#                                     range(4263,4278),
#                                     range(5000,7007)
#                                     ))

# bad_channels_tf_1st_picked = np.loadtxt('channel_picks/quality_labels_0727_2044_pickedjul17.txt')

# bad_channels_candas1_2nd_picked = np.loadtxt('channel_picks/quality_picks_candas1_picked_0724_1805_ch_4426-5988.csv', delimiter=',',skiprows=1)

# bad_channels_gc_twin = np.loadtxt('channel_picks/quality_picks_gc_twin_event_G20200822_1713_picked_0802_1839.csv', delimiter=',', dtype =float, skiprows=1)

# bad_channels_tf_twin = np.loadtxt('channel_picks/quality_picks_tf_twin_event_T20200822_1713_picked_0807_0337.csv', delimiter=',', dtype =float, skiprows=1)