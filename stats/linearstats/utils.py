import numpy as np
import scipy.stats as st
import pandas as pd

def minmax_scale(data):
    max = np.max(data)
    min = np.min(data)
    scaled = np.array([(x - min) / (max - min) for x in data]) 
    return scaled

def residuals(data):
    return np.array(data - int(np.mean(data)))

pddata = pd.read_csv('linear.csv')
data = pddata.to_numpy()

scaled = minmax_scale(data)
np.savetxt('scaled.csv', scaled, delimiter=',', fmt='%d')

res = residuals(data)
np.savetxt('res.csv', res, delimiter=',', fmt='%d')


off1st = pd.read_csv('../singlestats/dal_off1st_1621.csv')
off1st = off1st.to_numpy()
off1st = minmax_scale(off1st)

offpass = pd.read_csv('../singlestats/dal_offpass_1621.csv')
offpass = offpass.to_numpy()
offpass = minmax_scale(offpass)

offrush = pd.read_csv('../singlestats/dal_offrush_1621.csv')
offrush = offrush.to_numpy()
offrush = minmax_scale(offrush)

scaled = np.concatenate((off1st, offpass,offrush), axis=0)

np.savetxt('scaled.csv', scaled, delimiter=',')
