
# coding: utf-8

# In[1]:

import pandas as pd
from datetime import timedelta
import numpy as np
import time
import datetime
import math
import warnings
warnings.filterwarnings("ignore")
import sys
path= sys.argv[1]


# In[2]:

flow = pd.read_csv(path+'/flow.tsv', header=None, sep='\t')
prob = pd.read_csv(path+"/prob.tsv",sep="\t",header=None)


# In[3]:

nanDF = flow[flow.isnull().values]


# In[4]:

nanDF


# In[5]:

flowd = flow.as_matrix()
probd = prob.as_matrix()

flowd[np.where(np.isnan(flowd))] = np.take(np.nansum((flowd*(probd/np.nansum(probd, axis=1)[:,None])), axis=1), np.where(np.isnan(flowd))[0], axis=0)
probd[np.where(np.isnan(probd))] = np.take(np.nanmean(probd, axis=1), np.where(np.isnan(probd))[0], axis=0)

flowDF = pd.DataFrame(np.ceil(flowd).astype(int))
probDF = pd.DataFrame(probd)


# In[6]:

flowDF.head()


# ### Method 3

# In[7]:

pflow3 = flowDF.copy(deep=True)
pprob3 = probDF.copy(deep=True)
pflow3.columns = [str('flowD%s' % i) for i in range(pflow3.shape[1])]
pprob3.columns = [str('probD%s' % i) for i in range(pprob3.shape[1])]


# ### Method 1

# In[8]:

from sklearn.cross_validation import train_test_split
from sklearn import linear_model
reg = linear_model.LinearRegression()

flowD = flowDF.copy(deep=True)
probD = probDF.copy(deep=True)


# In[9]:

if(not flowD.shape[1]<2):
    for i in range(flowD.shape[1]):
        features = flowD.drop(i ,axis=1).as_matrix()
        target = flowD.as_matrix(columns=[i])
        reg.fit(features, target)
        locals()['pred%s' % i] = reg.predict(features)
        
    pflow1 = pd.DataFrame(index=range(len(flowD)))
    for i in range(flowD.shape[1]):
        pflow1[i] = pd.DataFrame(np.ceil(locals()['pred%s' % i]).astype(int))

    pprob1 = pd.DataFrame(index=range(len(prob)))
    for i in range(prob.shape[1]):
        feat = prob.drop(i ,axis=1)
        pprob1[i] = feat.mean(axis=1)

    pflow1.columns = [str('flowD%s' % i) for i in range(flowD.shape[1])]
    pprob1.columns = [str('probD%s' % i) for i in range(probD.shape[1])]
    
else:
    pflow1 = pd.DataFrame(np.zeros(len(flowD)).astype(int))
    pprob1 = pd.DataFrame(np.zeros(len(flowD)).astype(int))


# ### Method 2

# In[10]:

flowD = flowDF.copy(deep=True)
probD = probDF.copy(deep=True)

timestampD = pd.read_csv(path+'/timestamp.tsv', header=None, sep='\t')
timestampD[0] =  pd.to_datetime(timestampD[0], format='%Y-%m-%dT%H:%M:%S')
timestampD.fillna(pd.DataFrame(index=range(1)), inplace=True)


# In[11]:

flowA = pd.DataFrame(np.zeros(shape=(1,flowD.shape[1])).astype(int)).append(flowD, ignore_index=True)
flowA = flowA.drop(flowA.index[[len(flowA)-1]])
flowB = flowD[1:len(flowD)].reset_index().drop('index',1)
flowB.loc[len(flowB)] = 0
flowD.columns = [str('flowD%s' % i) for i in range(flowD.shape[1])]
flowA.columns = [str('flowA%s' % i) for i in range(flowA.shape[1])]
flowB.columns = [str('flowB%s' % i) for i in range(flowB.shape[1])]
flowdf = pd.concat([flowD,flowA,flowB], axis=1)


# In[12]:

probA = pd.DataFrame(np.zeros(shape=(1,probD.shape[1])).astype(int)).append(probD, ignore_index=True)
probA = probA.drop(probA.index[[len(probA)-1]])
probB = probD[1:len(probD)].reset_index().drop('index',1)
probB.loc[len(probB)] = 0
probD.columns = [str('probD%s' % i) for i in range(probD.shape[1])]
probA.columns = [str('probA%s' % i) for i in range(probA.shape[1])]
probB.columns = [str('probB%s' % i) for i in range(probB.shape[1])]
probdf = pd.concat([probD,probA,probB], axis=1)


# In[13]:

timedf = pd.DataFrame(index=range(1))
timedf[0] = pd.to_datetime(np.zeros(1).astype(int)[0])
timestampA = timedf.append(timestampD, ignore_index=True)
timestampA = timestampA.drop(timestampA.index[[len(timestampA)-1]])
timestampB = timestampD[1:len(timestampD)].reset_index().drop('index',1)
timestampB.loc[len(timestampB)] = timedf
timestampD.columns = [str('timeD')]
timestampA.columns = [str('timeA')]
timestampB.columns = [str('timeB')]
timestampD['timePre'] = (timestampD['timeD']-timestampA['timeA'])<timedelta(minutes=10)
timestampD['timeNext'] = (timestampB['timeB']-timestampD['timeD'])<timedelta(minutes=10)
timestampD = pd.concat([timestampD,flowdf,probdf], axis=1)
timestamp = timestampD.groupby(['timePre', 'timeNext'])


# In[14]:

start_time = time.time()
for group, i in zip(timestamp.groups, range(len(timestamp.groups))):
    locals()['result%s' % i] = timestamp.get_group(group)
    df = timestamp.get_group(group)

    if group == (True, True):
        for flA, flB, prA, prB, j in zip(flowA.columns, flowB.columns, probA.columns, probB.columns, range(flowD.shape[1])):
            fl = df[flA].multiply(df[prA]/(df[prA]+df[prB] + 1))            .add(df[flB].multiply(1-(df[prA]/(df[prA]+df[prB] + 1))))
            locals()['result%s' % i][str('flow%s' % j)] = np.ceil(fl).astype(int)
            locals()['result%s' % i][str('prob%s' % j)] = df[[prA, prB]].min(axis=1)
            
    if group == (True, False):
        for flA, prA, j in zip(flowA.columns, probA.columns, range(flowD.shape[1])):
            locals()['result%s' % i][str('flow%s' % j)] = df[flA]
            locals()['result%s' % i][str('prob%s' % j)] = df[prA]
            
    if group == (False, True):
        for flB, prB, j in zip(flowB.columns, probB.columns, range(flowD.shape[1])):
            locals()['result%s' % i][str('flow%s' % j)] = df[flB]
            locals()['result%s' % i][str('prob%s' % j)] = df[prB]
            
    if group == (False, False):
        for flD, prD, j in zip(flowD.columns, probD.columns, range(flowD.shape[1])):
            locals()['result%s' % i][str('flow%s' % j)] = df[flD]
            locals()['result%s' % i][str('prob%s' % j)] = df[prD]
            
print("--- %s seconds ---" % (time.time() - start_time))


# In[15]:

for j in range(4):
    locals()['result%s' % j] = locals()['result%s' % j][[str('flow%s' % i) for i in range(flowD.shape[1])] + [str('prob%s' % i) for i in range(probD.shape[1])]]
    
for i in reversed(range(3)):
    locals()['result%s' % 3] = locals()['result%s' % 3].append(locals()['result%s' % i])
    


# In[16]:

start_time = time.time()
locals()['result%s' % 3].sort_index(inplace=True)
print("--- %s seconds ---" % (time.time() - start_time))


# In[17]:

predicted = pd.concat([locals()['result%s' % 3]], axis=0)
pflow2 = predicted[[str('flow%s' % i) for i in range(flowD.shape[1])]]
pprob2 = predicted[[str('prob%s' % i) for i in range(probD.shape[1])]]


# ### Merged Flow

# In[18]:

f1 = pflow1.as_matrix()
f2 = pflow2.as_matrix()
f3 = pflow3.as_matrix()

p1 = pprob1.as_matrix()
p2 = pprob2.as_matrix()
p3 = pprob3.as_matrix()

w1 = np.nan_to_num(p1/(p1+p2+p3))
w2 = np.nan_to_num(p2/(p1+p2+p3))
w3 = np.nan_to_num(p3/(p1+p2+p3))


# In[19]:

mergeFinal = w1*f1 + w2*f2 + w3*f3
mergeFinalDF = pd.DataFrame(np.ceil(mergeFinal).astype(int))
mergeFinalDF.astype(int)


# In[20]:

for col in flow.columns:
    index = nanDF[nanDF[col].isnull().values][col].index
    mergeFinalDF.set_value(index, col, ' ')


# In[21]:

import sys


# In[22]:

mergeFinalDF.to_csv(path[-4:]+'.flow.txt', sep='\t', header=False, index=False)


# In[23]:

mergeFinalDF.isnull().values


# In[24]:

mergeFinalDF[~mergeFinalDF.isnull().values]


# In[ ]:



