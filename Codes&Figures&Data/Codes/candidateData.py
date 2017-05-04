
# coding: utf-8

# In[177]:

import sys
import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

CANDIDATES = ['Hillary','Cruz','Sanders','Donald']
primary_results = 'primary_results.csv'
county_facts = 'county_facts.csv'

print("loading %s" % primary_results)
primaryResultDF = pd.read_csv(primary_results,sep=',',low_memory=False)
print("loading %s" % county_facts)
countyFactsDF = pd.read_csv(county_facts,sep=',',low_memory=False)




# In[178]:

#remove the state rows
countyFactsDF = countyFactsDF[countyFactsDF['state_abbreviation']!= ""]


# In[180]:

def concatCountyFactsWithPrimaryResult(countyDF, primaryDF, popDF, sentDF, name):
    primaryDataForName = primaryDF[primaryDF.candidate == name]
    countyDF = countyDF[countyDF['fips'].isin(primaryDataForName['fips'])]
    popDF = popDF[popDF['county'].isin(primaryDataForName['county'])]
    sentDF = sentDF[sentDF['county'].isin(primaryDataForName['county'])]
    frames = [primaryDataForName,countyDF]
    result = pd.merge(primaryDataForName, countyDF, on=['fips', 'fips'])
    result.drop(['fips','party','candidate','fraction_votes','area_name','state_abbreviation_x','state_abbreviation_y'],inplace=True,axis=1)
    result.to_csv('./YYYYYY.csv')
    popDF.to_csv('./ZZZZZZZ.csv')
    result = pd.merge(result,popDF,on = ['state','county'],how = 'inner')
    result.to_csv('./XXXXXXX.csv')
    result = pd.merge(result,sentDF,left_on = ['state','county'],right_on = ['state','county'],how = 'inner')
    return result
    
def getCandidateDF(name):
    return concatCountyFactsWithPrimaryResult(countyFactsDF, primaryResultDF, name)

def getMatrix(name,train_or_test):
    list1= []
    list2 = []
    popDF = pd.read_excel('popularity_' + train_or_test + '.xlsx',sheetname = name,sep = ',')
    sentDF = pd.read_excel('sentiment_' + train_or_test + '.xlsx',sheetname = name,sep = ',')
    pInd = popDF.index
    pCol = popDF.columns
    sInd = sentDF.index
    sCol = sentDF.columns
    for i in pInd:
        popDF.loc[i,'county'] = popDF.loc[i,'county'].lstrip().rstrip()
        popDF.loc[i,'state'] = popDF.loc[i,'state'].lstrip().rstrip()
    for i in sInd:
        sentDF.loc[i,'county'] = sentDF.loc[i,'county'].lstrip().rstrip()
        sentDF.loc[i,'state'] = sentDF.loc[i,'state'].lstrip().rstrip()
    result = concatCountyFactsWithPrimaryResult(countyFactsDF,primaryResultDF,popDF,sentDF,name)
    result = result.reindex(np.random.permutation(result.index))
    result2 = result
    ind = result.index
    for i in ind:
        if result.loc[i,'votes'] == 0:
            result = result.drop(i)
    county = result2['county']
    state = result2['state']
    y = result2['votes']  
    result2.drop(['votes','county','state'],inplace=True,axis=1) 
    list2.append(result2)
    list2.append(y)
    list2.append(county)
    list2.append(state)
    county = result['county']
    state = result['state']
    y = result['votes']
    result.drop(['votes','county','state'],inplace = True,axis = 1)
    list1.append(result)
    list1.append(y)
    list1.append(county)
    list1.append(state)

    return [list1,list2]



    


# In[182]:



# In[ ]:




# In[ ]:




# In[ ]:



