# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 18:54:55 2022

@author: luguangli
"""

import json

import os
import numpy as np
import pandas as pd
import re
import datetime as dt
import time as tms
import gc
gc.enable()
import zipfile
import glob
import collections
from multiprocessing import Pool
from multiprocessing import cpu_count
import tarfile
from cleanco import basename
import swifter
from string_grouper import  match_most_similar ,match_strings, group_similar_strings, compute_pairwise_similarities
, StringGrouper


perc=[0.005, 0.01,0.025, 0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975, 0.99,0.995] ### used when checking the summary stats of a data

    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import time
import pandas as pd
import numpy as np
from cleanco import basename

import textdistance as td

def ratcliff_similarity(row,varname1,varname2):
    try:
        return td.ratcliff_obershelp(basename(row[varname1].lower()),basename(row[varname2].lower()))
    except:
        return np.nan
    
from difflib import SequenceMatcher
     

def isid(data, variables):
    dup = data.duplicated(variables)
    dupsum = dup.describe()
    if (dupsum['unique'] == 1):
        print(str(variables) + " uniquely identify this dataset")
    else:
        print(str(variables) + " Do NOT uniquely identify this dataset")
        
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 120)

#%% 
"""
================================================================================
    The data inputs for the code should be name pairs from fuzzy matching before
================================================================================
"""

#%%
"""chain together single letter"""
def chainsl(v):
    vlist=re.split('[ \.]',v)
    vlist=[k for k in vlist if k not in ['',' ']]
    nv=''
    if len(vlist)>0:
        nv=vlist[0]
        for n in range(1,len(vlist)):
            if len(vlist[n-1])==1 and vlist[n]=='&':
                nv=nv
            elif len(vlist[n-1])==1 and len(vlist[n])==1:
                nv=nv+vlist[n]
            else:
                nv=nv+' '+vlist[n]
    return nv

### example application:
for v in ['compname1','compname2']: 
    df[v]=df[v].swifter.allow_dask_on_strings().progress_bar(False).apply(chainsl)

#%% 
"""compute string similarity/distance: due to the restriction of the paralell computing package swifter, we'll have to specify the input variables """

def sr(row):
    a,b='compname1','compname2' #compname1/2 should be string variable name
    try:
        return SequenceMatcher(None, row[a], row[b]).ratio()    
    except:
        return 0
    
df['compdist']=df[['compname1','compname2']].swifter.allow_dask_on_strings().progress_bar(False).apply(sr,axis=1)    

#%%
"""because earlier parts of company names are usually more important, compute string distance of first 15 letters"""
def fdist(row): 
    fdist=[np.nan]*2
    a,b='compname1','compname2' #compname1/2 should be string variable name
    try:
        astr=re.sub('\W','',row[a])[:15]
        bstr=re.sub('\W','',row[b])[:15]
        fdist[0]=SequenceMatcher(None,astr,bstr).ratio()            
        lent=min(len(astr),len(bstr))
        fdist[1]=SequenceMatcher(None,astr[:lent],bstr[:lent]).ratio()         
    except:
        pass    
    return fdist

df['fdist']=df[['compname1','compname2']].swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply(fdist,axis=1) 
df['fdist1']=df['fdist'].str.get(0)
df['fdist2']=df['fdist'].str.get(1)


#%% 
"""compute gram-distance: due to the restriction of the paralell computing package swifter, we'll have to specify the input variables"""
def gramdist(row,n=3):
    a,b='compname1','compname2' #compname1/2 should be string variable name
    try:
        alist=re.split('\W',row[a])
        alist=[k for k in alist if k not in ['',' ']]
        if len(alist)>n:
            alist=alist[:n]
        r1=len([v for v in alist if v in row[b]])/len(alist) 
        
        blist=re.split('\W',row[b])
        blist=[k for k in blist if k not in ['',' ']]
        if len(blist)>n:
            blist=blist[:n]
        r2=len([v for v in blist if v in row[a]])/len(blist)
        
        return [r1,r2]
    
    except:
        return [np.nan]*2

df['compgdist']=df[['compname1','compname2']].swifter.allow_dask_on_strings().progress_bar(False).apply(gramdist,axis=1)  # the output is a list   
df['compgdist1']=df.compgdist.str.get(0) #extract data from output
df['compgdist2']=df.compgdist.str.get(1) #extract data from output

#%%
"""asses whether the first gram of a name is in another name: due to the restriction of the paralell computing package swifter, we'll have to specify the input variables"""
def samefirst(row):
    a,b='compname1','compname2' #compname1/2 should be string variable name
    reslist=[False,False,False]
    if len(row[a])>0 and len(row[b])>0:    
        alist=re.split('\W',row[a])
        alist=[k for k in alist if k not in ['',' ']]
        blist=re.split('\W',row[b])
        blist=[k for k in blist if k not in ['',' ']]
        if len(alist)>0 and len(blist)>0:
            reslist[0]= alist[0]==blist[0]
            reslist[2]=(alist[0] in blist[0]) or (blist[0] in alist[0])
            reslist[1]=reslist[2]
            if len(alist)>1 and len(blist)>1:
                reslist[1]=(alist[0]==blist[0]+blist[1]) or (blist[0]==alist[0]+alist[1])
            elif len(alist)>1 and len(blist)==1:           
                reslist[1]=(alist[0]==blist[0]) or (blist[0]==alist[0]+alist[1])
            elif len(alist)==1 and len(blist)>1:           
                reslist[1]=(alist[0]==blist[0]+blist[1]) or (blist[0]==alist[0])
    return reslist

df['samefirst']=df[['compname1','compname2']].swifter.allow_dask_on_strings().progress_bar(False).apply(samefirst,axis=1)    
df['samefirst1']=df['samefirst'].str.get(0).astype(float)
df['samefirst2']=df['samefirst'].str.get(1).astype(float)
df['samefirst3']=df['samefirst'].str.get(2).astype(float)

#%% 
"""Assess whether any the name in the name pairs contains stop words: these words will lower the matching quality mechanically"""

"""stopping words: punctuations"""
swpunc=[':',',',';','@','\|','\(','/','\!']

"""stopping words that indicate relationship"""
swrel=['at','formerly','now','acquired','prior( to)?','previously','before','(also )?known as','d(\.| )?b(\.| )?a','formally', 'outsourced to','via','a\.?k\.?a\.?']

"""stopping words that indicate suffix"""
suf=['inc','corp','ltd','grp','co','llc','plc','hldg','company','limited partner','na','spa','sae','ab','bv','corporate','ag','pa',
     'sa','corps','llp','as','lp','inc','co','worldwide','gmbh','cla','brand','de','nv','cos','holdings?']

"""keywords that indicate function/subunit/geographical location"""
funlist=['clearstaff','staff', 'receptionist','specialist',
        'affiliate','road','street','boulevard','full(\W?time)|($)','part(\W?time)|($)']

""" words often used associated with of: even locations appear in these kind of names, they are still normal names"""
oflist=['bank','bk','university','schools?','offices?','institute','city','county','department','boards?','state','states','hospital','office']

def splcont(v):
    vlist=[k for k in re.split('',v) if k not in ['',' ']]
    nv=v[0]
    for n in range(1,len(vlist)):
        nv=nv+' ?'+vlist[n]
    return nv

suf=[splcont(v) for v in suf]    
spstr1=''+'|'.join(swpunc)+''
spstr2='(\W'+'\W)|(\W'.join(swrel+suf)+'\W)'
spstr3='(\W'+')|(\W'.join(funlist)+')'
specstr=spstr1+spstr2+spstr3
ofstr='('+' of)|('.join(oflist)+' of)'

def has_spec_word(row):
    v=  float(bool(re.search(specstr, row['compname1']))) +  float(bool(re.search(specstr, row['compname2'])))
    v=v+float(bool(re.search(' of ', row['compname1'])))*(1-float(bool(re.search(ofstr, row['compname1'])))) +  \
        float(bool(re.search(' of ', row['compname2'])))*(1-float(bool(re.search(ofstr, row['compname2']))))
    return v

df['has_spec_word']=df.swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply(has_spec_word,axis=1)
   
#%% 
"""splitting by stopping words: note, this does not include funlist and oflist above"""
spstr1=''+'|'.join(swpunc)+''
spstr2='(\W'+'\W)|(\W'.join(swrel+suf)+'\W)'
spstr=spstr2+'|'+spstr1

def spt(text):
        tlist=re.split(spstr,text)
        tlist=[t for t in tlist if type(t)==str and len(t.strip())>1] #get rid of purely empty words and single letter words
        tlist= [t for t in tlist if bool(re.search(spstr, t))==False and t not in suf][:2] # extract first 2 non-spliting parts
        if bool(re.search('(\Wat\W)|(@)', v)): # when at or @ in the middle of the name, the first part is more important, e.g. starbuck @ **shopping mall
            tlist[1]=np.nan
        for n in range(len(tlist)):
            try:
                tlist[n]=re.sub('(^\W)|(\W$)','',re.sub(' +',' ',tlist[n])) #clean names by replace extra spaces and abnormal notations at the begining/end of word
            except:
                pass
        return tlist
    
for v in ['compname1','compname2']:
    df[v+'_parts']=df[v].swifter.allow_dask_on_strings().progress_bar(False).apply(spt)


"""Generating pairwise distance based on first/second parts of the standardized company names"""
def dislistfun(row):
    distlist=[np.nan]*4
    a,b='compname1_parts','compname2_parts'
    n=0    
    for i in [0,1]:
        for j in [0,1]:
            n+=1
            if i==1:
                if j==0:                
                   try:
                       distlist[n-1]=SequenceMatcher(None,re.sub('^the ','',row[a][i]),row[b][j]).ratio()                                      
                   except:
                       pass                       
                else:
                   try:
                       distlist[n-1]=SequenceMatcher(None,re.sub('^the ','',row[a][i]),re.sub('^the ','',row[b][j])).ratio()                                       
                   except:
                       pass                       
            else:
                if j==1:
                   try:
                       distlist[n-1]=SequenceMatcher(None,row[a][i],re.sub('^the ','',row[b][j])).ratio()                                       
                   except:
                       pass                       
                else: 
                   try:
                       distlist[n-1]=SequenceMatcher(None,row[a][i],row[b][j]).ratio()                                       
                   except:
                       pass                       
                     
    return distlist

df['distxy']=df[['compname1_parts','compname2_parts']].swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply(dislistfun,axis=1) # this distance is after getting rid of locations/special letters etc...

### extract distance variables
for n in range(4):
    df[f'distxy{n+1}']=df['distxy'].str.get(n)

def strlen(v):
    # vlist=re.split('\W',v)
    # vlist=[k for k in vlist if k not in ['',' ']]
    return len(vlist)  

for v in ['compname1_parts','compname2_parts']:
    df[v+'_numparts']=df[v].swifter.allow_dask_on_strings(enable=True).progress_bar(False).apply(strlen,axis=1)

#%%
"""
===================================================================================================================================================================
======================== at this point, you should merge back the database with raw company names, and compuste string distances based on raw names ===============
===================================================================================================================================================================

"""
def basic_std(v):
    try: 
        return re.sub('\W','', re.sub('^ *the',' ',v.lower())).strip()
    except:
        return v

def sr(row):
    a,b='compname_raw1_bstd','compname_raw2_bstd' #compname1/2 should be string variable name
    try:
        return SequenceMatcher(None, row[a], row[b]).ratio()    
    except:
        return 0
    

for v in ['compname_raw1','compname_raw2']:
    df[v+'_bstd']=df[v].swifter.allow_dask_on_strings().progress_bar(False).apply(basic_std)
    
    
df['comprcosim']=compute_pairwise_similarities(df['compname_raw1_bstd'],df['compname_raw2_bstd']) #if this one is too slow, can skip
df['comprdist']=df[['compname_raw1_bstd','compname_raw2_bstd']].swifter.allow_dask_on_strings().progress_bar(False).apply(sr,axis=1) 


"""
================================================================================================================================================================================================================
======================== at this point, please send me pairs of both raw company names and standardized company names, I'll use BERT model to estimate whether they are likely the same company ===============
================================================================================================================================================================================================================

"""


"""
================================================================================================================================================================================================================
======================== after getting bert score for each pairs, please do manual check of 10000-15000 company pairs, based on all the information, including name, geogrpahical, industyr ====================
================================================================================================================================================================================================================
"""


"""
======================================================================================================================================================
======================== after manual checking, please use random forest to estimate whether each pair of companies is true match ====================
======================================================================================================================================================
"""


