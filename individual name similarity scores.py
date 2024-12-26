# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:06:01 2022

@author: luguangli
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 17:33:58 2022

@author: luguangli
"""

### Search for partner death from google
import os
import pandas as pd
import numpy as np
import time as tm
from apify_client import ApifyClient
from cleanco import basename
import re
import glob
def isid(data, variables):
    dup = data.duplicated(variables)
    dupsum = dup.describe()
    if (dupsum['unique'] == 1):
        print(str(variables) + " uniquely identify this dataset")
    else:
        print(str(variables) + " Do NOT uniquely identify this dataset")
from string_grouper import  match_most_similar ,match_strings, group_similar_strings, compute_pairwise_similarities
, StringGrouper
        
perc=[0.005, 0.01,0.025, 0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975, 0.99,0.995]
#%% match ciqpro to pitchbook
"""concate name together, as the sequence of first/last names may switch, so it's better to concate together than score similarity based on algorithms not so dependent on order
   below is an example for one name.  
   You'll typically need to repeat the algorithm twice to standardize individual name from both datasets
"""
for v in ['fname','mname','lname']:
    df[v]=df[v].fillna('')
df['fullname']=df['fname']+' '+df['mname']+' '+df['lname']
df['fullname']=df['fullname'].str.replace(' +',' ').str.strip().str.lower()


"""compute overall string similarity score between individual names"""
def sr(row):
    a,b='fullname1','fullname2' # 'pname','fullname' should be string variable name
    try:
        return SequenceMatcher(None, row[a], row[b]).ratio()    
    except:
        return 0
    
df['perscore']=df.swifter.allow_dask_on_strings().progress_bar(False).apply(sr,axis=1)   


""" compute gram distance"""
def gramdist(row,n=10):
    a,b='fullname1','fullname2'
    try:
        alist=row[a].split()  #split name
        if len(alist)>n:      #only get first n parts
            alist=alist[:n] #only look at the first n strings
        r1=len([v for v in alist if v in row[b]])/len(alist) 
        
        blist=row[b].split() #only look at the first n strings
        if len(blist)>n:
            blist=blist[:n]
        r2=len([v for v in blist if v in row[a]])/len(blist)
        
        return [r1,r2]
    
    except:
        return [np.nan]*2

df['pergsim']=df.swifter.allow_dask_on_strings().progress_bar(False).apply(gramdist,axis=1)
df['pergsim1']=df['pergsim'].str.get(0)   
df['pergsim2']=df['pergsim'].str.get(1)   

"""assess the consistency of first letters between parts of names """
def first_letter_sim(row):
    a,b='fullname1','fullname2'
    try:
        alist=row[a].split() #split name
        alist=[v[0] for v in alist] #get the list of first letters
        
        blist=row[b].split() #only look at the first n strings
        blist=[v[0] for v in blist]
            
        r1=len([v for v in alist if v in blist])/len(alist) #fraction of first letter in another list of first letters
        
        r2=len([v for v in blist if v in alist])/len(blist) #fraction of first letter in another list of first letters
        
        return [r1,r2]
    
    except:
        return [np.nan]*2

df['flsim']=df.swifter.allow_dask_on_strings().progress_bar(False).apply(first_letter_sim,axis=1)
df['flsim1']=df['flsim'].str.get(0)   
df['flsim2']=df['flsim'].str.get(1)   




















