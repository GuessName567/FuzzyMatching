# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:26:47 2022

@author: luguangli
"""

import json

import os
import numpy as np
import pandas as pd
import re
import datetime as dt
import time as tm
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

pbidlink=r'D:\Dropbox\Company name matching general\checked company name pairs\name pairs from bright indeed same person company job match'.replace('\\','/')
pindeed='E:/Indeed'
pbright='E:/Bright'
#%%
"""Step 1: do fuzzy matching between company names in bright company file, select companies with similar names but different id for each bright company
   Step 2: use indeed-bright link to create more observations following the same logic 
   Step 3: train the name pairing BERT algorithm, and save the model
   Step 4: train the random forest model based on other features generated from name pairs, and save the model
"""
os.chdir(pbright)
bname=pd.read_pickle('bright_linkedin_individual_profiles_compname_compid.pkl')


alphalist=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def get_first_letter_list(v):
    vlist=v.split()
    vlist=[nv[0] for nv in vlist if type(nv)==str and len(nv)>0][:2] 
    return vlist

def segment_dataframe(df,filename,compname):
    """apply the function above to get a list of first letters"""
    df['flist']=df[compname].apply(get_first_letter_list)
    
    """extract first letters and rearrange the data to long format"""
    for n in range(2):
        df[f'fl{n+1}']=df.flist.str.get(n)
    df=df[[compname,'fl1','fl2']].set_index(compname)
    df.columns=pd.MultiIndex.from_tuples([('fl',n) for n in range(1,3)])
    df=df.stack().reset_index()
    df=df[[compname,'fl']]
    df.loc[df['fl'].isin(alphalist)==False,'fl']='other'
    
    """export data into different parts based on first letter"""
    for f, group in df.groupby('fl'):
        group.to_pickle(f'{filename}_fl_{f}.pkl')

compname,filename='compname','bright_linkedin_individual_profiles_compname_compid'

segment_dataframe(bname,filename,compname)
#%%
for f in alphalist+['other']:
    print(f'processing letter {f}')
    t1=tm.time()
    df=pd.read_pickle((f'{filename}_fl_{f}.pkl'))
    print(f'letter {f} has about {df.shape[0]} observations')
    namepair=match_strings(df['compname'].drop_duplicates(), df['compname'].drop_duplicates(), max_n_matches=5, 
                       min_similarity=0.3,ignore_index=True,n_blocks=(1,8))
    
    namepair.to_pickle(f'{filename}_namepair_fl_{f}.pkl')
    t2=tm.time()
    print(f'finish processing letter {f} and it takes {int((t2-t1)/60)} minutes')

#%%

# isid(bname,['company_id','compname'])
# bname['nuni']=bname.groupby('company_id').compname.transform('count')
# bname.nuni.describe(percentiles=perc)
bname=bname.reset_index(drop=True)
bname.to_feather('bright_nameid.feather')


""" fuzzy matching on compname"""




os.chdir(pbidlink)
bidlink=pd.read_pickle('indeed_bright_same_person_company_match_final.pkl')

temp=bidlink[['compname_i','compname_b','company_id']].sample(100)


