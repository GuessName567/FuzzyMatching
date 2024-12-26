#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:52:56 2022

@author: rnd
"""
import os
import pandas as pd
import glob
from string_grouper import  match_most_similar ,match_strings, group_similar_strings, compute_pairwise_similarities, StringGrouper
import numpy as np 
import swifter
import cloudpickle
import time as tm
import re
from difflib import SequenceMatcher
import shutil
from cleanco import basename

def isid(data, variables):
    dup = data.duplicated(variables)
    dupsum = dup.describe()
    if (dupsum['unique'] == 1):
        print(str(variables) + " uniquely identify this dataset")
    else:
        print(str(variables) + " Do NOT uniquely identify this dataset")
        
perc=[0.005, 0.01,0.025, 0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975, 0.99,0.995]
#%%
"""
*******************************************************************************
***********************classify names based on various name distances**********
*******************************************************************************
"""

### collect matching
npair=pd.concat(pd.read_pickle(file) for file in glob.glob('revelio_ciq_namematch*'))
npair.similarity.describe(percentiles=perc)
npair.columns=['namestd','cosim','ciqname']
npair=npair.drop_duplicates(['namestd','ciqname']).reset_index(drop=True)
npair.to_pickle('revelio_ciq_namepair.pkl')

### re-organize files in folder to make it easy to look
flist=glob.glob('ciqpro_name_std_fl_*')
for file in flist:
    shutil.move(f"./{file}", f"./ciqname by first letter/{file}")

flist=glob.glob('revelio_name_std_fl*')
for file in flist:
    shutil.move(f"./{file}", f"./revelio name by first letter/{file}")

flist=glob.glob('revelio_ciq_namematch_*.pkl')
for file in flist:
    shutil.move(f"./{file}", f"./revelio ciq name pair by first letter/{file}")

### compute various distances
"""compute overall string similarity score between individual names"""
a,b='namestd','ciqname' # 'pname','fullname' should be string variable name
def sr(row):
    try:
        return SequenceMatcher(None, row[a], row[b]).ratio()    
    except:
        return 0
    
npair['perscore']=npair.swifter.allow_dask_on_strings().progress_bar(False).apply(sr,axis=1)   


""" compute gram distance"""
def gramdist(row,n=10):
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

npair['pergsim']=npair.swifter.allow_dask_on_strings().progress_bar(False).apply(gramdist,axis=1)
npair['pergsim1']=npair['pergsim'].str.get(0)   
npair['pergsim2']=npair['pergsim'].str.get(1)   

"""assess the consistency of first letters between parts of names """
def first_letter_sim(row):
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

npair['flsim']=npair.swifter.allow_dask_on_strings().progress_bar(False).apply(first_letter_sim,axis=1)
npair['flsim1']=npair['flsim'].str.get(0)   
npair['flsim2']=npair['flsim'].str.get(1)   

npair.to_pickle('revelio_ciq_namepair_withscore.pkl')

#%%
npair['score_sum']=npair.pergsim1+npair.pergsim2+npair.flsim1+npair.flsim2#用于排序分组，无意义

### classify based on various name scores
#总分大于3.5
npair.loc[npair.score_sum>=3.5,'level']=1
#总分在3-3.5之间且pergsim和flsim至少各有一个1
npair.loc[(npair.score_sum<3.5)&(npair.score_sum>=3.0)&((npair.pergsim1==1)|\
   (npair.pergsim2==1))&((npair.flsim1==1)|(npair.flsim2==1)),'level']=2
#总分在3-3.5之间但pergsim或flsim只有一边有1
npair.loc[(npair.score_sum<3.5)&(npair.score_sum>=3.0)&\
          (((npair.pergsim1!=1)&(npair.pergsim2!=1)&((npair.flsim1==1)|(npair.flsim2==1)))|\
           (((npair.pergsim1==1)|(npair.pergsim2==1))&(npair.flsim1!=1)&(npair.flsim2!=1))) ,'level']=3
#总分在3-3.5且没有1
npair.loc[(npair.score_sum<3.5)&(npair.score_sum>=3.0)&(npair.pergsim1!=1)&\
   (npair.pergsim2!=1)&(npair.flsim1!=1)&(npair.flsim2!=1),'level']=4
#总分在2-3之间且pergsim和flsim至少各有一个1
npair.loc[(npair.score_sum<3.0)&(npair.score_sum>=2.0)&((npair.pergsim1==1)|\
   (npair.pergsim2==1))&((npair.flsim1==1)|(npair.flsim2==1)),'level']=5
#总分在2-3之间但pergsim或flsim只有一边有1  
npair.loc[(npair.score_sum<3.0)&(npair.score_sum>=2.0)&\
          (((npair.pergsim1!=1)&(npair.pergsim2!=1)&((npair.flsim1==1)|(npair.flsim2==1)))|\
           (((npair.pergsim1==1)|(npair.pergsim2==1))&(npair.flsim1!=1)&(npair.flsim2!=1))) ,'level']=6
#总分在2-3且没有1
npair.loc[(npair.score_sum<3.0)&(npair.score_sum>=2.0)&(npair.pergsim1!=1)&\
   (npair.pergsim2!=1)&(npair.flsim1!=1)&(npair.flsim2!=1),'level']=7
#总分小于2但有1
npair.loc[(npair.score_sum<2)&((npair.pergsim1==1)|\
   (npair.pergsim2==1)|(npair.flsim1==1)|(npair.flsim2==1)),'level']=8
#总分小于2且无1   
npair.loc[(npair.score_sum<2)&(npair.pergsim1!=1)&\
   (npair.pergsim2!=1)&(npair.flsim1!=1)&(npair.flsim2!=1),'level']=9

npair=npair.rename(columns={'level':'matchlevel'})    
    
# npair.groupby('matchlevel').sample(100).to_excel('idv name pairs by match level.xlsx')    
    
npair=npair[(npair.matchlevel<=4) | (npair.pergsim1==1) | (npair.pergsim2==1)]

npair=npair[(npair.perscore>0.85) | (npair.pergsim1==1) | (npair.pergsim2==1) ]

npair=npair.drop('score_sum',1)

npair.to_pickle('revelio_ciq_namepair_withscore_matchlevel.pkl')

#%%
"""match name pairs to companies in their respective datasets: indicating clear sources"""

npair=npair.fillna('')
ciqpro=ciqpro.fillna('')
cpair=npair.merge(ciqpro,on=['ciqname','altname'],how='left')

cpair=cpair.rename(columns={'personid':'pid_ciq', 'personname':'name_ciq',
'companyname':'comp_ciq', 'title':'title_ciq', 'startyear':'startyear_ciq', 
'endyear':'endyear_ciq'})

### realize that suffix data might be helpful, so get in
cpair=cpair.merge(ciqper[['personid','suffix']].rename(columns={'personid':'pid_ciq'}),
                  on='pid_ciq',how='left')
cpair=cpair.rename(columns={'suffix':'suffix_ciq'})

rname=pd.read_pickle('revelio_name_std.pkl')

isid(rname,list(rname.columns))

cpair=cpair.merge(rname,on='namestd',how='left')

cpair=cpair.rename(columns={'name':'name_rev','namestd':'revname','suffix':'suffix_rev'})

vlist=['suffix_rev','name_rev','revname', 'ciqname', 'name_ciq', 'suffix_ciq', 
       'altname', 'pid_ciq', 'matchlevel','cosim', 'perscore', 'pergsim1', 'pergsim2', 
       'flsim1', 'flsim2','comp_ciq', 'title_ciq', 'startyear_ciq', 'endyear_ciq']

cpair=cpair[vlist]

cpair=cpair.reset_index(drop=True)

cpair.to_pickle('revelio_ciq_namepair_bf_match2exp.pkl')

#%%
"""
**************************************************************
**************** match to revelio experience******************
**************************************************************
"""
os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/revlio experience/')

vlist=['user_id', 'fullname','company_raw',
       'jobtitle', 'startdate', 'enddate']

flist=glob.glob('*')

for file in flist:
    print(f'start processing {file}...')
    t1=tm.time()
    os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/revlio experience/')   
    df=pd.read_pickle(file)
    df=df[vlist].rename(columns={'fullname':'name_rev'}).merge(cpair,on='name_rev')
    os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/rev_ciq_match_allvar/')
    df.to_pickle(file.replace('us_us_exp_keyvar_name','rev_ciq_keyvar'))
    os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/rev_ciq_match_compname_pairs/')
    df[['company_raw','comp_ciq']].drop_duplicates().to_pickle(file.replace('us_us_exp_keyvar_name','rev_ciq_componly'))
    t2=tm.time()
    print(f'finish processing {file} and it takes {(int(t2-t1)/60)} mintues')
#%%
a,b='company_raw','comp_ciq' # 'pname','fullname' should be string variable name

def sr(row):
    try:
        return SequenceMatcher(None, basename(row[a]).lower(), basename(row[b]).lower()).ratio()    
    except:
        return 0

flist=glob.glob('*')

n=0
for file in flist:
    n+=1
    print(f'start processing file {n}...')
    t1=tm.time()
    df=pd.read_pickle(file)
    df['cdist']=df.swifter.allow_dask_on_strings().progress_bar(False).apply(sr,axis=1)
    df.to_pickle(file.replace('.pkl','_namedist.pkl'))
    t2=tm.time()
    print(f'finish processing {file} and it takes {(int(t2-t1)/60)} mintues')
#%%
flist1=glob.glob('*_namedist.pkl')
flist=glob.glob('*')
flist2=[f for f in flist if f not in flist1]

for file in flist2:
    os.remove(file)

#%%
os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/rev_ciq_match_allvar/')
flist=glob.glob('*')
n=0
for file in flist:
    n+=1
    print(f'start processing file {n}')
    os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/rev_ciq_match_allvar/')
    df=pd.read_pickle(file)
    os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/rev_ciq_match_compname_pairs/')
    name=pd.read_pickle(file.replace('keyvar','componly').replace('.pkl','_namedist.pkl'))
    name=name[name.cdist>=0.5]
    df=df.merge(name,on=['company_raw','comp_ciq'])
    df=df.sort_values(['user_id','pid_ciq','comp_ciq','cdist'],ascending=False)
    df['nrank']=df.groupby(['user_id','pid_ciq','comp_ciq']).cumcount()+1
    df=df[df.nrank<=3]
    os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/rev_ciq_match_pot_compname_pairs/')
    df[['company_raw','comp_ciq']].drop_duplicates().to_pickle(file.replace('keyvar','potcomponly'))
    os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/rev_ciq_match_pot_pairs/')
    df.to_pickle(file.replace('keyvar','keyvar_potpairs'))
#%%
os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/rev_ciq_match_pot_compname_pairs/')
flist=glob.glob('*potcomp*')
df=pd.concat(pd.read_pickle(file) for file in flist)
df=df.drop_duplicates()
df=df.reset_index(drop=True)   

for v in ['company_raw','comp_ciq']:
    df[v+'_lower']=df[v].str.lower().str.replace('^the ','')
    
    
df.to_pickle('rev_ciq_potcompname_all.pkl')

df2=df[['company_raw_lower','comp_ciq_lower']].drop_duplicates().reset_index(drop=True)

df2.to_pickle('rev_ciq_potcompname_all_lowercase.pkl')

for file in flist:
    os.remove(file)
#%%
os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/rev_ciq_match_pot_compname_pairs/')
flist1=glob.glob('*m1*')
flist2=glob.glob('*m2*')
flist3=glob.glob('*m3*')

df1=pd.concat(pd.read_pickle(file) for file in flist1)

df2=pd.concat(pd.read_pickle(file) for file in flist2)

df3=pd.concat(pd.read_pickle(file) for file in flist3)

df1=df1.rename(columns={'nlpscore':'nlpscore1'})

df2=df2.rename(columns={'nlpscore':'nlpscore2'})

df3=df3.rename(columns={'nlpscore':'nlpscore3'})

df=df1.merge(df2,on=list(df1.columns)[:2]).merge(df3,on=list(df1.columns)[:2])


df['nlpscoremax']=np.max(df[['nlpscore1','nlpscore2','nlpscore3']],axis=1)

df['nlpscoremin']=np.min(df[['nlpscore1','nlpscore2','nlpscore3']],axis=1)

df['nlpscoreavg']=np.mean(df[['nlpscore1','nlpscore2','nlpscore3']],axis=1)

df=df.drop(['nlpscore2','nlpscore3'],1)

npair=pd.read_pickle('rev_ciq_potcompname_all.pkl')

npair=npair.merge(df,on=list(df1.columns)[:2])

npair=npair.drop(list(df1.columns)[:2],1)

os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/rev_ciq_match_pot_pairs/')
flist=glob.glob('*keyvar_potpairs*')

df=pd.concat(pd.read_pickle(file) for file in flist)

df=df.merge(npair,on=list(npair.columns)[:2],how='left')

df['tid']=df.reset_index(drop=True).index

df.to_pickle('rev_ciq_keyvar_potpairs_withscores.pkl')

vlist=['tid', 'name_rev', 'company_raw', 'jobtitle', 'startdate',
       'enddate', 'suffix_rev', 'revname', 'ciqname', 'name_ciq', 'suffix_ciq',
       'altname', 'comp_ciq', 'title_ciq',
       'startyear_ciq', 'endyear_ciq']

# mc=df.loc[(df.nlpscoreavg>=0.9)| (df.cdist>=0.7),vlist].sample(3000)

mc.columns=['tid', 'name_rev', 'comp_rev', 'title_rev', 'startdate_rev',
        'enddate_rev', 'suffix_rev', 'name_std_rev', 'name_std_ciq', 'name_ciq', 'suffix_ciq',
        'altname_ciq', 'comp_ciq', 'title_ciq',
        'startyear_ciq', 'endyear_ciq']

vlist=['tid', 'comp_rev','comp_ciq', 'name_rev', 'name_ciq',
        'altname_ciq','suffix_rev','suffix_ciq',
        'title_rev', 'title_ciq','startdate_rev','startyear_ciq',
        'enddate_rev', 'endyear_ciq', 'name_std_rev', 'name_std_ciq']

# mc=mc[vlist]

# mc.to_excel('ciq_revelio_manualcheck.xlsx')

#%%%
"""compute distance for job title & date"""
df=pd.read_pickle('rev_ciq_keyvar_potpairs_withscores.pkl')

#rename to make it more clear
vlist1=['tid', 'name_rev', 'company_raw', 'jobtitle', 'startdate',
       'enddate', 'suffix_rev', 'revname', 'ciqname', 'name_ciq', 'suffix_ciq',
       'altname', 'comp_ciq', 'title_ciq',
       'startyear_ciq', 'endyear_ciq']
vlist2=['tid', 'name_rev', 'comp_rev', 'title_rev', 'startdate_rev',
        'enddate_rev', 'suffix_rev', 'name_std_rev', 'name_std_ciq', 'name_ciq', 'suffix_ciq',
        'altname_ciq', 'comp_ciq', 'title_ciq',
        'startyear_ciq', 'endyear_ciq']

rendict={}
for n in range(len(vlist1)):
    rendict[vlist1[n]]=vlist2[n]

df=df.rename(columns=rendict)

vlist3=['tid','user_id','pid_ciq', 'comp_rev','comp_ciq', 'name_rev', 'name_ciq',
        'altname_ciq','suffix_rev','suffix_ciq',
        'title_rev', 'title_ciq','startdate_rev','startyear_ciq',
        'enddate_rev', 'endyear_ciq', 'name_std_rev', 'name_std_ciq']
vlist=vlist3+[v for v in df.columns if v not in vlist3]

df=df[vlist]

df.to_pickle('rev_ciq_keyvar_potpairs_withscores_renamed.pkl')

#%%
title=df[['title_rev', 'title_ciq']].drop_duplicates()
for v in title.columns:
    title[v+'_lower']=title[v].str.lower()

title.to_pickle('title_pair.pkl')

tlowv=['title_rev_lower', 'title_ciq_lower']

tlow=title[tlowv].drop_duplicates()

tlow=tlow.dropna(how='any')
tlow=tlow[(tlow.title_rev_lower!='') & (tlow.title_ciq_lower!='')]

tlow=tlow.drop_duplicates().reset_index(drop=True)


a,b='title_rev_lower','title_ciq_lower' # 'pname','fullname' should be string variable name

def sr(row):
    try:
        return SequenceMatcher(None, row[a], row[b]).ratio()    
    except:
        return 0
    
tlow['tdist']=tlow.swifter.allow_dask_on_strings().progress_bar(False).apply(sr,axis=1)

tlow.to_pickle('title_pair_lower.pkl')
#%%
flist=list(glob.glob('title similarity chunk *.pkl'))
tsim=pd.concat(pd.read_pickle(file) for file in flist)
tlow=tlow.merge(tsim,on=list(tlow.columns)[:2])
title=title.merge(tlow,on=list(title.columns)[2:])
df=pd.read_pickle('rev_ciq_keyvar_potpairs_withscores_renamed.pkl')
df=df.merge(title.drop(['title_rev_lower', 'title_ciq_lower'],1),on=['title_rev', 'title_ciq'],how='left')
df=df.rename(columns={'nlpscore':'tnlp'})

df['styear_rev']=pd.to_datetime(df.startdate_rev).dt.year
df['enddate_rev']=df.enddate_rev.fillna('')
df['endyear_rev']=df.enddate_rev.str.split('-').str.get(0).astype(int,errors='ignore')

df['startyear_ciq']=df.startyear_ciq.astype(int,errors='ignore')
df['endyear_ciq']=df.startyear_ciq.astype(int,errors='ignore')

df['same_styear']=(df.styear_rev==df.startyear_ciq).astype(float)
df['same_endyear']=(df.endyear_rev==df.endyear_ciq).astype(float)

def samesuf(row):
    try:
        return float(row['suffix_rev'].lower() in row['suffix_ciq'].lower().split(','))
    except:
        return 0

df['samesuf']=df.apply(samesuf,axis=1)

vlist=['title_rev','title_ciq', 
       'startdate_rev', 'startyear_ciq', 
       'enddate_rev','endyear_ciq']

df['mistitle']=((df.title_rev.isnull()) | (df.title_rev=='')|(df.title_ciq.isnull()) | (df.title_ciq=='')).astype(float)
df['misstyear']=((df.startdate_rev=='nan') | (df.startyear_ciq.isnull())).astype(float)
df['missendyear']=((df.enddate_rev=='nan') | (df.endyear_ciq.isnull())).astype(float)

numlist=['matchlevel', 'cosim','perscore', 'pergsim1', 
         'pergsim2', 'flsim1', 'flsim2', 'cdist','nrank', 
         'nlpscore1', 'nlpscoremax', 'nlpscoremin', 'nlpscoreavg',
         'tdist', 'tnlp', 'same_styear','same_endyear', 'samesuf',
         'mistitle', 'misstyear', 'missendyear']
for v in numlist:
    df.loc[df[v].isnull(),v]=0

matched=pd.read_excel('ciq_revelio_checked_pairs.xlsx')

df=df.merge(matched[['tid','match']].rename(columns={'match':'matched'}),how='left')

df.to_pickle('rev_ciq_keyvar_potpairs_bfore_rf.pkl')    
    
#%%
"""clean sample with predicted same-person score"""
os.chdir('/nas_01/private/luguangli/capitaliq professional to revelio/rev_ciq_match_pot_pairs/')    
df=pd.read_pickle('rev_ciq_keyvar_potpairs_rfscore.pkl')    
df=df.sort_values(['pid_ciq','user_id','pred_matched'],ascending=False)
for n in range(1,10):
    print(n)
    df[f'samepc{n}']=(df.pred_matched>=(10-n)/10).astype(float)        
    df[f'samepc{n}']=df.groupby(['user_id','pid_ciq'])[f'samepc{n}'].transform('sum')    

df2=df[['pid_ciq','user_id','comp_rev', 'comp_ciq',
        'pred_matched','samepc1', 'samepc2', 'samepc3', 'samepc4',
        'samepc5', 'samepc6', 'samepc7', 
        'samepc8', 'samepc9']].sort_values(['pid_ciq','user_id','pred_matched'],ascending=False)

df2=df2.drop_duplicates(['pid_ciq','user_id','comp_rev', 'comp_ciq'])    

for n in range(1,10):
    print(n)
    df2[f'samepc{n}']=(df2.pred_matched>=(10-n)/10).astype(float) 
    df2[f'samepc{n}_2']=df2.groupby(['user_id','pid_ciq'])[f'samepc{n}'].transform('sum')    

vlist=['pid_ciq','user_id']+[v for v in df2.columns if '_2' in v]
    
df2=df2[vlist]

df2=df2.drop_duplicates()

df=df.merge(df2,on=['pid_ciq','user_id'],how='left')

df.to_pickle('rev_ciq_keyvar_potpairs_befrf2.pkl')
