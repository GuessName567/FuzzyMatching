import os
import numpy as np
import pandas as pd
import re
import datetime as dt
import time as tm
import gc
import zipfile
import glob
import collections

gc.enable()
perc=[0.005, 0.01,0.025, 0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975, 0.99,0.995] ### used when checking the summary stats of a data
import textdistance as td


def ratcliff_similarity(row, varname1, varname2):
    try:
        return td.ratcliff_obershelp(basename(row[varname1].lower()), basename(row[varname2].lower()))
    except:
        return np.nan

from difflib import SequenceMatcher

def similiarity_ratio(row, var1, var2):
    try:
        return SequenceMatcher(None, basename(basename(row[var1].lower())),
                               basename(basename(row[var2].lower()))).ratio()
    except:
        return np.nan

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

pbright='D:/Data/Bright'
os.chdir(pbright)
#%% basic standardization1: standardize names to get first/middle/last names
flist=glob.glob('bright_linkedin_individual_profile_keyvar_file*pkl')
df=pd.DataFrame()
for file in flist:
    print(file)
    temp=pd.read_pickle(file)[['id','name']].drop_duplicates()
    df=df.append(temp)
isid(df,'id') #correct

df = df[df['name'].str.contains('[a-zA-Z]')] #throw away the other that contains no letters, which are basically emply string

df['spelet']=df.name.str.replace(' +','').str.extract('([\W]|[\d])') #non-word & non-space letters
df['has_spelet']=df.name.str.replace(' +','').str.contains('([\W]|[\d])') #non-word & non-space letters
df['length']=df.name.str.split().str.len()

df.to_pickle('bright_nameonly.pkl')

#%%
"""standardize names by dealing with special letters & names that have multiple sections: mainly, shikun's code"""
df=pd.read_pickle('bright_nameonly.pkl')

df['has_n_upper']=df.name.str.contains('[A-Z][A-Z]+').astype(float)

name=df[(df.length>=4) | (df.has_spelet==True) | (df.has_n_upper==True)] #either has special letters or has at least 4 parts
name.to_pickle('special_names.pkl')

#%%
name=pd.read_pickle('special_names.pkl')
isid(name,'id')
"""Deal with cases of long names with no special letters"""
# name6=name[(name.length>=6) & (name.has_spelet==False)]
# name6['namestd']=name6['name'].str.replace('( [A-Z]+)+$','')
# name6['namestd']=name6['namestd'].str.replace(' [A-Z]+ [A-Za-z]+$','')
# name6['length']=name6['namestd'].str.split().str.len()
# name6=name6[name6.length.between(2,4)]
# name6.to_pickle('idv_name_long_no_special_letter.pkl')
# name=name[(name.length<6) | (name.has_spelet==True)]

name['namestd']=name.name

# get rid of strings after the first upper letter group: they are most likely useless suffixes indicating education degreee or job title
name['filter1']=name.namestd.str.contains('^([A-Z]\.)?([A-Z][a-z]+).*([A-Z][a-z]+).*,?[A-Z][A-Z]+.*').astype(float)
name.loc[name['filter1']==1,'namestd']=name.loc[name['filter1']==1,'name'].str.replace(',? ?[A-Z][A-Z]+.*','').str.replace('( ?|-?|\(?|( ?[0-9]+)?)$','')

#override filter1 for cases where upper letter group is in the middle of first name and last name
name['filter2']=name.name.str.contains('^([A-Z][a-z]+).(\(|\')?[A-Z][A-Z]+(\)|\')?(.?([A-Z][a-z]+))+').astype(float)
name1=name[name.filter2==1].reset_index(drop=True)
# name.loc[name.filter2==1,'namestd']=name.loc[name.filter2==1,'name'].str.extract('(^([A-Z][a-z]+).(\(|\')?[A-Z][A-Z]+(\)|\')?(.([A-Z][a-z]+))+)',expand=True)
temp=name1.name.str.extract('(^([A-Z][a-z]+).(\(|\')?[A-Z][A-Z]+(\)|\')?(.?([A-Z][a-z]+))+).*',expand=False)

name1=name1.merge(temp[0],left_index=True,right_index=True,how='left')

name=name.merge(name1[['id',0]],on='id',how='left')

name=name.drop(0,1)

### Get back suffixes that are deleted
# temp=name[(name.filter1+name.filter2==1) ].sample(1000)
suflist=['I\.?I\.?I\.?','I\.?I\.?','I\.?V\.?','V\.?I\.?','V\.?I\.?I\.?','J\.?R\.?','S\.?R\.?']
sufstr='((\W)'+'($|\W))|((\W)'.join(suflist)+'(\W|$))'
for v in suflist:
    print('(\W)'+v+'(\W|$)')
sufdict={}
for v in suflist:
    sufdict[v]=re.sub('\W','',v)
# name2=name[name.name.str.contains(sufstr)]
for v in suflist:
    name.loc[name.name.str.contains('(\W)'+v+'(\W|$)'),'suffix']=sufdict[v]

# temp=name[name.suffix.notna()].sample(1000)
# temp=name[(name.filter1+name.filter2==1) ].sample(1000)
# temp=name[(name.filter1+name.filter2==0) & (name.has_n_upper==1) & (name.spelet.isnull()==True)].sample(1000)

# After checking above, no need to deal with upper case group anymore
name['has_spelet']=name.namestd.str.replace(' +','').str.contains('([\W]|[\d])')
name['spelet']=name.namestd.str.replace(' +','').str.extract('([\W]|[\d])')
#name.has_spelet.value_counts()
name[name.has_spelet==False].to_pickle('name_clean1.pkl')

name[name.has_spelet==True].to_pickle('name_spelet.pkl')
#%%
name=pd.read_pickle('name_spelet.pkl')
name=name[name.has_spelet==True]
name['namestd']=name['namestd'].str.replace('(',' (').str.replace(')',') ').str.replace(' +',' ').str.strip()

name['nos']=name.groupby('spelet').namestd.transform('count')
temp=name[name.nos>=100].groupby('spelet').sample(100).sort_values('nos',ascending=False)
# temp.to_excel('spelet_check.xlsx',index=False)

name['comma']=name.namestd.str.contains(',').astype(float)
name['filter3']=name.namestd.str.contains('([A-Z][a-z]+){1,2}, ?([A-Z][a-z]+){1,2}').astype(float)
name['filter4']=name.namestd.str.contains('([A-Z][a-z]+){1,2}, ?([A-Z][a-z]+)([A-Z]|\.)').astype(float)
### below are cases that we need to extract the first part before , for sure
name.loc[(name.comma==1) & (name.filter3==0) & (name.namestd.str.contains('[a-z]')==True),'namestd']=name.loc[(name.comma==1) & (name.filter3==0) & (name.namestd.str.contains('[a-z]')==True),'namestd'].str.split(',').str.get(0).str.strip()
name.loc[(name.filter3==1) & (name.filter4==1),'namestd']=name.loc[(name.filter3==1) & (name.filter4==1),'namestd'].str.split(',').str.get(0).str.strip()

# temp=name.loc[(name.comma==1) & (name.filter3==0),['name','namestd']].sample(100)
# temp=name.loc[(name.filter3==1) & (name.filter4==1),['name','namestd']].sample(100)
# temp=name.loc[(name.filter3==1) & (name.filter4==0),['name','namestd']].sample(100)

remlist=['Esq\.','\(she/her\)','\(She/Her\)','\(she/her/hers\)','\(She/Her/Hers\)','\(he/him\)','\(He/Him\)','\(he/him/his\)','\(He/Him/His\)']
remstr='('+')|('.join(remlist)+')'
name['namestd']=name.namestd.str.replace(remstr,' ').str.replace(' +',' ').str.strip()

name['l1']=name.namestd.str.split(' |,').str.get(-1)

remlist=['Ph\.D\.?','M\.?D\.?','P\.?E\.?','M\.?Ed\.?','M\.?A\.?','☁','☁','MBA','Ed\.?D\.?','D\.?r\.?']
remstr='((,| )'+('$)|((,| )').join(remlist)+'$)'

name['namestd']=name.namestd.str.replace(remstr,'').str.replace('\W$','')

name['filter5']=(name.namestd.str.contains('\(')==True) & (name.namestd.str.contains('\)')==False)
name.loc[name.filter5,'namestd']=name.loc[name.filter5,'namestd']+')'

name['filter6']=name.namestd.str.lower().str.contains('(\(she.*/.*\))|(\(she\))|(\(he\))|(\(he.*/.*\))|(\(they)')
name.loc[name.filter6==True,'namestd']=name.loc[name.filter6==True,'namestd'].str.replace('\(.*\)','')

name['has_spelet']=name.namestd.str.replace('( +)|(\')|(\")|(-)|(\.)|(\()|(\)|(,))','').str.contains('([\W]|[\d])')
name['spelet']=name.namestd.str.replace('( +)|(\')|(\")|(-)|(\.)|(\()|(\)|(,))','').str.extract('([\W]|[\d])')

name.loc[name.has_spelet==True,'namestd']=name.loc[name.has_spelet==True,'namestd'].str.replace('\W',' ')
name['namestd']=name['namestd'].str.replace(' +',' ').str.strip()

name['namestd']=name['namestd'].str.replace('(Psy\.D$)|(Pharm ?D$)|(PhD$)|(B\.S\.A\.$)|(Certified$)','').str.strip(',')

name1=pd.read_pickle('name_clean1.pkl')

vlist=[v for v in name.columns if v in name1.columns]

name=name[vlist].append(name1[vlist]).reset_index(drop=True)

name['upp']=name.name.str.upper()


suflist=['I','I\.?I\.?I\.?','I\.?I\.?','I\.?V\.?','V\.?I\.?','V\.?I\.?I\.?','J\.?R\.?','S\.?R\.?']
sufstr='((\W)'+'($|\W))|((\W)'.join(suflist)+'(\W|$))'

for v in suflist:
    print('(\W)'+v+'(\W|$)')
sufdict={}
for v in suflist:
    sufdict[v]=re.sub('\W','',v)
# name2=name[name.name.str.contains(sufstr)]
for v in suflist:
    name.loc[name.upp.str.contains('(\W)'+v+'(\W|$)'),'suffix']=sufdict[v]

name=name[['id','name','namestd','suffix']]
name[['id','name','namestd','suffix']].to_pickle('name_clean.pkl')

# get rid of intermediate inputs
removelist=['name_spelet.pkl','name_clean1.pkl','special_names.pkl','parts.xlsx','idv_name_long_no_special_letter.pkl']
for file in removelist:
    try:
        os.remove(file)
    except:
        pass

#%%
df=pd.read_pickle('bright_nameonly.pkl')
df=df.merge(name,on=['id','name'],how='left',indicator=True)
# df._merge.value_counts() #correct
df=df.drop('_merge',1).rename(columns={'suffix':'suffix1'})
df.loc[df.namestd.isnull(),'namestd']=df.loc[df.namestd.isnull(),'name']
df.to_pickle('bright_nameonly_befparser.pkl')
#%%
df=pd.read_pickle('bright_nameonly_befparser.pkl')

df=pd.DataFrame(df['namestd'].drop_duplicates())

from nameparser import HumanName as hn

def parse_name(x):
    try:
        l=[np.nan]*5
        y=hn(x)
        try:
            l[0]=y.last
        except:
            pass
        try:
            l[1]=y.first
        except:
            pass
        try:
            l[2]=y.middle
        except:
            pass
        try:
            l[3]=y.suffix
        except:
            pass
        try:
            l[4]=y.nickname
        except:
            pass
        return l
    except:
        return  [np.nan]*5

num=10

split_dfs = np.array_split(df, num)

for n in range(num):
    t1=tm.time()
    print('Start processing part {}'.format(n+1))
    tdf=split_dfs[n]
    tdf[['lname','fname','mname','suffix','nickname']]=tdf['namestd'].map(parse_name).apply(pd.Series)
    tdf.to_pickle('bright_name_clean{}.pkl'.format(n+1))
    t2=tm.time()
    print('Part {} processed, and it takes {} minutes'.format(n+1,int((t2-t1)/60) ))

#%%
flist=glob.glob('bright_name_clean*.pkl')
name=pd.concat(pd.read_pickle(file) for file in flist)
# isid(name,'namestd')
df=pd.read_pickle('bright_nameonly_befparser.pkl')

df=df.merge(name,on='namestd',how='outer',indicator=True)
df=df.drop('_merge',1)

vlist=name.columns[1:]
for v in vlist:
    print(v)
    df[v]=df[v].str.lower().str.replace('\W',' ').str.replace(' +',' ').str.strip()

df.loc[df.suffix.isnull(),'suffix']=df.loc[df.suffix.isnull(),'suffix1'].str.lower()

df=df[['id', 'name', 'namestd', 'lname', 'fname', 'mname', 'suffix', 'nickname']]

df.to_pickle('bright_name_final.pkl')

for file in flist:
    os.remove(file)

#%%
ndf=pd.read_pickle('bright_name_final.pkl').drop('name',1)

ndf['fl'] = ndf['lname'].str.slice(stop=1)
ndf.loc[ndf['fl'].str.contains('[a-z]') == False, 'fl'] = 'other'

letterlist=list(ndf.fl.unique())

flist=[f'bright_linkedin_individual_profile_keyvar_file{n}.pkl' for n in range(1,41)]

for file in flist:
    t1=tm.time()
    print('Start processing',file)
    edf = pd.read_pickle(file)
    edf=edf.merge(ndf,on='id',how='left')
    for fl in letterlist:
        edf[edf.fl==fl].drop('fl',1).to_pickle(file.split('.')[0]+f'_letter_{fl}.pkl')
    t2=tm.time()
    print(f'{file} processed, and it takes {int(t2-t1)} seconds')
del edf
del ndf
#%%
for fl in letterlist:
    print(f'processing {fl} now...')
    flist=glob.glob(f'bright_linkedin_individual_profile_keyvar_file*_letter_{fl}.pkl')
    df=pd.concat(pd.read_pickle(file) for file in flist).reset_index(drop=True)
    df.to_pickle(f'bright_linkedin_individual_profile_keyvar_letter_{fl}.pkl')
    for file in flist:
        os.remove(file)
# #%%check validilty
# df=pd.DataFrame()
# for file in glob.glob('bright_linkedin*letter*pkl'):
#     temp=pd.read_pickle(file)[['id','name']].drop_duplicates()
#     df=df.append(temp)  
# isid(df,'id')


