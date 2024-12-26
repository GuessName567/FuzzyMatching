import os
import pandas as pd
import numpy as np
import time as tm
from cleanco import basename
from difflib import SequenceMatcher
import re
import glob
from string_grouper import  match_most_similar ,match_strings, group_similar_strings, compute_pairwise_similarities, StringGrouper
import swifter

def isid(data, variables):
    dup = data.duplicated(variables)
    dupsum = dup.describe()
    if (dupsum['unique'] == 1):
        print(str(variables) + " uniquely identify this dataset")
    else:
        print(str(variables) + " Do NOT uniquely identify this dataset")

pciq='/nas_01/private/luguangli03/match pitchbook people with ciq people/ciq'
ppit = '/nas_01/private/luguangli03/match pitchbook people with ciq people/pitchbook'

"""match with personid from capital iq to get indicator on company type and alive professionals"""
### professional data
os.chdir(pciq)
ciqpro=pd.read_csv('ciq_professional.csv',error_bad_lines=False,encoding='ISO-8859-1')
isid(ciqpro,['companyid','personid','proid'])

### basic person information
ciqper=pd.read_csv('ciq_person.csv',error_bad_lines=False,encoding='ISO-8859-1')
isid(ciqper,'personid')
ciqper.personid.nunique()

### bio information if available
ciqbio=pd.read_csv('ciq_person_bio.csv',error_bad_lines=False,encoding='ISO-8859-1')
ciqbio=ciqbio[ciqbio.personid.str.contains('^[0-9]+$')]
ciqbio['personid']=ciqbio['personid'].astype(float)
ciqper=ciqper.merge(ciqbio,on='personid',how='outer',indicator=True)
ciqper=ciqper[ciqper._merge!='right_only'].drop('_merge',1)
# _merge是自动生成的

### merge professional data with basic person information
ciqpro=ciqpro.merge(ciqper,on='personid',how='outer',indicator=True)

ciqpro=ciqpro.drop('_merge',1)

#这里注意，合并两个df时，名字一样的列例如yearborn在合并之后会分别变成yearbor_x和yearborn_y
ciqpro.loc[ciqpro.yearborn_y.notna(),'yearborn_x']=ciqpro.loc[ciqpro.yearborn_y.notna(),'yearborn_y']
#选取yearborn_y列不为空的行，将其yearborn_x位置的数据替换成yearborn_y的数据

ciqpro=ciqpro.drop('yearborn_y',1).rename(columns={'yearborn_x':'yearborn'})

ciqpro=ciqpro.sort_values(['personid','proid','profunctionid'])
#按照['personid','proid','profunctionid']的顺序排序

ciqpro=ciqpro.rename(columns={'companyname':'compname','personname':'name','profunctionid':'profuncid','profunctionname':'profuncname',
                              'yearfounded':'founded','firstname':'fname', 'middlename':'mname', 'lastname':'lname','emailaddress':'email','phonevalue':'phone'})
#经查看，此数据集含有以上所有列

vlist1=['personid','proid','profuncid','fname','lname','mname','prefix','suffix','companyid','compname','biography']
vlist=vlist1+[v for v in ciqpro.columns if v not in vlist1]

ciqpro=ciqpro[vlist]

### get in companytype 
ciqcomp=pd.read_csv('ciq_compinfo.csv',error_bad_lines=False,encoding='ISO-8859-1')
ctype=pd.read_csv('ciq_companytype.csv',error_bad_lines=False,encoding='ISO-8859-1')
ciqcomp=ciqcomp.merge(ctype,on='companytypeid')

rendict={'companytypename':'comptype','countryid':'comp_countryid'}
ciqpro=ciqpro.merge(ciqcomp[['companyid','companytypename','countryid']].rename(columns=rendict).drop_duplicates('companyid'),on='companyid',how='left',indicator=True)
# ciqpro.compname.isnull().value_counts()

ciqpro=ciqpro.drop('_merge',1)

os.chdir(pciq)
ciqpro.to_pickle('ciqpro_linkedto_person_comptype.pkl')
#与death无关


# os.chdir(pciq)
# ciqpro=pd.read_pickle('ciqpro_linkedto_person_comptype.pkl')

ciqpro=ciqpro[['personid', 'name', 'fname', 'lname', 'mname', 'prefix', 'suffix','companyid', 'compname', 'comptype']].drop_duplicates()

ctypelist= ['Private Investment Firm','Public Investment Firm','Financial Service Investment Arm','Corporate Investment Arm',
            'Private Fund','Public Company','Private Company','Government Institution','Public Fund','Educational Institution',
            'Trade Association','Foundation/Charitable Institution','GCP Industry','Industry', 'Assets/Products']
otherlist=[c for c in ciqpro.comptype.unique() if c not in ctypelist]

ciqpro=ciqpro[ciqpro.comptype.isin(ctypelist)]
ciqpro.personid.nunique() ### 

                                                  
ciqpro.personid.nunique() ###
for v in ['fname','mname','lname']:
    ciqpro[v]=ciqpro[v].fillna('')
ciqpro['fullname']=ciqpro['fname']+' '+ciqpro['mname']+' '+ciqpro['lname']
ciqpro['fullname']=ciqpro['fullname'].str.replace(' +',' ').str.strip().str.lower()


ciqpro.to_pickle('ciq_all.pkl')

# temp=ciqpro.sample(1000)