#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:13:36 2023

@author: rnd
"""


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
pciq='/nas_01/private/luguangli03/match pitchbook people with ciq people/ciq'
ppit = '/nas_01/private/luguangli03/match pitchbook people with ciq people/pitchbook'
pptv = '/nas_01/private/luguangli03/match pitchbook people with ciq people/match'
os.chdir(ppit)
#%%
df_deal=pd.read_stata('pitchbook_person_company_deal.dta').drop_duplicates('person_personid')
df_deal=df_deal[[ 'person_personid','person_firstname', 'person_fullname','person_gender',
        'person_lastname','person_location','person_middlename','person_phone','person_prefix',
        'company_companyformername', 'company_companyid','company_companyname','person_biography']]

df_advisor=pd.read_stata('pitchbook_person_company_advisor.dta').drop_duplicates('person_personid')
df_advisor=df_advisor[[ 'person_personid','person_firstname', 'person_fullname','person_gender',
        'person_lastname','person_location','person_middlename','person_phone','person_prefix',
        'company_companyformername', 'company_companyid','company_companyname','person_biography']]
df=pd.concat([df_deal,df_advisor],ignore_index=True)

df_investor=pd.read_stata('pitchbook_person_company_investor.dta').drop_duplicates('person_personid')
df_investor=df_investor[[ 'person_personid','person_firstname', 'person_fullname','person_gender',
        'person_lastname','person_location','person_middlename','person_phone','person_prefix',
        'company_companyformername', 'company_companyid','company_companyname','person_biography']]
df=pd.concat([df,df_investor],ignore_index=True)

df_board=pd.read_stata('pitchbook_person_company_board.dta').drop_duplicates('person_personid')
df_board=df_board[[ 'person_personid','person_firstname', 'person_fullname','person_gender',
        'person_lastname','person_location','person_middlename','person_phone','person_prefix',
        'company_companyformername', 'company_companyid','company_companyname','person_biography']]
df=pd.concat([df,df_board],ignore_index=True)

df.drop_duplicates('person_personid')
indexNames=df[df['person_personid']=='No'].index
df.drop(indexNames,inplace=True)

df=df.rename(columns={"person_personid": "personid","company_companyid":"companyid","person_fullname":"fullname",
"company_companyname":"companyname","person_biography":"personbio"})

df.to_pickle('pitchbook_person_deal_advisor_investor_board.pkl')

#%%

df=pd.read_pickle('pitchbook_person_deal_advisor_investor_board.pkl')
person=pd.read_csv('Person.csv',error_bad_lines=False,encoding='ISO-8859-1')

person=person[['PersonID', 'FirstName', 'FullName', 'Gender', 'LastName', 'PrimaryPosition','MiddleName', 'Phone', 
'Prefix', 'PrimaryCompanyID', 'PrimaryCompany', 'Biography']].rename(columns={'PersonID':'personid', 
'FirstName':'person_firstname', 'FullName':'fullname', 'Gender':'person_gender', 'LastName':'person_lastname', 
'PrimaryPosition':'person_location','MiddleName':'person_middlename', 'Phone':'person_phone', 
'Prefix':'person_prefix', 'PrimaryCompanyID':'companyid', 'PrimaryCompany':'companyname', 'Biography':'personbio'})


comp=pd.read_csv('Company.csv',error_bad_lines=False,encoding='ISO-8859-1')

comp1=comp[['CompanyID','CompanyFormerName']].rename(columns={"CompanyID": "companyid","CompanyFormerName":"company_companyformername"})

person=person.merge(comp1,on=['companyid'],how='left',indicator=True)
person=person.drop('_merge',1)

df=pd.concat([df,person],ignore_index=True).drop_duplicates('personid')

df.to_pickle('pitchbook_person_deal_advisor_investor_board1.pkl')

comp=comp[['CompanyID', 'CompanyAlsoKnownAs', 'CompanyLegalName', 'CompanyFormerName']].rename(columns={"CompanyID": "companyid"})

df=df.merge(comp, on=['companyid'], how='left', indicator=True)




####################################

os.chdir(ppit)
df_startup = pd.read_csv('Startup_Investor_profile.csv').drop_duplicates('PersonID_investor')

#df_startup = df_startup[['CompanyID','PersonID_investor', 'person_type', 'FullName','InvestorID','InvestorName',
#'EntityID', 'EntityName', 'EntityType','CompanyName', 'CompanyAlsoKnownAs', 'CompanyFormerName','CompanyLegalName']]

df_startup = df_startup[['CompanyID','PersonID_investor', 'person_type', 'FullName','InvestorID','InvestorName',
'EntityID', 'EntityName', 'EntityType','CompanyName']]


df_startup=df_startup.rename(columns={"InvestorID": "investorid","CompanyID":"companyid","PersonID_investor":"personid"})

#%%
vcp=df.merge(df_startup,on=['personid'],how='left')

vcp=vcp.drop('_merge',1)

vcp.to_pickle('pitchbook_person_startup.pkl')

####################################










os.chdir(pciq)

ciqpro=pd.read_pickle("ciq_all.pkl")

# match with person name

vcp=vcp.rename(columns={"fullname": "pname"})

df=match_strings(ciqpro.fullname, vcp.pname, max_n_matches=3,min_similarity=0.3,ignore_index=True,number_of_processes=15)
df=df.drop_duplicates()
df.columns=['fullname','cosim','pname']

df=df[['pname','fullname','cosim']].drop_duplicates()
df=df.merge(ciqpro,on='fullname')

'''
['personid', 'person_firstname', 'fullname', 'person_gender',
       'person_lastname', 'person_location', 'person_middlename',
       'person_phone', 'person_prefix', 'company_companyformername',
       'companyid', 'companyname', 'PersonID_investor', 'person_type',
       'investorname', 'investorid', 'InvestorName','EntityID', 'EntityName', 'EntityType']
'''

"""Second: compute company names from capital iq to company names from pitchbook"""
# vlist=['LeadPartnerID', 'investorid', 'FullName','pname','InvestorName','companyname','CompanyName',
# 'CompanyAlsoKnownAs', 'CompanyFormerName','CompanyLegalName','EntityName','PrimaryInvestorType']

vcp=vcp.rename(columns={"personid": "LeadPartnerID","person_type":"PrimaryInvestorType"})






#%%

'''
vlist=['LeadPartnerID', 'investorid', 'FullName','pname','companyname','InvestorName','CompanyAlsoKnownAs',
'company_companyformername','CompanyLegalName','PrimaryInvestorType']
'''


vlist=['LeadPartnerID', 'investorid', 'FullName','pname','companyname','CompanyAlsoKnownAs',
'company_companyformername','CompanyLegalName','InvestorName','PrimaryInvestorType']

idlist=['LeadPartnerID', 'FullName','pname', 'investorid','PrimaryInvestorType','companyname']

vcp2=vcp[vlist].set_index(idlist)
vcp2.columns=pd.MultiIndex.from_tuples(('invname',n) for n in range(1,5))
vcp2=vcp2.stack().reset_index()
vcp2=vcp2.drop('level_6',1).drop_duplicates()
vcp2['miscomp']=(vcp2.invname.str.replace(' +','')=='').astype(float)
vcp2['minmis']=vcp2.groupby(['LeadPartnerID','investorid'],dropna=False).miscomp.transform('min')
vcp2=vcp2[(vcp2.minmis==1) | (vcp2.miscomp==0)]
vcp2=vcp2.drop_duplicates()

df=df.merge(vcp2,on='pname')

os.chdir(pptv)
df.to_pickle('tempfile_fuzzy_person_comp1.pkl')




def stdname(v):
    try:
        v=re.sub('(^\W+)|(\W+$)|(^the )','',basename(v.lower())).strip()
        
        vlist=[k for k in re.split('[ \.]',v) if type(k)==str and len(k)>0 and k not in ['',' ']]
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
        return basename(nv).strip()

    except:
        return v
    
df['compname_std']=df['compname'].apply(stdname)
df['invname_std']=df['invname'].apply(stdname)


### compute similarity scores of company names
def srcomp(row):
    a=row['compname_std']
    b=row['invname_std']
    vlist=[np.nan]*5
    try:
        alist=re.split('\W',a)
        alist=[k for k in alist if type(k)==str and len(k)>0]                
        blist=re.split('\W',b)
        blist=[k for k in blist if type(k)==str and len(k)>0]
           
        vlist[0]=SequenceMatcher(None, a, b).ratio() 
        vlist[1]=len([k for k in alist if k in b])/len(alist)  
        vlist[2]=len([k for k in blist if k in a])/len(blist) 
        vlist[3]=float(alist[0]==blist[0])
        vlist[4]=float(alist[:2]==blist[:2])
    except:
        pass
          
    return vlist



df['compdist']=df[['compname_std','invname_std']].apply(srcomp,axis=1) 

df['compscore']=df['compdist'].str.get(0)
df['compgdist1']=df['compdist'].str.get(1)
df['compgdist2']=df['compdist'].str.get(2)
df['samefirst1']=df['compdist'].str.get(3)
df['samefirst2']=df['compdist'].str.get(4)


### compute similarity scores of individual names: essentially similar scoring approach as before, just different 
def srper(row):
    a=re.sub(' +',' ', re.sub('(^the )|(\W)',' ',row['fullname']).lower()).strip()
    b=re.sub(' +',' ',re.sub('(^the )|(\W)',' ',row['pname']).lower()).strip()
    
    vlist=[np.nan]*4
    
    alist=a.split()
    blist=b.split()
    try:
        vlist[0]=SequenceMatcher(None, a, b).ratio() 
        vlist[1]=float((alist[0] in b and alist[-1] in b) or (blist[0] in a and blist[-1] in a)) 
        vlist[2]=float((alist[-1] in b) or (blist[-1] in a))
        vlist[3]=float((alist[0] in b) or (blist[0] in a))
    except:
        pass
        
    return vlist

df['namedist']=df[['fullname','pname']].apply(srper,axis=1) 

df['perscore']=df['namedist'].str.get(0)
df['samefl']=df['namedist'].str.get(1)
df['samel']=df['namedist'].str.get(2)
df['samef']=df['namedist'].str.get(3)

####################################

df=df.drop(['namedist','compdist'],1)

df=df.sort_values(['LeadPartnerID','personid','samefirst2','samefirst1','compscore'],ascending=False)

df['matchrank']=df.groupby(['LeadPartnerID','personid']).cumcount()+1

df=df[df.matchrank<=3]


os.chdir(pptv)
df.to_pickle('tempfile_fuzzy_person_comp.pkl')




#%%

os.chdir(pptv)
df=pd.read_pickle('tempfile_fuzzy_person_comp.pkl')


vlist=['PersonID','personid','personname','fullname','companyname','compname','invname_std', 'compname_std','compscore', 'compgdist1', 'compgdist2',
       'samefirst1', 'samefirst2', 'perscore', 'samefl', 'samel', 'samef']
       
df=df[(df.samef+df.samel>=1) & ((df.perscore>=0.8) | (df.samefl==1))]
df=df[(df.compgdist1+df.compgdist2>=1) | (df.samefirst1>=1)] 
df['max_compscore']=df.groupby(['LeadPartnerID','personid']).compscore.transform('max')
df['filter1']=((df.perscore==1) | (df.samefl==1)) & (df.max_compscore==1)

match1=df[df.filter1].drop('filter1',1)
df=df[df.filter1==False]
df['filter1']=((df.samefl==1) & (df.samefirst2==1) & (df.compname_std.str.contains('^(bank)|(university)')==False)).astype(float) 
df['max_filter1']=df.groupby(['LeadPartnerID','personid']).filter1.transform('max')
match2=df[df.max_filter1==1].drop(['filter1','max_filter1'],1)
df=df[df.max_filter1==0].drop(['filter1','max_filter1'],1)
df['filter1']=((df.samefl==1) & ((df.compgdist1==1) | (df.compgdist2==1)) & (df.compname_std.str.contains('^(bank)|(university)')==False)).astype(float) 
df['max_filter1']=df.groupby(['LeadPartnerID','personid']).filter1.transform('max')

match3=df[df.max_filter1==1].drop(['filter1','max_filter1'],1)
df=df[df.max_filter1==0].drop(['filter1','max_filter1'],1)
df['filter1']=((df.perscore>=0.9) & ((df.compgdist1==1) | (df.compgdist2==1) | (df.samefirst2==1) | (df.compscore>=0.9)) & (df.compname_std.str.contains('^(bank)|(university)')==False)).astype(float) 
df['max_filter1']=df.groupby(['LeadPartnerID','personid']).filter1.transform('max')

match4=df[df.max_filter1==1].drop(['filter1','max_filter1'],1)
df=df[df.max_filter1==0].drop(['filter1','max_filter1'],1)
df['filter1']=((df.compgdist1==1) | (df.compgdist2==1) |  (df.samefirst1==1) | (df.compscore>=0.9) ).astype(float) 
df['max_filter1']=df.groupby(['LeadPartnerID','personid']).filter1.transform('max')

match5=df[df.max_filter1==1].drop(['filter1','max_filter1'],1)
df=df[df.max_filter1==0].drop(['filter1','max_filter1'],1)
df['filter1']=((df.compgdist1+df.compgdist2>=1.5) | (df.compscore>=0.8) ).astype(float) 
df['max_filter1']=df.groupby(['LeadPartnerID','personid']).filter1.transform('max')

match6=df[df.max_filter1==1].drop(['filter1','max_filter1'],1)
df=df[df.max_filter1==0].drop(['filter1','max_filter1'],1)

matchlist=[match1,match2,match3,match4,match5,match6]
n=0
for matchdf in matchlist:
    n+=1
    matchdf['matchcode']=n

df=pd.concat(matchlist+[df])
df.loc[df.matchcode.isnull(),'matchcode']=7
df.loc[(df.matchcode==4) & ((df.compname_std.str.split().str.len()==1) | (df.invname_std.str.split().str.len()==1)),'matchcode']=5
df['tid']=df.reset_index(drop=True).index
vlist=['name', 'fullname', 'personid','companyid', 'compname', 'comptype','LeadPartnerID', 'investorid','PrimaryInvestorType']
df=df[vlist]
df.to_pickle('tempfile_fuzzy_person_comp_.pkl')

angelmatch=df[df.PrimaryInvestorType.isin(['angel'])]
angelmatch.to_pickle('angel_match_ciq.pkl')


###################################

###################################

os.chdir(pptv)
df=pd.read_pickle('tempfile_fuzzy_person_comp.pkl')
# df=df[(df.PrimaryInvestorType.isin(['Angel'])==False) & (df.name_in_inv!=1)]
df=df[df.PrimaryInvestorType.isin(['angel'])==False]

vlist=['LeadPartnerID','personid','pname','fullname','invname','compname','invname_std', 'compname_std','compscore', 'compgdist1', 'compgdist2',
       'samefirst1', 'samefirst2', 'perscore', 'samefl', 'samel', 'samef']
# temp=df[vlist].merge(df[['LeadPartnerID','personid']].drop_duplicates().sample(100),on=['LeadPartnerID','personid'])

"""after checking, requiring a set of minimum person match scores """
df=df[(df.samef+df.samel>=1) & ((df.perscore>=0.8) | (df.samefl==1))] #either first name match or last name match, and overall person name match score>=0.6

"""after checking, requiring a set of minimum company match scores """
df=df[(df.compgdist1+df.compgdist2>=1) | (df.samefirst1>=1)] 

# df.LeadPartnerID.nunique() #87588 unique vc/pe lead partners

"""code below will assign matchcode based on matching scores for person names and company names, respectively"""
df['max_compscore']=df.groupby(['LeadPartnerID','personid']).compscore.transform('max')

df['filter1']=((df.perscore==1) | (df.samefl==1)) & (df.max_compscore==1)

match1=df[df.filter1].drop('filter1',1)
# match1.LeadPartnerID.nunique() ###66867 can be identified in this way

df=df[df.filter1==False]

df['filter1']=((df.samefl==1) & (df.samefirst2==1) & (df.compname_std.str.contains('^(bank)|(university)')==False)).astype(float) 

df['max_filter1']=df.groupby(['LeadPartnerID','personid']).filter1.transform('max')

match2=df[df.max_filter1==1].drop(['filter1','max_filter1'],1)


df=df[df.max_filter1==0].drop(['filter1','max_filter1'],1)

df['filter1']=((df.samefl==1) & ((df.compgdist1==1) | (df.compgdist2==1)) & (df.compname_std.str.contains('^(bank)|(university)')==False)).astype(float) 
df['max_filter1']=df.groupby(['LeadPartnerID','personid']).filter1.transform('max')

match3=df[df.max_filter1==1].drop(['filter1','max_filter1'],1)

df=df[df.max_filter1==0].drop(['filter1','max_filter1'],1)



df['filter1']=((df.perscore>=0.9) & ((df.compgdist1==1) | (df.compgdist2==1) | (df.samefirst2==1) | (df.compscore>=0.9)) & (df.compname_std.str.contains('^(bank)|(university)')==False)).astype(float) 
df['max_filter1']=df.groupby(['LeadPartnerID','personid']).filter1.transform('max')

match4=df[df.max_filter1==1].drop(['filter1','max_filter1'],1)

df=df[df.max_filter1==0].drop(['filter1','max_filter1'],1)

df['filter1']=((df.compgdist1==1) | (df.compgdist2==1) |  (df.samefirst1==1) | (df.compscore>=0.9) ).astype(float) 
df['max_filter1']=df.groupby(['LeadPartnerID','personid']).filter1.transform('max')

match5=df[df.max_filter1==1].drop(['filter1','max_filter1'],1)

# match1['LeadPartnerID'].nunique()+match2['LeadPartnerID'].nunique()+match3['LeadPartnerID'].nunique()+match4['LeadPartnerID'].nunique()+match5['LeadPartnerID'].nunique()


df=df[df.max_filter1==0].drop(['filter1','max_filter1'],1)

df['filter1']=((df.compgdist1+df.compgdist2>=1.5) | (df.compscore>=0.8) ).astype(float) 
df['max_filter1']=df.groupby(['LeadPartnerID','personid']).filter1.transform('max')

match6=df[df.max_filter1==1].drop(['filter1','max_filter1'],1)

df=df[df.max_filter1==0].drop(['filter1','max_filter1'],1)

matchlist=[match1,match2,match3,match4,match5,match6]
n=0
for matchdf in matchlist:
    n+=1
    matchdf['matchcode']=n

df=pd.concat(matchlist+[df])

df.loc[df.matchcode.isnull(),'matchcode']=7

### after checking, need to change the matchcode below
df.loc[(df.matchcode==4) & ((df.compname_std.str.split().str.len()==1) | (df.invname_std.str.split().str.len()==1)),'matchcode']=5

df['tid']=df.reset_index(drop=True).index

df.to_pickle('tempfile_fuzzy_person_comp_nonangel.pkl')


################################################################
################################################################


keylist=['LeadPartnerID','personid','pname', 'fullname','invname','compname']
df.loc[df.matchcode.isin([5,6]),['tid','matchcode']+keylist].to_excel('matchcode_5_6.xlsx',index=False)
df.loc[df.matchcode.isin([7]),['tid','matchcode']+keylist].to_excel('matchcode_7.xlsx',index=False)
df.loc[df.matchcode<=4,'LeadPartnerID'].nunique() #81414 unique pitchbook partners can be matched to capital iq
df=pd.read_pickle('tempfile_fuzzy_person_comp_nonangel.pkl')


match56=pd.read_excel('matchcode_5_6.xlsx')

# match56=match56[match56.MATCH.isnull()].drop('MATCH',1) #only keep valid match, non-missing MATCH indicates bad match

df=df.merge(match56['tid'].drop_duplicates(),on='tid',how='left',indicator=True)

df=df[(df.matchcode.isin([5,6])==False) | (df._merge=='both')].drop('_merge',1)

match7=pd.read_excel('matchcode_7.xlsx')

# match7=match7[match7.MATCH==1].drop('MATCH',1)  #only keep valid match

df=df.merge(match7.tid.drop_duplicates(),on='tid',how='left',indicator=True)

df=df[(df.matchcode!=7) | (df._merge=='both')]

# df.PrimaryInvestorType.value_counts() #
df=df.sort_values(['personid','matchcode','compscore'],ascending=[True,True,False]).drop_duplicates(['personid'])

df=df.sort_values(['LeadPartnerID','matchcode','compscore'],ascending=[True,True,False]).drop_duplicates(['LeadPartnerID'])

vlist=['pname', 'fullname', 'personid','companyid', 'compname', 'comptype','LeadPartnerID', 'investorid','PrimaryInvestorType']
df=df[vlist]

df.to_pickle('pitchbook_capitaliq_non_angel_final_match.pkl')


################################################################
################################################################

os.chdir(ppit)
vcp=pd.read_pickle('pitchbook_person_startup.pkl')
vcp=vcp.rename(columns={"fullname": "pname"})
vcp=vcp.rename(columns={"personid": "LeadPartnerID","person_type":"PrimaryInvestorType","investorid":"InvestorID"})
vcp=vcp.drop_duplicates(['LeadPartnerID','InvestorID'])



os.chdir(pptv)
angel=pd.read_pickle('angel_match_ciq.pkl')
angel=angel.rename(columns={"investorid":"InvestorID"})

angel=angel.merge(vcp[['InvestorID','LeadPartnerID','personbio']].drop_duplicates(),on=['InvestorID','LeadPartnerID']) ### forgot to ad description 




angel['compstd']=angel.compname.str.lower().apply(basename).apply(basename).str.replace('^ ?the ','').str.replace('\W',' ').str.replace(' +',' ').str.strip()
angel['bio']=angel.personbio.str.lower().str.replace('\W',' ').str.replace(' +',' ')

#%%
def inbio(row):
    res=[np.nan]*3
    alist=row['compstd'].split()
    b=row['bio']
    try:
        if len(alist)>=1:
            res[0]=alist[0] in b
            res[2]=len([v for v in alist if v in b])/len(alist)
        if len(alist)>=2:
            res[1]=' '.join(alist[:2]) in b
        return res
    except:
        print(111)

angel['inbio']=angel.apply(inbio,axis=1)    
angel['inbio1']=angel.inbio.str.get(0).astype(float)    
angel['inbio2']=angel.inbio.str.get(1).astype(float)     
angel['inbio3']=angel.inbio.str.get(2)    

angel['biomax']=np.nanmax(angel[['inbio1','inbio2','inbio3']],axis=1)

angel=angel[angel.biomax>0]

angel=angel[(angel.inbio1==1) & ((angel.inbio2==1) | (angel.inbio3>=2/3))]

angel['biomin']=np.nanmin(angel[['inbio1','inbio2','inbio3']],axis=1)



def srper(row):
    a=re.sub(' +',' ', re.sub('(^the )|(\W)',' ',row['fullname']).lower()).strip()
    b=re.sub(' +',' ',re.sub('(^the )|(\W)',' ',row['name']).lower()).strip()
    
    vlist=[np.nan]*4
    
    alist=a.split()
    blist=b.split()
    try:
        vlist[0]=SequenceMatcher(None, a, b).ratio() 
        vlist[1]=float((alist[0] in b and alist[-1] in b) or (blist[0] in a and blist[-1] in a)) 
        vlist[2]=float((alist[-1] in b) or (blist[-1] in a))
        vlist[3]=float((alist[0] in b) or (blist[0] in a))
    except:
        pass
        
    return vlist

angel['namedist']=angel[['fullname','name']].apply(srper,axis=1) 
angel['perscore']=angel['namedist'].str.get(0)
angel['samefl']=angel['namedist'].str.get(1)
angel['samel']=angel['namedist'].str.get(2)
angel['samef']=angel['namedist'].str.get(3)

angel=angel.drop('namedist',1)    

angel['tid']=angel.reset_index(drop=True).index

angel=angel[(angel.samef+angel.samel>=1) & ((angel.perscore>=0.8) | (angel.samefl==1))] #either first name match or last name match, and overall person name match score>=0.6

angel.to_pickle('angel_potential_matched.pkl')

angel.loc[(angel.samel==0) | (angel.samef==0) | (angel.perscore<=0.9) | (angel.biomin<1),['tid','personid','LeadPartnerID','fullname','name','compname','personbio']].to_excel('angel_match_to_check.xlsx',index=False)


################################################################
################################################################

angel_match=pd.read_excel('angel_match_to_check.xlsx')
# angel_match['matched']=(angel_match.person_comp==1).astype(float)

angel=pd.read_pickle('angel_potential_matched.pkl')

angel=angel.merge(angel_match[['tid']],on=['tid'],how='left',indicator=True)

angel=angel[(angel._merge!='both')]

angel=angel.sort_values(['personid','biomin','samefl','perscore'],ascending=[True,False,False,False]).drop_duplicates(['personid'])

angel=angel.sort_values(['LeadPartnerID','biomin','samefl','perscore'],ascending=[True,False,False,False]).drop_duplicates(['LeadPartnerID'])

vlist=['name', 'fullname', 'personid','companyid', 'compname', 'comptype','LeadPartnerID', 'InvestorID','PrimaryInvestorType']
angel=angel[vlist]

angel.to_pickle('pitchbook_capitaliq_angel_final_match.pkl')







################################################################
################################################################


df=pd.read_pickle('pitchbook_capitaliq_angel_final_match.pkl')
angel=pd.read_pickle('pitchbook_capitaliq_non_angel_final_match.pkl')

df=df.append(angel).reset_index(drop=True)

df['angel']=(df.PrimaryInvestorType=='Angel (individual)').astype(float)

df=df.sort_values(['personid','angel'],ascending=False).drop_duplicates('personid').drop('angel',1)

df['matchround']=2

# df2=pd.read_pickle('pitchbook_ciq_matched.pkl') #get in prior match

# df2=df2.rename(columns={'companytypename':'comptype'})

# vlist=[v for v in df2.columns if v in df.columns]

# df2['matchround']=1
# df=df.append(df2[vlist]).drop_duplicates(['LeadPartnerID','personid'])

df=df.sort_values(['personid','matchround'],ascending=False).drop_duplicates(['personid'])

df=df.sort_values(['LeadPartnerID','matchround'],ascending=False).drop_duplicates(['LeadPartnerID'])

# df.matchround.value_counts() #correct, all are matchround2, implying that the second round match was complete

df=df.drop('matchround',1)

df.to_csv('pitchbook_capitaliq_final_match.csv',index=False)

df.to_pickle('pitchbook_capitaliq_final_match.pkl')