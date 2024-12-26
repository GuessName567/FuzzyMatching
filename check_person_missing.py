#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:10:18 2023

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

all=pd.read_pickle('pitchbook_person_deal_advisor_investor_board1.pkl')
vlist=['personid']
all=all[vlist]


os.chdir(pptv)

df=pd.read_pickle('pitchbook_capitaliq_final_match.pkl')
df=df[['LeadPartnerID']]
df=df.rename(columns={'LeadPartnerID':'personid'})



all=all.merge(df,on=['personid'],how='left',indicator=True)
missing=all[all._merge=='left_only']


missing.to_pickle('Person_not_match.pkl')
