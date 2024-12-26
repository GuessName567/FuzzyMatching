# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:26:45 2022

@author: SME-GuangliLu
"""

import os 
import pandas as pd
os.chdir('C:/Users/SME-GuangliLu/Downloads/Research-Liu')
need_merge=pd.read_csv('individual_name_similarity.csv')
for_merge=pd.read_csv('individual_name_similarity_first_sim.csv')
need_merge=need_merge.drop(['flsim','flsim1','flsim2','anchor','Unnamed: 0'],axis=1)
final=pd.merge(need_merge,for_merge[['left_lname','similarity','right_oname','left_fullname',
    'right_fullname','flsim','flsim1','flsim2']],on=['left_lname','similarity','right_oname',
    'left_fullname','right_fullname'],how='left')

final['score_sum']=final.pergsim1+final.pergsim2+final.flsim1+final.flsim2#用于排序分组，无意义


summary=pd.DataFrame(final[['pergsim1','pergsim2','flsim1','flsim2']].value_counts())
summary.columns=['freq']
summary=summary.reset_index()
summary['score_sum']=summary.pergsim1+summary.pergsim2+summary.flsim1+summary.flsim2#用于排序，无意义
summary=summary.sort_values(by=['score_sum','pergsim1','pergsim2','flsim1','flsim2'],ascending=False)

#对final data and summary的similarity 进行分组
#总分大于3.5
final.loc[final.score_sum>=3.5,'level']=1
#总分在3-3.5之间且pergsim和flsim至少各有一个1
final.loc[(final.score_sum<3.5)&(final.score_sum>=3.0)&((final.pergsim1==1)|\
   (final.pergsim2==1))&((final.flsim1==1)|(final.flsim2==1)),'level']=2
#总分在3-3.5之间但pergsim或flsim只有一边有1
final.loc[(final.score_sum<3.5)&(final.score_sum>=3.0)&\
          (((final.pergsim1!=1)&(final.pergsim2!=1)&((final.flsim1==1)|(final.flsim2==1)))|\
           (((final.pergsim1==1)|(final.pergsim2==1))&(final.flsim1!=1)&(final.flsim2!=1))) ,'level']=3
#总分在3-3.5且没有1
final.loc[(final.score_sum<3.5)&(final.score_sum>=3.0)&(final.pergsim1!=1)&\
   (final.pergsim2!=1)&(final.flsim1!=1)&(final.flsim2!=1),'level']=4
#总分在2-3之间且pergsim和flsim至少各有一个1
final.loc[(final.score_sum<3.0)&(final.score_sum>=2.0)&((final.pergsim1==1)|\
   (final.pergsim2==1))&((final.flsim1==1)|(final.flsim2==1)),'level']=5
#总分在2-3之间但pergsim或flsim只有一边有1  
final.loc[(final.score_sum<3.0)&(final.score_sum>=2.0)&\
          (((final.pergsim1!=1)&(final.pergsim2!=1)&((final.flsim1==1)|(final.flsim2==1)))|\
           (((final.pergsim1==1)|(final.pergsim2==1))&(final.flsim1!=1)&(final.flsim2!=1))) ,'level']=6
#总分在2-3且没有1
final.loc[(final.score_sum<3.0)&(final.score_sum>=2.0)&(final.pergsim1!=1)&\
   (final.pergsim2!=1)&(final.flsim1!=1)&(final.flsim2!=1),'level']=7
#总分小于2但有1
final.loc[(final.score_sum<2)&((final.pergsim1==1)|\
   (final.pergsim2==1)|(final.flsim1==1)|(final.flsim2==1)),'level']=8
#总分小于2且无1   
final.loc[(final.score_sum<2)&(final.pergsim1!=1)&\
   (final.pergsim2!=1)&(final.flsim1!=1)&(final.flsim2!=1),'level']=9
###################################
#总分大于3.5
summary.loc[summary.score_sum>=3.5,'level']=1
#总分在3-3.5之间且pergsim和flsim至少各有一个1
summary.loc[(summary.score_sum<3.5)&(summary.score_sum>=3.0)&((summary.pergsim1==1)|\
   (summary.pergsim2==1))&((summary.flsim1==1)|(summary.flsim2==1)),'level']=2
#总分在3-3.5之间但pergsim或flsim只有一边有1
summary.loc[(summary.score_sum<3.5)&(summary.score_sum>=3.0)&\
          (((summary.pergsim1!=1)&(summary.pergsim2!=1)&((summary.flsim1==1)|(summary.flsim2==1)))|\
           (((summary.pergsim1==1)|(summary.pergsim2==1))&(summary.flsim1!=1)&(summary.flsim2!=1))) ,'level']=3
#总分在3-3.5且没有1
summary.loc[(summary.score_sum<3.5)&(summary.score_sum>=3.0)&(summary.pergsim1!=1)&\
   (summary.pergsim2!=1)&(summary.flsim1!=1)&(summary.flsim2!=1),'level']=4
#总分在2-3之间且pergsim和flsim至少各有一个1
summary.loc[(summary.score_sum<3.0)&(summary.score_sum>=2.0)&((summary.pergsim1==1)|\
   (summary.pergsim2==1))&((summary.flsim1==1)|(summary.flsim2==1)),'level']=5
#总分在2-3之间但pergsim或flsim只有一边有1  
summary.loc[(summary.score_sum<3.0)&(summary.score_sum>=2.0)&\
          (((summary.pergsim1!=1)&(summary.pergsim2!=1)&((summary.flsim1==1)|(summary.flsim2==1)))|\
           (((summary.pergsim1==1)|(summary.pergsim2==1))&(summary.flsim1!=1)&(summary.flsim2!=1))) ,'level']=6
#总分在2-3且没有1
summary.loc[(summary.score_sum<3.0)&(summary.score_sum>=2.0)&(summary.pergsim1!=1)&\
   (summary.pergsim2!=1)&(summary.flsim1!=1)&(summary.flsim2!=1),'level']=7
#总分小于2但有1
summary.loc[(summary.score_sum<2)&((summary.pergsim1==1)|\
   (summary.pergsim2==1)|(summary.flsim1==1)|(summary.flsim2==1)),'level']=8
#总分小于2且无1   
summary.loc[(summary.score_sum<2)&(summary.pergsim1!=1)&\
   (summary.pergsim2!=1)&(summary.flsim1!=1)&(summary.flsim2!=1),'level']=9
## for test
# final.loc[(final.pergsim1==0)&(final.pergsim2==0)&(final.flsim1==1)&(final.flsim2==0.4),
#           ['left_fullname','right_fullname']].head()
# final.loc[final.score_sum>=3.5,['left_fullname','right_fullname']].sample(10)
# final.loc[(final.score_sum<3)&(final.score_sum>=2.5),['left_fullname','right_fullname']].sample(10)
# final.loc[(final.score_sum<4)&(final.score_sum>=3.5)&(final.pergsim1!=1)&\
#    (final.pergsim2!=1)&(final.flsim1!=1)&(final.flsim2!=1),\
#    ['left_fullname','right_fullname']].sample()
# final.loc[(final.score_sum<3)&(final.score_sum>=2.0)&((final.pergsim1==1)|\
#    (final.pergsim2==1)|(final.flsim1==1)|(final.flsim2==1)),\
#    ['left_fullname','right_fullname']].sample(5)

###generate sample
sample=final.groupby('level').sample(30)
 
sample.to_pickle('individual_name_simlevel_sample.pkl')
sample.to_csv('individual_name_simlevel_sample.csv')  
final.to_pickle('individual_name_similarity_final.pkl')
final.to_csv('individual_name_similarity_final.csv')
summary.to_pickle('individual_name_similarity_freq.pkl')
summary.to_csv('individual_name_similarity_freq.csv')