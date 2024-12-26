

import nltk   

import re
import pandas as pd
import numpy as np
import os
import glob

import time as tm

import datetime

from sklearn.model_selection import train_test_split

import torch
print(f'cuda is available ={torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler,TensorDataset, random_split
from torch import nn

from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel,BertPreTrainedModel,BertConfig, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader,TensorDataset, random_split
from sentence_transformers import models, SentenceTransformer,losses,InputExample, util

from string_grouper import  match_most_similar ,match_strings, group_similar_strings,compute_pairwise_similarities, StringGrouper

perc=[0.005, 0.01,0.025, 0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.975, 0.99,0.995] ### used when checking the summary stats of a data
    
import warnings
warnings.filterwarnings('ignore')

def isid(data, variables):
    dup = data.duplicated(variables)
    dupsum = dup.describe()
    if (dupsum['unique'] == 1):
        print(str(variables) + " uniquely identify this dataset")
    else:
        print(str(variables) + " Do NOT uniquely identify this dataset")
        
#%% get training data
# os.chdir('D:\Dropbox\Lu-Yang\VC gender culture\Data\pitchbook to sdc ipo')
df1=pd.read_feather('bright_nameid_namepair_clean.feather')
df2=pd.read_feather('enhanced_indeed_bright_company_pair.feather')

df=df1[['compname1','compname2','samecomp']].append(df2[['compname1','compname2','samecomp']]).drop_duplicates(['compname1','compname2']).reset_index(drop=True)
df=df.dropna(how='any')
df['tid']=df.index
df3=df[['compname1','compname2','tid']]
df3.columns=['compname2','compname1','tid']
df=df.merge(df3[['compname2','compname1','tid']],on=['compname1','compname2'],how='left')
df=df[(df.tid_y.isnull()) | (df.tid_x<df.tid_y)].drop(['tid_x','tid_y'],1)


train=df[(df.tid_y.isnull()) | (df.tid_x<df.tid_y)].drop(['tid_x','tid_y'],1)
train=train.rename(columns={'samecomp':'matched'})
# train1=pd.read_csv('pairs from yets to compustat.csv')
# train2=pd.read_excel('pairs from indeed compustat.xlsx')
# train3=pd.read_csv('pairs from glassdoor to compustat.csv')
# train4=pd.read_csv('pairs from bright to ciqcomp.csv')
# train4['matched']=1
# train4=train4.drop('Unnamed: 0',1)
# train1.columns=['compname1','compname2','matched']
# train2.columns=train1.columns
# train3.columns=train1.columns
# train4.columns=train1.columns
# train=pd.concat([train1,train2,train3,train4]).sort_values(['compname1','compname2','matched'],ascending=False).drop_duplicates(['compname1','compname2'])
# train['compname1']=train.compname1.str.lower().str.replace(' +',' ').str.strip()
# train['compname2']=train.compname2.str.lower().str.replace(' +',' ').str.strip()
# train=train[train.matched.notna()]

### convert label format to float32, necessary for sbert
train['matched']=train['matched'].astype('float32') 
# use sbert to predict, then save models

os.chdir('/nas_01/private/luguangli/company name similarity with bert/')

max_seq_length=32

model = SentenceTransformer('all-MiniLM-L6-v2',device='cuda')

model.max_seq_length = max_seq_length

train=train.reset_index(drop=True)

train_samples = []
for idx in train.index:
    train_samples.append(InputExample(
        texts=[train.loc[idx,'compname1'], train.loc[idx,'compname2']],
        label=train.loc[idx,'matched']
    ))


batch_size = 128

loader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

loss = losses.ContrastiveLoss(model=model)

epochs = 1

warmup_steps = int(len(loader) * epochs * 0.05)

t1=tm.time()

model.fit(train_objectives=[(loader, loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path='./sbert_contrastiveloss_created_all_bright_indeed_comp_pairs',
    show_progress_bar=True,
)

t2=tm.time()

nom=int((t2-t1)/60)

print(f'training takes {nom} minutes')

#%%
# tokenzie/encode data to prepare for bert
# allpairs=pd.read_feather('all pairs from bright to ciqcomp.feather')

# allpairs=allpairs[['compname_bright','compname_ciqcomp']]
os.chdir(r'D:\Dropbox\Lu-Yang\VC gender culture\Data\pitchbook to sdc ipo'.replace('\\','/'))

allpairs=pd.read_csv('cleaned_group_ipo_preliminary_pair.csv')

allpairs=allpairs[['comp_name_std', 'Issuer_std']].drop_duplicates()
allpairs=allpairs.dropna(how='any')

# for v in ['comp_name', 'Issuer', 'comp_name_std', 'Issuer_std']:
#     allpairs[v+'_bert']=allpairs[v].str.lower().str.replace(' +',' ').str.strip()

# allpairs.drop('Unnamed: 0',1).to_csv('cleaned_group_ipo_preliminary_pair_with_stdname_for_bert.csv',index=False)

for v in allpairs.columns:
    allpairs[v]=allpairs[v].str.lower().str.replace(' +',' ').str.strip()

allpairs=allpairs.reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # by default, autotokenizer is much faster than bert tokenizer...

sentences1=allpairs[list(allpairs.columns)[0]].to_list()  ### input for autotokenizer must be a list, can't be a numpy array, which is what needed for bert tokenizer
sentences2=allpairs[list(allpairs.columns)[1]].to_list()

max_sentence_length=32

encoded_dict1 = tokenizer.batch_encode_plus (
                    sentences1,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = max_sentence_length,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    truncation=True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )

encoded_dict2 = tokenizer.batch_encode_plus (
                    sentences2,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = max_sentence_length,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    truncation=True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )


input_ids1, attention_masks1,input_ids2,attention_masks2=encoded_dict1['input_ids'],encoded_dict1['attention_mask'],encoded_dict2['input_ids'],encoded_dict2['attention_mask']

pred_dataset = TensorDataset(input_ids1, attention_masks1,input_ids2, attention_masks2)

"""Adapt the pooling in SBERT to pooling in my setting"""
batch_size,ndim=256, 384
pred_dataloader=DataLoader(
            pred_dataset,  # The samples.
            sampler = SequentialSampler(pred_dataset), # Pull out batches sequentially.
            batch_size = batch_size 
        )
os.chdir(r'D:\Dropbox\Company name matching general\bert models\unsupervised models'.replace('\\','/'))
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


model=AutoModel.from_pretrained('./sbert_mnrl_with_1epochs')   #sbert_contrastiveloss_created_bright_comp_pairs
model=model.to(device)
model.zero_grad()  # speed will be faster without computing gradients (which is by default)

reslist=[None]*len(pred_dataloader)       
step=0
t0 = tm.time()    
for input_id1,attention_mask1,input_id2,attention_mask2 in iter(pred_dataloader):
    step+=1
    
    if step % 50 == 0 and not step == 1:
        # Calculate elapsed time in minutes.
        elapsed = format_time(tm.time() - t0)
 
        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(pred_dataloader), elapsed))
    
    try: ### there would be sample problem with the last step due to dimension mismatch
        input_dict1 = {
            'input_ids':input_id1.cuda(),
            'attention_mask': attention_mask1.cuda()
        }
        
        input_dict2 = {
            'input_ids':input_id2.cuda(),
            'attention_mask': attention_mask2.cuda()
        }
        
           
        output1, output2 =model(**input_dict1), model(**input_dict2)
        
        embeddings1,embeddings2=output1.last_hidden_state.view(batch_size,max_sentence_length,ndim),output2.last_hidden_state.view(batch_size,max_sentence_length,ndim) #get layer output 
        
        mask1,mask2 = input_dict1['attention_mask'].unsqueeze(-1).expand(embeddings1.size()).float(),input_dict2['attention_mask'].unsqueeze(-1).expand(embeddings2.size()).float() #get attention mask
        
        masked_embeddings1,masked_embeddings2=embeddings1*mask1,embeddings2*mask2 #mask embedding when needed
        
        summed1,summed2=torch.sum(masked_embeddings1,axis=1),torch.sum(masked_embeddings2,axis=1)   #sum up embeddings
        
        counted1,counted2 = torch.clamp(mask1.sum(1), min=1e-9),torch.clamp(mask2.sum(1), min=1e-9) #sum up # of items with attention=1
        
        pooled1,pooled2=summed1/counted1,summed2/counted2 #compute average embeddings
        
        cosim=np.diagonal(util.cos_sim(pooled1, pooled2).detach().cpu().numpy()) #compute correlation 
        
        reslist[step-1]=cosim
        # if step==1:
        #     all_cosim=cosim
        # else:
        #     all_cosim=np.concatenate((all_cosim,cosim)) #store all things
    except:
        pass

all_cosim=np.concatenate(reslist[:-1])

allpairs.loc[:len(all_cosim)-1,'nlpscore']=all_cosim

os.chdir(r'D:\Dropbox\Lu-Yang\VC gender culture\Data\pitchbook to sdc ipo'.replace('\\','/'))
allpairs.to_csv(f'all standardized name pairs from pitchbook to sdc with bert score3.csv',index=False)

# allpairs.reset_index(drop=True).to_feather(f'other pitchbook to ciq mnrl_with_{n}epochs.feather')
#%%
model=AutoModel.from_pretrained(f'/nas_01/private/luguangli/company name similarity with bert/sbert_consim_20221007') 
model=model.to(device)
model.zero_grad()  # speed will be faster without computing gradients (which is by default)
       
step=0
t0 = tm.time()    
for input_id1,attention_mask1,input_id2,attention_mask2 in iter(pred_dataloader):
    step+=1
    
    if step % 50 == 0 and not step == 1:
        # Calculate elapsed time in minutes.
        elapsed = format_time(tm.time() - t0)
 
        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(pred_dataloader), elapsed))
    
    try: ### there would be sample problem with the last step due to dimension mismatch
        input_dict1 = {
            'input_ids':input_id1.cuda(),
            'attention_mask': attention_mask1.cuda()
        }
        
        input_dict2 = {
            'input_ids':input_id2.cuda(),
            'attention_mask': attention_mask2.cuda()
        }
        
           
        output1, output2 =model(**input_dict1), model(**input_dict2)
        
        embeddings1,embeddings2=output1.last_hidden_state.view(batch_size,max_sentence_length,ndim),output2.last_hidden_state.view(batch_size,max_sentence_length,ndim) #get layer output 
        
        mask1,mask2 = input_dict1['attention_mask'].unsqueeze(-1).expand(embeddings1.size()).float(),input_dict2['attention_mask'].unsqueeze(-1).expand(embeddings2.size()).float() #get attention mask
        
        masked_embeddings1,masked_embeddings2=embeddings1*mask1,embeddings2*mask2 #mask embedding when needed
        
        summed1,summed2=torch.sum(masked_embeddings1,axis=1),torch.sum(masked_embeddings2,axis=1)   #sum up embeddings
        
        counted1,counted2 = torch.clamp(mask1.sum(1), min=1e-9),torch.clamp(mask2.sum(1), min=1e-9) #sum up # of items with attention=1
        
        pooled1,pooled2=summed1/counted1,summed2/counted2 #compute average embeddings
        
        cosim=np.diagonal(util.cos_sim(pooled1, pooled2).detach().cpu().numpy()) #compute correlation 
        
        if step==1:
            all_cosim=cosim
        else:
            all_cosim=np.concatenate((all_cosim,cosim)) #store all things
    except:
        pass

allpairs.loc[:len(all_cosim)-1,'nlpscore']=all_cosim

allpairs.to_csv(f'all pairs from bright to ciqcomp with supervised bert score.csv',index=False)





















