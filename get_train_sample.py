import os
import random 
import pandas as pd 
import csv 
import numpy as np
import time

RATIO = 0.00125
SCRIPT = "/home/med/Downloads/2554/download/XML/Scripts/oneWordPerLine.xsl"
OUTPUT_FILE = "/home/med/Desktop/Studies/m1_mlsd/nlp/projet/output.txt"
ANNOTATED_THAT = "/home/med/Desktop/Studies/m1_mlsd/nlp/projet/train_data/fixed_train_data.txt"
def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def fix_duplicates_tokens(df):
    df = df.dropna(subset=['tag'])
    df["lemma"] = df["lemma"].fillna('<unknown>')
    df = df.groupby(['token','lemma'])["tag"].apply(list).reset_index()
    df["tag"] = df["tag"].apply(lambda x : "-".join(set([elem2 for elem in x for elem2 in elem.strip().split('-')])))
    #print("before",len(df))
    #print("unique",len(df["token"].unique()))
    df = df.groupby('token').agg({
    'lemma': list,
    'tag': list
        }).reset_index()
    #df["lemma"] = df["lemma"].apply(lambda x : "-".join(set([elem2 for elem in x for elem2 in elem.strip().split('-')])))
    #df["lemma"] = df["lemma"].apply(lambda x : "-".join(set([elem2 for elem in x for elem2 in elem.strip().split('-')])))
    #print("after",len(df))
    #print(df.head())
    #print("============================================")
    #exit()
    return df

directory = "/home/med/Downloads/2554/download/Texts"
file_paths = get_all_file_paths(directory)
nb_files = len(file_paths)
nb_files_sample = 2

#choose random files 
ran = random.sample(range(0,nb_files),nb_files_sample)
file_paths_sample = [file_paths[i] for i in ran]

#that_df = pd.read_csv(ANNOTATED_THAT)
#that_ratio = len(that_df["num_sen"].unique())//nb_files_sample
#if that_ratio == 0:
#    print("u need more 'that' annotated sentences")
#    exit()

final_df = pd.DataFrame(columns=["token","tag","lemma"])
for i,file in enumerate(file_paths_sample):
    print(f"preprocessing |{i+1}/{len(file_paths_sample)}|{file}")
    os.system(f"xsltproc {SCRIPT} {file} > {OUTPUT_FILE}")
    tmp_df = pd.read_csv(OUTPUT_FILE,sep='\t',on_bad_lines='skip',quoting=csv.QUOTE_NONE,names=["num_sen","token","lemma","tag","extra"])
    tmp_df = tmp_df.drop("extra",axis=1)
    tmp_df = tmp_df.drop("num_sen",axis=1)
    tmp_df = tmp_df.dropna(subset=["token"])
    tmp_df = fix_duplicates_tokens(tmp_df)
    print("total number of words in the file",len(tmp_df))
    tmp_df = pd.merge(tmp_df,final_df,how='left',on='token',suffixes=('_new','_old'))
    print("adding new words",len(tmp_df.loc[tmp_df["tag_old"].isna()]))
    #adding new words
    final_df = pd.concat([final_df,
                          tmp_df.loc[tmp_df["tag_old"].isna(),["token","tag_new","lemma_new"]].rename(
                              columns = {
                                  "tag_new" :"tag",
                                  "lemma_new" :"lemma",
                              }
                          )
                          ]
                         ,ignore_index=True)
    
    #preprocessing old words 
    tmp_df = tmp_df.loc[~ tmp_df["tag_old"].isna()]
    print("modifying existing words",len(tmp_df))
    tmp_df["tag_old"] = tmp_df["tag_old"].str.strip() +'-'+ tmp_df["tag_new"].str.strip()
    tmp_df["tag_old"] = tmp_df["tag_old"].apply(lambda x : '-'.join(set(x.split('-'))))
    tmp_df = tmp_df[["token","tag_old","lemma_old"]].rename(
        columns={
            "tag_old" : "tag",
            "lemma_old" : "lemma", 
        }
    )

    final_df = pd.concat([final_df,tmp_df],ignore_index=True)
    #choose random sentences from extra annotated that sentences
    #ran = np.random.choice(that_df["num_sen"].unique(), size=that_ratio, replace=False)
    #final_df = pd.concat([final_df,that_df[that_df["num_sen"].isin(ran)]],ignore_index=True)
    #delete the added that sentences
    #that_df = that_df[~that_df["num_sen"].isin(ran)]
#if not that_df.empty:
#    final_df = pd.concat([final_df,that_df[that_df["num_sen"].isin(ran)]],ignore_index=True) 
 
print("the length of final df")
print(len(final_df))   

#preprocessing to adapt as a lexicon
new_column = []
for i,row in final_df.iterrows():
    print(f"preprocessing row {i+1}/{len(final_df)}",flush=True,end='\r')
    new_value = ""
    row_tags = row["tag"].strip().split('-')
    for a,b in zip([row["lemma"]] * len(row_tags),row_tags):
        new_value += f"{b} {a} "
    new_column.append(new_value.strip())
final_df["tag"] = pd.Series(new_column,index=final_df.index)
final_df = final_df.drop("lemma",axis=1)
print(len(final_df["token"].unique()))
final_df.to_csv(OUTPUT_FILE,index=False,header=False,sep='\t')   
print(f"file stored in {OUTPUT_FILE}")