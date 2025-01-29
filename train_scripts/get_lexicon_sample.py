import os
import random 
import pandas as pd 
import csv 
import numpy as np
import time
import subprocess
import argparse



parser = argparse.ArgumentParser(description="Randomize training data.")
# Required arguments
parser.add_argument("-nb_lexicon", type=int, required=True, help="Number of lexicon files")
parser.add_argument("-seed", type=int, required=True,help="Random seed (default: 42)")
parser.add_argument("-root", type=str, required=True,help="the absolute path of the train_pipeline script")

# Parse arguments
args = parser.parse_args()

ROOT =  args.root
SCRIPT = args.root + "/train_data/lexicon_scripts/oneWordPerLine.xsl"
LEXICON_FILE = args.root + "/train_data/lexicon.txt"
BASE_LEXICON = args.root + "/train_data/base_lexicon.txt"
TAGSET_FILE = args.root + "/train_data/tagset.txt"
TEXTS = args.root + "/train_data/Texts"
TAG_SET = set()


def get_all_file_paths(directory):
    """function that stores all the paths of available files of annotated data in a list
    Args:
        directory: the Parent folder of all the files
    Returns:
        a list of all files absolute paths
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def fix_duplicates_tokens(df):
    """
        a function that formats a file as follows :
            If we got the same token and lemma with multiple tags we keep one entry with tags seperated by - 
                example: 
                    token   lemma   tag 
                    test    test    NN1
                    test    test    VVB
                the result will be :
                    test    test    NN1-VVB
            and if the same token got different lemma the function formats that for a forward proccessing as follows:
                example: 
                    token   lemma   tag 
                    being   be      VBG
                    being   being   NN1
                the result will be :
                    being   [be,being]  [VBG,NN1]
    Args:
        df (pandas.DataFrame): a dataframe that contains 3 columns token,lemma and tag loaded from an annotated data
    Returns:
        a pandas.DataFrame with 3 columns token,lemma and tag with no duplicated tokens
    """
    df["token"]=df["token"].apply(str.strip)
    df = df.dropna(subset=['tag'])
    df["lemma"] = df["lemma"].fillna('<unknown>')
    df = df.groupby(['token','lemma'])["tag"].apply(list).reset_index()
    df["tag"] = df["tag"].apply(lambda x : "-".join(set([elem2 for elem in x for elem2 in elem.strip().split('-')])))
    df = df.groupby('token').agg({
    'lemma': list,
    'tag': list
        }).reset_index()
    return df


file_paths = get_all_file_paths(TEXTS)
nb_files = len(file_paths)
nb_files_sample = nb_files if int(args.nb_lexicon) > nb_files else int(args.nb_lexicon)

#choose random files
random.seed(args.seed)
ran = random.sample(range(0,nb_files),nb_files_sample)
file_paths_sample = [file_paths[i] for i in ran]
file_paths_sample.insert(0,BASE_LEXICON)
final_df = pd.DataFrame(columns=["token","tag","lemma"])
for i,file in enumerate(file_paths_sample):
    print(f"\tpreprocessing |{i+1}/{len(file_paths_sample)}|{file}")
    if i != 0:
        os.system(f"xsltproc {SCRIPT} {file} > {LEXICON_FILE}")
        tmp_df = pd.read_csv(LEXICON_FILE,sep='\t',on_bad_lines='skip',quoting=csv.QUOTE_NONE,names=["num_sen","token","lemma","tag","extra"])
        tmp_df = tmp_df.drop("extra",axis=1)
        tmp_df = tmp_df.drop("num_sen",axis=1)
    else : 
        tmp_df = pd.read_csv(BASE_LEXICON,sep='\t',names=["token","lemma","tag"])
    tmp_df = tmp_df.dropna(subset=["token"])
    tmp_df = fix_duplicates_tokens(tmp_df)
    d = {
        "token":[],
        "lemma":[],
        "tag":[]
    }
    for i,row in tmp_df.iterrows():
        new_lemma= []
        new_tag= []
        for lemme, tags in zip(row['lemma'], row['tag']):
            tag_list = tags.split("-") 
            #tag_list = set(tag_list) -set(new_tag)
            new_lemma.extend([lemme] * len( tag_list))  
            new_tag.extend(tag_list) 
        
        d["token"].append(row["token"])
        d["tag"].append(new_tag)
        d["lemma"].append(new_lemma)

        TAG_SET.update(new_tag)

    tmp_df = pd.DataFrame(d)
    
    print("\ttotal number of words in the file",len(tmp_df))
    tmp_df = pd.merge(tmp_df,final_df,how='left',on='token',suffixes=('_new','_old'))
    print("\tadding new words",len(tmp_df.loc[tmp_df["tag_old"].isna(),"token"].unique()))
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

    tmp_df = tmp_df.loc[~tmp_df["tag_old"].isna()]
    final_df = final_df.loc[~ final_df["token"].isin(tmp_df["token"])]
    print("\tmodifying existing words",len(tmp_df["token"].unique()))
    tmp_df["lemma_old"] = tmp_df["lemma_old"]+tmp_df["lemma_new"]
    tmp_df["tag_old"] = tmp_df["tag_old"]+ tmp_df["tag_new"]
    
    tmp_df = tmp_df[["token","tag_old","lemma_old"]].rename(
        columns={
            "tag_old" : "tag",
            "lemma_old" : "lemma", 
        }
    )
    final_df = pd.concat([final_df,tmp_df],ignore_index=True)
    
    
print("\tthe length of final df",len(final_df))

#preprocessing to adapt as a lexicon
new_column = []
for i,row in final_df.iterrows():
    print(f"\tpreprocessing row {i+1}/{len(final_df)}",flush=True,end='\r')
    new_value = []
    existing_tag = set()
    for a,b in zip(row["lemma"],row["tag"]):
        if b in existing_tag:
            continue
        existing_tag.add(b)
        new_value.append(f"{b.strip()} {a.strip()}")
    new_column.append('\t'.join(set(new_value)))

print()
final_df["tag"] = pd.Series(new_column,index=final_df.index)
final_df = final_df.drop("lemma",axis=1)
# Supprimer les doublons en fonction de la colonne 'id'
final_df = final_df.drop_duplicates(subset="token", keep="first")
final_df.to_csv(LEXICON_FILE,index=False,header=False,sep='\t')
print(f"\tLexicon stored in {LEXICON_FILE}")

# Convert the set to a space-separated string
TAG_SET = " ".join(map(str, TAG_SET))

# Write the string to a file
with open(TAGSET_FILE, "w") as file:
    file.write(TAG_SET)

print(f"\ttagset stored in {TAGSET_FILE}!")