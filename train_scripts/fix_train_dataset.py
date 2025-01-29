import pandas as pd 
import argparse
#this script uses annotated data 

parser = argparse.ArgumentParser(description="Randomize training data.")
# Required arguments
parser.add_argument("-root", type=str, required=True,help="the absolute path of the train_pipeline script")

# Parse arguments
args = parser.parse_args()

#paths
ROOT =  args.root
REAL_TAGS = args.root + "/train_data/train_dataset/real_tags.txt"
TRAIN_DATA = args.root + "/train_data/train_dataset/mixed_annotated.txt"

real_tags = pd.read_csv(REAL_TAGS,header=None)

df = pd.read_csv(TRAIN_DATA,names=["token","tag","lemma"],sep="\t")
df = df.drop("lemma",axis=1)
real_tags.index = df.loc[df["token"].str.lower() == "that","tag"].index
df.loc[real_tags.index,"tag"] = real_tags.squeeze()
df.loc[df["tag"]=="SENT","tag"] = "PUN"
df.to_csv(TRAIN_DATA,sep='\t',index=False,header=False)
print(f"\tformatted train_dataset stored in {TRAIN_DATA}")
