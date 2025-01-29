import pandas as pd 
import argparse

parser = argparse.ArgumentParser(description="Randomize training data.")
# Required arguments
parser.add_argument("-nb_sen", type=int, required=True, help="Number of that sentences")
parser.add_argument("-root", type=str, required=True,help="the absolute path of the train_pipeline script")

# Parse arguments
args = parser.parse_args()

#paths
ROOT =  args.root
REAL_TAGS = args.root + "/train_data/train_dataset/real_tags.txt"
TRAIN_DATA = args.root + "/train_data/train_dataset/mixed.txt"


that_cats = {"relative","conjunction","determiner","adverb"}
final_df = pd.DataFrame()

for cat in that_cats:
   df = pd.read_csv(f"{ROOT}/train_data/train_dataset/that_{cat}.txt",sep="|",names=["sen"],nrows=args.nb_sen)
   
   match cat :
      case "relative": 
         real_tag = "PNQ"
      case "conjunction":
         real_tag = "CJT"
      case "determiner":
         real_tag = "DT0"
      case "adverb" :
         real_tag = "AV0"

   df["real_tag"] = real_tag
   #concat to the final dataset
   final_df = pd.concat([final_df,df],ignore_index=True,axis=0)

final_df = final_df.sample(frac=1,ignore_index=True)
#file to train
final_df["sen"].to_csv(TRAIN_DATA,index=False,header=False)
print(f"\ttrain_dataset stored in {TRAIN_DATA}")
#file to fix annotation
final_df["real_tag"].to_csv(REAL_TAGS,index=False,header=False)
print(f"\t real tags are stored in {REAL_TAGS}")