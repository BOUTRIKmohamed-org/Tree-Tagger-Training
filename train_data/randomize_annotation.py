import pandas as pd 

#this script uses annotated data 

that_cats = {"relative","conjunction","determiner","adverb"}
final_df = pd.DataFrame()
for cat in that_cats:
   df = pd.read_csv(f"./that_{cat}.txt",sep="|",names=["sen"])

   #fix if there is an error 
   match cat : 
      case "relative": 
         real_tag = "PNQ"
      case "conjunction":
         real_tag = "CJT"
      case "determiner":
         real_tag = "AT0"
      case "adverb" :
         real_tag = "AV0"

   df["real_tag"] = real_tag

   #concat to the final dataset
   final_df = pd.concat([final_df,df],ignore_index=True,axis=0)
final_df = final_df.sample(frac=1).reset_index(drop=True)
#file to train
final_df["sen"].to_csv("/home/med/Desktop/Studies/m1_mlsd/nlp/projet/train_data/train_data.txt",index=False,header=False)
#file to fix annotation
final_df["real_tag"].to_csv("/home/med/Desktop/Studies/m1_mlsd/nlp/projet/train_data/real_tags.txt",index=False,header=False)
