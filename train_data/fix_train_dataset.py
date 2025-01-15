import pandas as pd 

#this script uses annotated data 

real_tags = pd.read_csv("/home/med/Desktop/Studies/m1_mlsd/nlp/projet/train_data/real_tags.txt",header=None)

df = pd.read_csv(f"/home/med/Desktop/Studies/m1_mlsd/nlp/projet/train_data/train_data_annotated.txt",names=["token","tag","lemma"],sep="\t")
#Attributing an index to each sentence
#df["num_sen"] = (df["tag"] == "SENT").cumsum()
#df.loc[df["tag"] == "SENT","num_sen"] -= 1
df = df.drop("lemma",axis=1)
real_tags.index = df.loc[df["token"].str.lower() == "that","tag"].index
df.loc[real_tags.index,"tag"] = real_tags.squeeze()
df.loc[df["tag"]=="SENT","tag"] = "PUN"
df.to_csv("/home/med/Desktop/Studies/m1_mlsd/nlp/projet/train_data/fixed_train_data.txt",sep='\t',index=False,header=False)