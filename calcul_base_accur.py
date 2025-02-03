import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Randomize training data.")
# Required arguments
parser.add_argument("-root", type=str, required=True,help="the absolute path of the train_pipeline script")


# Parse arguments
args = parser.parse_args()

#paths
ROOT =  args.root
ANNOTATED_TEST_DATA = ROOT + "/annotated_test_data"

that_cats = {"adv","conjunction","determiner","pronoun"}

macro_accuracy = 0
for cat in that_cats:
   df = pd.read_csv(ANNOTATED_TEST_DATA + f"/that_{cat}.txt",sep='\t',header=None,names=["token","tag","lemma"])
   df["token"] = df["token"].str.lower() 
   match cat :
      case "pronoun":
         true = len(df[(df["token"] == "that") & (df["tag"] == "PNQ")])
         pronoun_accuracy = true/len(df[(df["token"] == "that")])
         macro_accuracy +=pronoun_accuracy
         print(f"the accuracy of relative that after training the model is : {true}/{len(df[(df['token'] == 'that')])} = {pronoun_accuracy:.2f}")
      case "conjunction":
         true = len(df[(df["token"] == "that") & (df["tag"] == "CJT")])
         cjt_accuracy = true/len(df[(df["token"] == "that")])
         macro_accuracy +=cjt_accuracy
         print(f"the accuracy of conjunction that after training the model is : {true}/{len(df[(df['token'] == 'that')])} = {cjt_accuracy:.2f}")
      
      case "determiner":
         true = len(df[(df["token"] == "that") & (df["tag"] == "DT0")])
         det_accuracy = true/len(df[(df["token"] == "that")])
         macro_accuracy +=det_accuracy
         print(f"the accuracy of determiner that after training the model is : {true}/{len(df[(df['token'] == 'that')])} = {det_accuracy:.2f}")
      
      case "adv":
         true = len(df[(df["token"] == "that") & (df["tag"] == "AV0")])
         adv_accuracy = true/len(df[(df["token"] == "that")])
         macro_accuracy +=adv_accuracy
         print(f"the accuracy of adverb that after training the model is : {true}/{len(df[(df['token'] == 'that')])} = {adv_accuracy:.2f}")

print(f"The macro accuracy is : {macro_accuracy/len(that_cats):.3f}")
#with open("macro_accuracy.txt", "a") as file:
#   file.write(f"{pronoun_accuracy},{cjt_accuracy},{det_accuracy},{adv_accuracy},{macro_accuracy/len(that_cats):.3f}\n")