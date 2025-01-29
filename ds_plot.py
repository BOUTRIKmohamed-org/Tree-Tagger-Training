import os
import pandas as pd
import matplotlib.pyplot as plt

params = {
   "cl" :2,
   "dtg": 0.7,
   "sw" : 1,
   "ecw": 0.5,
   "stg" : 1,
   "ptg": 1,
   "lt" : 0,
   "nb_lexicon" : 3
   }



for i in range(10):

   params["nb_sen"] = (i+1)*10 
   arguments = [f"-{k} {v}" for k,v in params.items()]
   os.system(f"{os.getcwd()}/train_pipeline.sh {" ".join(arguments)} >> train.log")
   os.system(f"{os.getcwd()}/test_pipeline.sh >> test.log")

df = pd.read_csv("./macro_precision.txt",header=None,names=["pronoun_accuracy","cjt_accuracy","det_accuracy","adv_accuracy","macro_accuracy"])
df["num_sentences"] = pd.Series([(i+1)*10 for i in range(10)])

# Plotting
plt.figure(figsize=(12, 6))
plt.grid(True, linestyle="--", alpha=0.6)

# Plot each class accuracy
plt.plot(df["num_sentences"], df["pronoun_accuracy"], label="That Relative Pronoun", color="blue")
plt.plot(df["num_sentences"], df["cjt_accuracy"], label="That Conjunction", color="orange")
plt.plot(df["num_sentences"], df["det_accuracy"], label="That Determiner", color="green")
plt.plot(df["num_sentences"], df["adv_accuracy"], label="That Adverb", color="red")

# Plot overall accuracy (dotted line)
plt.plot(df["num_sentences"], df["macro_accuracy"], label="Overall Accuracy", color="black", linestyle="--")


# Ajouter les annotations directement sur les lignes
for i in range(len(df)):
    plt.text(df["num_sentences"][i], df["pronoun_accuracy"][i], f"{df['pronoun_accuracy'][i]:.3f}", fontsize=9, color="blue")
    plt.text(df["num_sentences"][i], df["cjt_accuracy"][i], f"{df['cjt_accuracy'][i]:.3f}", fontsize=9, color="orange")
    plt.text(df["num_sentences"][i], df["det_accuracy"][i], f"{df['det_accuracy'][i]:.3f}", fontsize=9, color="green")
    plt.text(df["num_sentences"][i], df["adv_accuracy"][i], f"{df['adv_accuracy'][i]:.3f}", fontsize=9, color="red")
    plt.text(df["num_sentences"][i], df["macro_accuracy"][i], f"{df['macro_accuracy'][i]:.3f}", fontsize=9, color="black")

# Add labels, title, and legend
plt.xlabel("Number of Sentences per Class", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(loc="lower right", fontsize=10)

# Show the plot
plt.tight_layout()
plt.savefig("accuracy_vs_sentences.svg", format="svg")  # Save as SVG