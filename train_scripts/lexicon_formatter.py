import csv 
import os 
import argparse

"""
   This script will just remove some characters added using to_csv method from pandas package
   like the "" that are added when the column contains the seperator character
"""

parser = argparse.ArgumentParser()
parser.add_argument("-root", type=str, required=True,help="the absolute path of the train_pipeline script")

# Parse arguments
args = parser.parse_args()
ROOT =  args.root


input_file = ROOT + "/train_data/lexicon.txt"
output_file = ROOT + "/train_data/lexicon_formatted.txt"



with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
   reader = csv.reader(infile, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
   for row in reader:
      formatted_line = "\t".join(row)
      outfile.write(formatted_line + "\n")
      
print(f"\tTraitement terminé. Les données formatées sont dans {output_file}.")