#!/bin/bash

ROOT=$(dirname "$(realpath "$0")")


# Define the list of that categories
categories=("adv" "conjunction" "determiner" "pronoun")  # Replace with your actual categories

# Loop through each category
for category in "${categories[@]}"; do
  # Generate input and output file names
  input_file="${ROOT}/test_data/that_${category}.txt"
  output_file="${ROOT}/annotated_test_data/that_${category}.txt"
  
  # Run the tree-tagger command
  tree-tagger-english-FTthatdev "$input_file" > "$output_file"
  
  # Print a message for progress
  echo "Processed $input_file -> $output_file"
done

python ${ROOT}/calcul_base_accur.py -root $ROOT
