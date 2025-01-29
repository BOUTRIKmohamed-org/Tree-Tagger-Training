#!/bin/bash

# Default values for optional arguments
SEED=42
ST="SENT"
CL=2
DTG=0.7
ECW=0.5
STG=1
PTG=1
SW=1
LT=0
ROOT=$(dirname "$(realpath "$0")")

#Function to display usage
usage() {
   echo "Usage: $0 -nb_sen <number> -nb_lexicon <number> [-seed <number>] [-st <string>] [-cl <number>] [-dtg <float>] [-ecw <float>] [-atg <float>]"
   exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -nb_sen)
            NB_SEN=$2
            shift 2
            ;;
        -nb_lexicon)
            NB_LEXICON=$2
            shift 2
            ;;
        -seed)
            SEED=$2
            shift 2
            ;;
        -st)
            ST=$2
            shift 2
            ;;
        -cl)
            CL=$2
            shift 2
            ;;
        -dtg)
            DTG=$2
            shift 2
            ;;
        -sw)
            SW=$2
            shift 2
            ;;
        -ecw)
            ECW=$2
            shift 2
            ;;
        -stg)
            STG=$2
            shift 2
            ;;
        -ptg)
            PTG=$2
            shift 2
            ;;
        -lt)
            LT=$2
            shift 2
            ;;
         
        *)
            usage
            ;;
    esac
done

# Check if required arguments are set
if [[ -z "$NB_SEN" || -z "$NB_LEXICON" ]]; then
    echo "Error: -nb_sen and -nb_lexicon are required arguments."
    usage
fi

# Validate folder structure
if [[ ! -d "train_data/Texts" || ! -d "train_data/lexicon_scripts" || ! -d "train_data/train_dataset" ]]; then
    echo "Error: Required folder structure is missing."
    echo "Ensure 'train_data/Texts', 'train_data/lexicon_scripts', and 'train_dataset' directories exist."
    exit 1
fi

# Validate required files in train_dataset
REQUIRED_FILES=("that_adverb.txt" "that_conjunction.txt" "that_determiner.txt" "that_relative.txt")
for FILE in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "train_data/train_dataset/$FILE" ]]; then
        echo "Error: Missing required file 'train_dataset/$FILE'"
        exit 1
    fi
done

# Print the table of the hyperparameters
printf "\n%-35s %-10s\n" "Description" "Value"
printf "%-35s %-10s\n" "-----------------------------------" "----------"
printf "%-35s %-10s\n" "Seed" "$SEED"
printf "%-35s %-10s\n" "End of sentence" "$ST"
printf "%-35s %-10s\n" "Contexte length" "$CL"
printf "%-35s %-10s\n" "Decision tree gain threshold" "$DTG"
printf "%-35s %-10s\n" "Equivalence class weight" "$ECW"
printf "%-35s %-10s\n" "Smoothing weight" "$SW"
printf "%-35s %-10s\n" "Suffix tree gain threshold" "$STG"
printf "%-35s %-10s\n" "Prefix tree gain threshold" "$PTG"
printf "%-35s %-10s\n" "Lexical entries probability threshold" "$LT"
echo -e "----------------------------------------------\n"

echo "Step 1: Launch randomize_train_dataset.py"
python3 $ROOT/train_scripts/randomize_train_dataset.py -nb_sen $NB_SEN -root $ROOT|| { echo "Error: randomize.py failed."; exit 1; }

echo "Step 2: Launch TreeTagger system command"
tree-tagger-english-bnc $ROOT/train_data/train_dataset/mixed.txt > $ROOT/train_data/train_dataset/mixed_annotated.txt || { echo "Error: TreeTagger command failed."; exit 1; }

echo "Step 3: Launch fix_train_dataset.py"
python3 $ROOT/train_scripts/fix_train_dataset.py -root $ROOT || { echo "Error: fix_train_dataset.py failed."; exit 1; }

echo "Step 4: Launch get_lexicon_sample.py"
python3 $ROOT/train_scripts/get_lexicon_sample.py -nb_lexicon $NB_LEXICON -seed $SEED -root $ROOT || { echo "Error: get_lexicon_sample.py failed."; exit 1; }

echo "Step 5: Launch lexicon_formatter.py"
python3 $ROOT/train_scripts/lexicon_formatter.py -root $ROOT || { echo "Error: lexicon_formatter.py failed."; exit 1; }

echo "Step 6: Final TreeTagger training command"
train-tree-tagger -st "PUN" $ROOT/train_data/lexicon_formatted.txt $ROOT/train_data/tagset.txt $ROOT/train_data/train_dataset/mixed_annotated.txt $ROOT/TreeTagger/lib/english-FTthatdev.par \
    -cl $CL -dtg $DTG -sw $SW -ecw $ECW -stg $STG -ptg $PTG -lt $LT|| { echo "Error: Training command failed."; exit 1; }

echo "parameters are stored in $ROOT/TreeTagger/lib/english-FTthatdev.par"
cp $ROOT/TreeTagger/cmd/tree-tagger-english $ROOT/TreeTagger/cmd/tree-tagger-english-FTthatdev 

sed -i 's|^PARFILE=.*|PARFILE=${LIB}/english-FTthatdev.par|' $ROOT/TreeTagger/cmd/tree-tagger-english-FTthatdev

echo "Pipeline executed successfully! And the trained tree-tagger can be used with the command <tree-tagger-english-FTthatdev>"