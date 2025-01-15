import string

input_file_path = "/home/med/Desktop/Studies/m1_mlsd/nlp/projet/test_data/that_pronoun.txt"
output_file_path = "/home/med/Desktop/Studies/m1_mlsd/nlp/projet/test_data_formatted/that_pronoun.txt"
# Open the file in read mode
in_file = open(input_file_path, "r")

line = in_file.readline()
all_parts = []

while line:
   line = line.strip()
   line = line.rstrip(string.punctuation)
   line_parts = line.split(' ')
   all_parts += line_parts
   line = in_file.readline()  

with open(output_file_path, "w", encoding="utf-8") as file:
    for word in all_parts:
        file.write(word + "\n")

file.close()