import os
from pdfminer.high_level import extract_text

path = "./"

for sample in os.listdir(path):
    if sample.endswith("pdf"):
        with open(f"text/{sample.split('.')[0]}.txt",  "w", encoding="UTF-8") as file:
            text = extract_text(sample)
            file.write(text)
