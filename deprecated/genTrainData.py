import csv
import os
import re
import pandas as pd
pathAnn = "/Users/yayafung/Desktop/opp/OPP-115/annotations"
os.chdir(pathAnn)
aDataset = []  # {index: [label, text]}
typeIndex = 5
itemIndex = 6
start = len('"selectedText": "')
end = -2


def readCsv(file_path):
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        label = 0
        for row in csv_reader:
            text = re.findall(
                f'"selectedText"\:(.*?)\,', row[6])
            if (row[5]) == "Other":
                label = 1
            if len(text) > 0:
                for t in text:
                    aDataset.append([label, t])


for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".csv"):
        file_path = f"{pathAnn}/{file}"
        readCsv(file_path)

df = pd.DataFrame(aDataset, columns=['label', 'text'])
print(df.sort_values(by='label', ascending=False))
os.chdir("/Users/yayafung/Desktop/opp")
# df.to_csv('labeled_text_dataset.csv')
""" with open("/Users/yayafung/Desktop/opp/OPP-115/annotations/1360_thehill.com.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    label = 0
    for row in csv_reader:
        if (row[5]) == "Other":
            label = 1
        text = re.findall(
            f'"selectedText"\:(.*?)\,', row[6])
        print(text) """
