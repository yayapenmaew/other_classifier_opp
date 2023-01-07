import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
# Import Module
import os
# Folder Path
path = "/Users/yayafung/Desktop/opp/OPP-115/sanitized_policies"
# Change the directory
os.chdir(path)
# Read text File
dataset = {}


def read_text_file(file_path):
    # Opening the html file
    HTMLFile = open(file_path, "r")
    # Reading the file
    index = HTMLFile.read()
    # Creating a BeautifulSoup object and specifying the parser
    S = BeautifulSoup(index, 'lxml').text
    soup_string = str(S)
    return soup_string


# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".html"):
        file_path = f"{path}/{file}"

        # call read text file function
        dataset[file] = read_text_file(file_path)


def tfIdf(dataset):
    tfIdfVectorizer = TfidfVectorizer(
        stop_words='english', use_idf=True, ngram_range=(1, 3))
    tfIdf = tfIdfVectorizer.fit_transform(dataset.values())
    df = pd.DataFrame(tfIdf[0].T.todense(
    ), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    return df


c = tfIdf(dataset)
print(c.head(25))
print(len(dataset.keys()))
