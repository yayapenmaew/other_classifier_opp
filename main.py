import csv
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
pathAnn = "/Users/yayafung/Desktop/opp/OPP-115/annotations"
os.chdir(pathAnn)
aDataset = []  # {index: [label, text]}
x = []
y = []
typeIndex = 5
itemIndex = 6
start = len('"selectedText": "')
end = -2


def tfIdf(dataset):
    tfIdfVectorizer = TfidfVectorizer(
        stop_words='english', use_idf=True, ngram_range=(1, 3))
    tfIdf = tfIdfVectorizer.fit_transform(dataset.values())
    df = pd.DataFrame(tfIdf[0].T.todense(
    ), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    return df


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
                    y.append(label)
                    x.append(t)


for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".csv"):
        file_path = f"{pathAnn}/{file}"
        readCsv(file_path)

df = pd.DataFrame(aDataset, columns=['label', 'text'])

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(
    1, 3), lowercase=True)
X_train_text = text_transformer.fit_transform(X_train)
X_test_text = text_transformer.transform(X_test)

logit = LogisticRegression(
    C=5e1, solver='lbfgs', multi_class='multinomial', random_state=17, n_jobs=4, max_iter=400)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
cv_results = cross_val_score(
    logit, X_train_text, y_train, cv=skf, scoring='f1_micro')
print(cv_results, cv_results.mean())
