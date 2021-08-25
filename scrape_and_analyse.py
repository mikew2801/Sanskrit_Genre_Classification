"""
This script experiments with genre classification in Sanskrit texts. It scrapes files containing Sanskrit text transliterated according to the IAST conventions from the GRETIL repository. This text is cleaned, and tagged in a dataframe for classification. The tags are simply the genres the GRETIL repository ascribes to each text.

This dataframe (which is very large) is saved as a CSV file. It is then vectorised, split into training and test sets, and fed into a Multinomial Naive Bayes model and a simple Logistic Regression model. Neither produce particularly satisfying results: MNB clocks in at 65%, whereas Logistic Regression manages only 62%. This could be improved by tagging the Sanskrit corpus properly, although the software to do this in Python does not exist at present.
"""

import requests
from bs4 import BeautifulSoup as bs
import string
import re

base_url = "http://gretil.sub.uni-goettingen.de/gretil.html"

gretil_base_page = requests.get(base_url)
base_soup = bs(gretil_base_page.text, "html.parser")
sanskrit_main = base_soup.find("div", {"id":"outline-container-org72df5ce"})

literatures = sanskrit_main.find_all("div", {"class":"outline-3"})

text_dict = {}

# Note: Vedic diacritic marks should be added to this
accepted_chars = string.ascii_lowercase + "ṅḥāṇṣśṭṛūīṃñḍ\n "

def clean_text_file(text):
    # First get rid of the header that is given as standard in (nearly) all cases
    output = []
    if text.find("# Text") != -1:
        text = text[text.find("# Text")+7:]

    # Get rid of some standard character patterns that are not useful for the analysis
    text = re.sub(r"\/\/.*\/\/", "", text)

    # The result will only contain lower case ascii chars and the standard diacritic marks for classical/Vedic Sanskrit
    text = "".join([char for char in text if char in accepted_chars])

    # Split the text up into segments according to line breaks
    output = text.splitlines()

    # Finally remove any null strings from the output
    output = [segment for segment in output if segment != ""]
    return output

for body in literatures:
    body_texts = []

    # The name of the genre is given by GRETIL in a h3 tag at the beginning of every division on its page
    current_genre = body.find("h3").text
    text_links = body.find_all("a", {"href":True})
    urls =["http://gretil.sub.uni-goettingen.de/" + url["href"] for url in text_links if ".txt" in url["href"]]

    for url in urls:
        try:
            body_texts += clean_text_file(requests.get(url).text)
        # Some URLS were broken
        except requests.ConnectionError:
            pass

    text_dict[current_genre] = body_texts
    body_texts = []

final_segments = []
final_tags = []

for key in text_dict.keys():
    for segment in text_dict[key]:
        final_segments.append(segment)
        final_tags.append(key)

final_dict = {"Texts":final_segments, "Tags":final_tags}

import pandas as pd
text_df = pd.DataFrame(final_dict)

# The dataframe is large, so save it for later analysis
text_df.to_csv("output.csv")

# ANALYSIS

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("output.csv")

# Split the dataset into independent/dependent variables
X = df["Texts"]
y = df["Tags"]

# Now encode the lyrics column using the sklearn's CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split the data into training and test sets in an 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Multinomial Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Now compare our predictions to the actual results and print the result
predicted = clf.predict(X_test)
accuracy = accuracy_score(predicted, y_test)
print("Accuracy of MNB = " + str(accuracy))


# Now try simple logisitic regression on the same data

from sklearn import feature_extraction
from sklearn import pipeline
from sklearn import linear_model.LogisticRegression()

model = LogisticRegression()

model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
acc = (accuracy_score(y_test, y_predicted)) * 100
print("Accuracy = " + str(acc))

