**Attempting to Classify Sanskrit Texts by Genre Using MNB/Logistic Regression**

This script experiments with genre classification in Sanskrit texts. It scrapes files containing Sanskrit text transliterated according to the IAST conventions from the GRETIL repository. This text is cleaned, and tagged in a dataframe for classification. The tags are simply the genres the GRETIL repository ascribes to each text.

This dataframe (which is very large) is saved as a CSV file. It is then vectorised, split into training and test sets, and fed into a Multinomial Naive Bayes model and a simple Logistic Regression model. Neither produce particularly satisfying results: MNB clocks in at 65%, whereas Logistic Regression manages only 62%. This could be improved by tagging the Sanskrit corpus properly, although the software to do this in Python does not exist at present.



