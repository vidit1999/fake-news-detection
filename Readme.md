# Fake News Detection using ML

## Data Collection and Pre-processing

Data as a CSV file is collected from [Kaggle](https://www.kaggle.com/c/fake-news/data?select=train.csv). Only `text` and `label` attributes are used for classification purpose, others are dropped. Data is pre-processed using,

```py
stop_words = stopwords.words('english')
lemma = WordNetLemmatizer()

def clean_text(text):
    text = text.lower() # lowering
    text = text.encode("ascii", "ignore").decode() # non ascii chars
    text = re.sub(r'\n',' ', text) # remove new-line characters
    text = re.sub(r'\W', ' ', text) # special chars
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # single characters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) # single char at first
    text = re.sub(r'[0-9]', ' ', text) # digits
    text = re.sub(r'\s+', ' ', text, flags=re.I) # multiple spaces
    return ' '.join([lemma.lemmatize(word) for word in word_tokenize(text) if word not in stop_words])
```
---

## Text to Features using TF-IDF

`25%` of total `20,761` data points are set aside for testing purpose. Text from rest of the data is converted into features using `TfidfVectorizer`. Test data is also transformed info features using this. There are a total of `128,387` features in our training data.

```py
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(x_train)
test_X = tfidf.transform(x_test)
```

---

## Classifier Training and Results

Two classifiers, `MultinomialNB` and `PassiveAggressiveClassifier`, are trained on training data. It took less than `5` seconds to train for both classifiers. Testing accuracies of the classifiers are `0.8695819687921402` and `0.9614717780774418` respectively.

---

## License

[MIT License](License.md)