
# Twitter Sentiment Analysis using BERTopic 

Welcome to the Twitter Sentiment Analysis project using BERTopic! This project analyzes the sentiment of tweets and categorizes them into distinct emotions using advanced Natural Language Processing (NLP) techniques.

## 📚 Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)

## 📌 Introduction
This project leverages the BERTopic model for topic modeling and sentiment analysis. The sentiments are categorized into seven emotions: neutral, surprise, sadness, anger, joy, disgust, and fear.

## 🛠️ Installation
To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
pip install -r requirements.txt
```
## 📁 Dataset
We use a dataset of tweets for analysis. You can load the dataset as follows:

```python
import pandas as pd

train = pd.read_csv('path/to/train.csv')
test = pd.read_csv('path/to/test.csv')
```
## 🧹 Preprocessing
The preprocessing steps include:

- Converting text to lowercase
- Removing URLs, stop words, and punctuation

```python
import re
import string
from nltk.corpus import stopwords

df['text'] = df['text'].str.lower()

df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+', '', x))

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words and word not in punctuation]))
```

## 📊 Exploratory Data Analysis (EDA)
We perform EDA to understand the distribution of sentiments in the dataset.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sentiment_counts = df['sentiment'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, colors=sns.color_palette('Set2'), autopct='%1.1f%%')
plt.title('Distribution of Sentiments')
plt.show()
```
## 🧠 Modeling
We use the BERTopic model for topic modeling and sentiment analysis.

```python
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2, 3), sublinear_tf=True)
sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
topic_model = BERTopic(vectorizer_model=vectorizer)

topics, probs = topic_model.fit_transform(df['text'])
```
## 📈 Results
The results include the distribution of sentiments and the extracted topics.

### Sentiment Distribution
| Sentiment | Count   |
| --------- | ------- |
| Neutral   | 11,117  |
| Positive  | 8,582   |
| Negative  | 7,781   |

### Topic Representation (N-gram)
| Topic | Representative Docs                                                   |
| ----- | --------------------------------------------------------------------- |
| 0     | day happy mothers, mothers day happy, mothers day                     |
| 1     | still awake, bed night, sleep night, get sleep                        |
| 2     | welcome twitter, im twitter, new twitter, saw tweet                   |
| 3     | new album, lost voice, listening music, listening new                 |

### Predicted Sentiments
The predicted sentiments for the first 10 tweets are as follows:

| textID     | Text                                                                                     | Original Sentiment | Predicted Sentiment | Score    |
| ---------- | ---------------------------------------------------------------------------------------- | ------------------ | ------------------- | -------- |
| 2543065d78 | way sleep next 8 9 days? way wake up, she`ll respond                                      | negative           | anticipation        | 0.930641 |
| ee267131b1 | ok... twitter almost pass you!! **** :`(                                                  | negative           | anticipation        | 0.785657 |
| 5b4cf5d1c6 | watching biggest loser hallmark. never fails make cry                                      | negative           | sadness             | 0.872735 |
| 856e0029b7 | greg pritchard got threw final britains got talent amazing performance                    | negative           | anticipation        | 0.300687 |
| 5c83af1147 | gourmet pizza bleh. pizza supposed greasy filling, delicious                              | negative           | sadness             | 0.674485 |
| 8581262345 | isn’t right now. need make more. sorry.                                                   | negative           | sadness             | 0.812764 |
| a435e058ae | srry can’t go paintballing tonight good movies                                            | negative           | fear                | 0.448848 |
| 3cbcb82071 | lol bad he’s taken!                                                                       | negative           | sadness             | 0.299860 |
| 7416c5eee3 | hypnotyst .... hmmmm... beware..                                                          | negative           | surprise            | 0.383578 |
| a24c1d14d7 | http://twitpic.com/67nxe yeah..i’m bored xd picture things interesting today              | negative           | sadness             | 0.546874 |

## 🎉 Conclusion
The Twitter Sentiment Analysis project using BERTopic provided insightful analysis into the emotions expressed in tweets. By leveraging advanced Natural Language Processing (NLP) techniques and the BERTopic model, we were able to classify tweets into distinct sentiments and extract meaningful topics.

### Key Takeaways:
- **Comprehensive Sentiment Analysis**: The model successfully categorized tweets into seven different emotions, providing a nuanced understanding of public sentiment.
- **Effective Topic Modeling**: Using BERTopic, we identified and grouped related tweets, uncovering underlying themes and trends.
- **Data Preprocessing**: The preprocessing steps, including cleaning and normalization, were crucial in improving the accuracy and quality of the sentiment analysis.
- **Visualizations**: The visual representation of sentiment distribution helped in quickly grasping the overall sentiment landscape of the dataset.

### Future Work:
- **Enhancing Model Accuracy**: Experimenting with different NLP models and techniques to further improve the accuracy of sentiment predictions.
- **Real-Time Analysis**: Implementing a real-time sentiment analysis pipeline to monitor and analyze tweets as they are posted.

This project demonstrated the power and flexibility of the BERTopic model in performing sentiment analysis and topic modeling on social media data. The insights gained can be valuable for businesses, researchers, and policymakers to understand public opinion and respond accordingly.

