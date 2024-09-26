#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import r2_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import matthews_corrcoef


# In[2]:


# Load IMDb dataset
df = pd.read_csv('IMDB Dataset.csv')
df.head()



# In[ ]:






# In[3]:


from sklearn.preprocessing import LabelBinarizer
#Replace the value of negative and positive to 0 and 1
encoder = LabelBinarizer()
df['sentiment'] = encoder.fit_transform(df['sentiment'])
df


# In[ ]:





# In[12]:


word_value = df['sentiment'].value_counts()
print(word_value)
word_value.plot(kind='bar')
plt.xlabel("Positive and Negative")
plt.ylabel("Value counts")
plt.title("IMDB Dataset")
plt.show()


# In[5]:


# Preprocess the text data 
df['review'] = df['review'].str.lower()
df['review'] = df['review'].replace('[^\w\s]', '', regex=True)
df['review'] = df['review'].replace('<br />', '', regex=True)
df['review'] = df['review'].replace(r"https\S+|www\S+|http\S+", '', regex=True)
df['review'] 


# In[ ]:





# In[6]:


stop_words = set(stopwords.words('english'))

def stopword_tokenize(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Join the words back into a string
    return ' '.join(filtered_words)

df['review'] = df['review'].apply(stopword_tokenize)
df['review']


# In[7]:


from collections import Counter

# Create a list of all words in the 'review' column
words = [word for review in df['review'] for word in review.split()]

# Count the frequency of each word
word_counts = Counter(words)

# Get the most common word
most_common_word = word_counts.most_common(10)
most_common_word


# In[ ]:





# In[ ]:





# In[8]:


# Split the dataset
X= df['review']
y= df['sentiment']
X_train = X[:40000]
y_train = y[:40000]
X_test = X[40000:]
y_test = y[40000:]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Vectorize the text using CountVectorizer
vectorizer = CountVectorizer(max_features=5000) 
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)



# In[9]:


# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Make predictions on the test set
lrpred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, lrpred)
report = classification_report(y_test, lrpred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)


# In[10]:


cmlr = confusion_matrix(y_test, lrpred)
sns.heatmap(cmlr, cmap="Greens",annot=True ,fmt='d')


# In[11]:


# Calculate MCC
mcc = matthews_corrcoef(y_test, lrpred)
print("Matthews Correlation Coefficient:", mcc)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




