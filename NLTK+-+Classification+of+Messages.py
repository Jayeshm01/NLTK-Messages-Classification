
# coding: utf-8

# In[2]:


# Import Packages
import nltk 
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[4]:


# Read the reviews
messages = [line.rstrip() for line in open ('SMSSpamCollection')]
print(len(messages))


# In[5]:


# Print the reviews
for message_no, message in enumerate(messages[:10]):
    print(message_no,message)
    print('\n')


# In[7]:


messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label','message'])
messages.head(10)


# In[5]:


messages.describe()


# In[6]:


messages.groupby('label').describe()


# In[7]:


messages['length'] = messages['message'].apply(len)


# In[8]:


messages.head(10)


# In[9]:


messages['length'].plot.hist(bins=150)


# In[10]:


messages.hist(column='length',by='label',bins=100, figsize=(12,6))


# In[11]:


# text Pre processing


# In[12]:


import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)


# In[13]:


nopunc


# In[14]:


#nltk.download_shell()


# In[15]:


from nltk.corpus import stopwords
stopwords.words('english')[0:10]


# In[17]:


clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess


# In[18]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join  characters to form the string.
    nopunc = ''.join(nopunc)
    
    #  remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[22]:


# Check 
messages['message'].head(5).apply(text_process)


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer


# In[24]:



bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))


# In[25]:


messages_bow = bow_transformer.transform(messages['message'])


# In[26]:


print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


# In[23]:


sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format((sparsity)))


# In[28]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[29]:


messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# In[30]:


# Naive Bais
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


# In[31]:


all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[33]:


from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))


# In[34]:


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# In[35]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[31]:


pipeline.fit(msg_train,label_train)


# In[32]:


predictions = pipeline.predict(msg_test)


# In[33]:


print(classification_report(predictions,label_test))

