import pandas as pd
import numpy as np
import seaborn as sns

# Load data
d=pd.read_csv('/content/indianecoinsta.csv')    #d is original data file
d=pd.DataFrame(d)
d[:2]

#Data Exploration
d.shape
d.info()
d.isnull().sum()
d.describe()
d['query'].value_counts()
d['type'].value_counts()

# Data Preprocessing

#d1 is the data for identification, source, type, post and its impressions
d1=d[['commentCount','likeCount','postId','ownerId','viewCount','description','type','pubDate','imgUrl']]
d1=d[['postId','ownerId','pubDate','type','description','imgUrl','viewCount','likeCount','commentCount']]
d1[:2]

d1.isnull().sum() #let's drop columns viewCount, imgUrl and rows where description is empty

d2 = d1.dropna(subset=['description'])
d2.isnull().sum()

d2.drop(['viewCount','imgUrl'],axis=1,inplace=True)
d2[:2]

d2.isnull().sum()

#Data Analysis

from textblob import TextBlob
d2['Subjectivity'] = d2['description'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
d2['Polarity'] = d2['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
d2['Sentiment'] = d2['Polarity'].apply(lambda x: 'negative' if x<0 else 'positive' if x>0 else 'neutral')
d2[:2]

d2.describe()#aggregate sentiment is 'a bit positive'

#Data analysis of posts that got liked atleast n times

def polarity(data,n):
  from textblob import TextBlob
  for i in range(len(data)):
    if data[i]['likeCount']>=n:
      data[i]['Polarity']=data['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
  return data

def subjectivity(data,n):
  from textblob import TextBlob
  for i in range(len(data)):
    if data[i]['likeCount']>=n:
      data[i]['Subjectivity']=data['description'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
  return data

polarity(d2,211)

subjectivity(d2,211)
