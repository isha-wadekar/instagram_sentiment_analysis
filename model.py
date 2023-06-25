#Importing necessary libraries
import pandas as pd
import seaborn as sns
from textblob import TextBlob

# Load data
d=pd.read_csv('/content/indianecoinsta.csv')    #d is original data file
d=pd.DataFrame(d)
print(d[:2])

#Data Exploration
print(d.shape)
print(d.info())
print(d.isnull().sum())
print(d.describe())
print(d['query'].value_counts())
print(d['type'].value_counts())

# Data Preprocessing
#d1 is the data for identification, source, type, post and its impressions
d1=d[['commentCount','likeCount','postId','ownerId','viewCount','description','type','pubDate','imgUrl']]
d1=d[['postId','ownerId','pubDate','type','description','imgUrl','viewCount','likeCount','commentCount']]
print(d1[:2])

##handling null values
d1.isnull().sum() #let's drop columns viewCount, imgUrl and rows where description is empty
d2 = d1.dropna(subset=['description'])
print(d2.isnull().sum())
d2.drop(['viewCount','imgUrl'],axis=1,inplace=True)
print(d2[:2])
print(d2.isnull().sum())

#Data Analysis
d2['Subjectivity'] = d2['description'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
d2['Polarity'] = d2['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
d2['Sentiment'] = d2['Polarity'].apply(lambda x: 'negative' if x<0 else 'positive' if x>0 else 'neutral')
d2[:2]

print(d2.describe())  #aggregate sentiment is 'a bit positive'

#Data analysis of posts that got liked atleast n times
def polarity(data,n):
  for i in range(len(data)):
    if data[i]['likeCount']>=n:
      data[i]['Polarity']=data['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
  return data

def subjectivity(data,n):
  for i in range(len(data)):
    if data[i]['likeCount']>=n:
      data[i]['Subjectivity']=data['description'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
  return data

print(polarity(d2,211))
print(subjectivity(d2,211))
