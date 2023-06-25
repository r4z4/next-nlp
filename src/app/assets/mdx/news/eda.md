# 20_Newsgroup Dataset

---

This is a pretty common dataset that I have seen used in various NLP tasks that you may have come across. Similar to some of the other runs here, I wanted to simplify things a bit and so I began by just focusing on the ```subject``` column and avoiding the body of the text. This ended up having the opposite effect, and so the lesson learned here was that taking shortcuts early on often leads to poor results, which likely means you'll spend time redoing things. Regardless, though, here is a look at the original data and then the final format with which we can then use in our model definition.

There is certainly a lot of information there, and it would seem obvious that it would improve the model's performance, no? Well, we will just need to return back to find out. The answer is yes, though, it does improve the performance. But solely using the subject should still allow the model to be able to differentiate between newsgroups. This does bring up another issue though, and that is a very important hyperparameter of ```number_of_classes``` (in some cases this would be considered a hyperparameter, where it is under our control, wherase in others it may not be, such as a boolean classifier where there really is no input as to what the classes should be).

Twenty is a lot, especially when we are cutting off most of its data to begin with. For that reason, we start by combining the data into some sensible categories that we anticpate the model whould be able to distinguish. For example, we just merge the newsgroups of 'rec.cport.baseball' & 'rec.sport.hockey' together into a single 'sports' category. This gets us down to seven classes that we will be predicting on. This is still probably too high of a number, again with such limited data, but they are sensible categories so we should expect some decent results. Here is a look at some of the data in its final form. This is taken from the 'sci_med' group which encompasses the 'space' newsgroup that our example above was taken from.



![png](/images/news/original_20_text.png)


```python
import pandas as pd
df = pd.read_pickle('data/pkl/to_use/sci_med.pkl')
print(df.sample(frac=1).reset_index(drop=True).head().to_markdown(tablefmt="grid"))
```


  |    | newsgroup   | subject                                                         |
  |----|-------------|-----------------------------------------------------------------|
  |  0 | sci_med     | Barbecued foods and health risk                                 |
  |  1 | sci_med     | Why not give $1 billion to first year-long moon residents?      |
  |  2 | sci_med     | Blindsight                                                      |
  |  3 | sci_med     | Contraceptive pill                                              |
  |  4 | sci_med     | Thrush ((was: Good Grief! (was Candida Albicans: what is it?))) |



As you can see we are losing quite a bit of information here. Look at #2. Why would any sort of classifier think that belongs in a category devoted to science and medicine? In practice we go through and address this by either just using more data and including more features (the rest of the text here as a 'body' column/feature) or by removing these types of entries - perhaps just by length alone, but hopefully with a better method. It is also worth noting our goal here is use the augmentation techniques as well to improve our dataset, so this approach also allows us to compare those two later on. For now, though, we carry on.


```python
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from textwrap import wrap
```


```python
import os
dirpath = 'data/pkl/to_use/'
directory = os.fsencode(dirpath)
df = pd.DataFrame(columns=['newsgroup', 'subject'])
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     name, ext = filename.split('.')
     if filename.endswith(".pkl"): 
         df_ = pd.read_pickle(dirpath + filename) 
         df = pd.merge(
            df, df_, how="outer"
        )
         continue
     else:
         continue
```


```python
print(df.head().to_markdown(tablefmt="grid"))
```

    +----+-------------+--------------------------------+
    |    | newsgroup   | subject                        |
    +====+=============+================================+
    |  0 | sci_med     | INFO NEEDED: Gaucher's Disease |
    +----+-------------+--------------------------------+
    |  1 | sci_med     | INFO NEEDED: Gaucher's Disease |
    +----+-------------+--------------------------------+
    |  2 | sci_med     | jiggers                        |
    +----+-------------+--------------------------------+
    |  3 | sci_med     | jiggers                        |
    +----+-------------+--------------------------------+
    |  4 | sci_med     | jiggers                        |
    +----+-------------+--------------------------------+


More bad news for our already limited dataset here was that there are a good amount of duplicated. This is due to the fact that the ```subject``` field is often replied to, and so while the new text may differ that subject will be the same. I hope this is at least beginning to show us how important carefully going over our data and thinking about our approach will become. I believe it was at this point where I really started losing hope in the outcome here. But, we press on, and just deop those duplcate rows for now then take a look at what we have.


```python
# Drop dups
df.drop_duplicates(inplace = True)
```


```python
print(df.head().to_markdown(tablefmt="grid"))
```

    +----+-------------+--------------------------------+
    |    | newsgroup   | subject                        |
    +====+=============+================================+
    |  0 | sci_med     | INFO NEEDED: Gaucher's Disease |
    +----+-------------+--------------------------------+
    |  2 | sci_med     | jiggers                        |
    +----+-------------+--------------------------------+
    |  8 | sci_med     | Breech Baby Info Needed        |
    +----+-------------+--------------------------------+
    | 10 | sci_med     | Analgesics with Diuretics      |
    +----+-------------+--------------------------------+
    | 16 | sci_med     | Lactose intolerance            |
    +----+-------------+--------------------------------+


Alright, so at least that is fixed. But now we have to cringe as we check what that brought our number down to. This is just one dataset, but still.


```python
df.shape
```


    (8562, 2)


```python
def normalize_text(s):
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    
    return s

df['subject'] = [normalize_text(s) for s in df['subject']]
```


```python
print(df.sample(frac=1).reset_index(drop=True).head().to_markdown(tablefmt="grid"))
```

    +----+-------------+-------------------------------------------+
    |    | newsgroup   | subject                                   |
    +====+=============+===========================================+
    |  0 | comp_elec   | what type of ic is this???                |
    +----+-------------+-------------------------------------------+
    |  1 | comp_elec   | dos quick c 2.5 crashes windows 3.1?      |
    +----+-------------+-------------------------------------------+
    |  2 | seller      | ** dayna etherprint 10base-t new cheap ** |
    +----+-------------+-------------------------------------------+
    |  3 | comp_elec   | price drop on c650 within 2 months?       |
    +----+-------------+-------------------------------------------+
    |  4 | politics    | arab leaders and bosnia                   |
    +----+-------------+-------------------------------------------+


We just used our really simple normalize method here for text cleaning. On our next run we will use a slightly better one. Now lets take some of the standard EDA (Exploratory Data Analysis) techniques to get a better sense of the data. First step there would be to create a Document Term Matrix.


```python
df_grouped=df[['newsgroup','subject']].groupby(by='newsgroup').agg(lambda x:' '.join(x))
print(df_grouped.sample(frac=1).reset_index(drop=True).head().to_markdown(tablefmt="grid"))
```


CountVectorizer is a method to convert text to numerical data. Examples always help.
text = [‘Computers do not like words, they do like numbers’]

        comp      do      like     not    numbers    they     words
+----+--------+--------+--------+--------+--------+--------+--------+
|  0 |  1     |  2     |  2     |  1     |  1     |  1     |  1     |
+----+--------+--------+--------+--------+--------+--------+--------+

text = [‘Computers do not like words', 'they do like numbers’]

        comp      do      like     not    numbers    they     words
+----+--------+--------+--------+--------+--------+--------+--------+
|  0 |  1     |  1     |  1     |  1     |  0     |  0     |  1     |
+----+--------+--------+--------+--------+--------+--------+--------+
|  1 |  0     |  0     |  1     |  0     |  1     |  1     |  0     |
+----+--------+--------+--------+--------+--------+--------+--------+


```python
cv=CountVectorizer(analyzer='word')
data=cv.fit_transform(df_grouped['subject'])
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names_out())
df_dtm.index=df_grouped.index
df_dtm.head(6)
```



Again we see many of the same issues we have seen before, which just reiterates why we need a better cleaning method on this next run. As a litte sanity check though, and For a nice visual representation, let's create the word cloud. We learned quite a bit from some previous examples and so I have actually started to use these a little more frequently.


```python
# Function for generating word clouds
def generate_wordcloud(data,title):
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title('\n'.join(wrap(title,60)),fontsize=13)
  plt.show()
  
# Transposing document term matrix
df_dtm=df_dtm.transpose()

# Plotting word cloud for each product
for index,product in enumerate(df_dtm.columns):
  generate_wordcloud(df_dtm[product].sort_values(ascending=False),product)
```


![png](/images/news/20_newsgroup_eda_19_0.png#wordcloud)



![png](/images/news/20_newsgroup_eda_19_1.png#wordcloud)



![png](/images/news/20_newsgroup_eda_19_2.png#wordcloud)



![png](/images/news/20_newsgroup_eda_19_3.png#wordcloud)



![png](/images/news/20_newsgroup_eda_19_4.png#wordcloud)



![png](/images/news/20_newsgroup_eda_19_5.png#wordcloud)



![png](/images/news/20_newsgroup_eda_19_6.png#wordcloud)


We need to remove stopwords. As our confidence sheds, let's get the polarity. The Seller category looks pretty decent at least.


```python
from textblob import TextBlob
df['polarity']=df['subject'].apply(lambda x:TextBlob(x).sentiment.polarity)
print(df.sample(frac=1).reset_index(drop=True).head().to_markdown(tablefmt="grid"))
```

    +----+-------------+--------------------------------------------+------------+
    |    | newsgroup   | subject                                    |   polarity |
    +====+=============+============================================+============+
    |  0 | comp_elec   | any nanao 750i compatible mac video cards? |        0   |
    +----+-------------+--------------------------------------------+------------+
    |  1 | comp_elec   | gateway 4dx2-66v update                    |        0   |
    +----+-------------+--------------------------------------------+------------+
    |  2 | sci_med     | interesting dc-x cost anecdote             |        0.5 |
    +----+-------------+--------------------------------------------+------------+
    |  3 | comp_elec   | third party monitor on iisi                |        0   |
    +----+-------------+--------------------------------------------+------------+
    |  4 | comp_elec   | compiling mh-6.8 and xmh on sco 3.2.4.     |        0   |
    +----+-------------+--------------------------------------------+------------+



```python
question_polarity_sorted=pd.DataFrame(df.groupby('newsgroup')['polarity'].mean().sort_values(ascending=True))

plt.figure(figsize=(16,8))
plt.xlabel('Polarity')
plt.ylabel('Newsgroups')
plt.title('Polarity of Different Newsgroups from 29_Newsgroups Dataset')
polarity_graph=plt.barh(np.arange(len(question_polarity_sorted.index)),question_polarity_sorted['polarity'],color='lightgreen',)

# Writing product names on bar
for bar,product in zip(polarity_graph,question_polarity_sorted.index):
  plt.text(0.005,bar.get_y()+bar.get_width(),'{}'.format(product),va='center',fontsize=11,color='black')

# Writing polarity values on graph
for bar,polarity in zip(polarity_graph,question_polarity_sorted['polarity']):
  plt.text(bar.get_width()+0.001,bar.get_y()+bar.get_width(),'%.3f'%polarity,va='center',fontsize=11,color='black')
  
plt.yticks([])
plt.show()
```


![png](/images/news/20_newsgroup_eda_22_0.png#img-thumbnail)

