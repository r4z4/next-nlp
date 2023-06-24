```python
import numpy as np
import regex as re
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

### Load Normalized Data

---
While we were exploring the dataset we went ahead and just created the dataframes we need for these next few runs. Again, like some of the other runs, we have some extras and some with some different embeddings or preprocessed texts.  The upside of using such small and limited examples is that at least it gives us a chance to quickly perform changes like that to get a better sense of how the data is responding.

---

```python
df = pd.read_pickle('../trivia_classification/data/dataframes/trivia_qs_normalized.pkl')
df.shape
```

    (34460, 2)


```python
df.head()
```


```python
X = df['Questions'].values
y = df['category'].values
```


```python
print(X)
```

    ['what hollywood actor portrayed casanova in the 2005 movie entitled casanova based on the life of the popular adventurer '
     'which of these is not a son of adam and eve '
     'noah sent these two birds out of the ark to search for land ' ...
     'on what kind of surface is the sport called bandy practiced '
     'what type of sport is enduro '
     'elements of which sport does the game called pickleball include ']



```python
target_names = np.unique(y)
```


```python
from sklearn.preprocessing import OrdinalEncoder
enc=OrdinalEncoder() 

# Encode categorical values
df['category_enc']=enc.fit_transform(df[['category']])

# Check encoding results in a crosstab
pd.crosstab(df['category'], df['category_enc'], margins=False)
```



```python
y = df['category_enc'].values
```


```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_features=50,
                      min_df=8,
                      max_df=0.7,
                      stop_words=stopwords.words("english"))
cleaned = vec.fit_transform(X).toarray()
```

```python
X_train, X_test, y_train, y_test = train_test_split(cleaned, y, test_size=0.2, random_state=0)
```

```python
len(X_train)
```

    27568

```python
len(y_train)
```

    27568

```python
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
```


```python
y_hat_lda_model = model.predict(X_test)
actual_and_lda_model_preds = pd.DataFrame({"Actual Category": y_test,
                                           "Predicted Category": y_hat_lda_model})
actual_and_lda_model_preds
```



```python
from sklearn import metrics
lda_model_rpt = pd.DataFrame(metrics.classification_report(y_test, y_hat_lda_model, output_dict=True)).transpose()
lda_model_rpt
```


```python
#checking for the model accuracy using score method
model.fit(X_train, y_train).score(X_train, y_train)
```

    0.4252756819500871


```python
y_pred = model.predict(X_test)
```


```python
model = LinearDiscriminantAnalysis()
data_plot = model.fit(X_train, y_train).transform(X_train)

#create LDA plot
plt.figure()
colors = ['red', 'green', 'blue', 'orange', 'purple', 'violet', 'teal', 'maroon', 'lime', 'grey', 'aqua', 'gold', 'yellow', 'black']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], target_names):
    plt.scatter(data_plot[y_train == i, 0], data_plot[y_train == i, 1], alpha=.8, color=color,
                label=target_name)

#add legend to plot
plt.legend(loc='best', shadow=False, scatterpoints=1)

#display LDA plot
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()
```


![png](/images/trivia/lda_trivia_0.png#img-thumbnail)



```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train, y_train)
```

PCA selects the components which'd result in highest spread (retain most info) & not necessarily the ones that maximize separation between classes.


```python
pca.explained_variance_ratio_
```

    array([0.06503106, 0.05712134])


```python
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y_train,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)
```


```python
data_plot[1,1]
```

    0.11961195950953266

![png](/images/trivia/lda_trivia_1.png#img-thumbnail)

```python
print(data_plot)
```

    [[ 1.41130353  1.13511287 -0.58189461 ...  1.29220416 -0.35522076
       0.49487595]
     [-1.05597864  0.11961196  1.01331247 ... -1.17341956  1.1449998
       1.04536396]
     [-0.57571158  0.08735004  0.58077694 ... -0.2778546   0.42449414
      -0.22682093]
     ...
     [-0.57571158  0.08735004  0.58077694 ... -0.2778546   0.42449414
      -0.22682093]
     [-1.32424386  0.12582777  1.68527725 ... -2.42901658 -0.36184077
       1.8126642 ]
     [-0.51144126  0.18091163  0.57832353 ... -0.94078746  0.97648786
      -0.40167008]]

