# Newsgroup 20 Dataset with Body - Clean Run 03

---

After seeing some pretty lackluster results from the subject only dataset, lets now go ahead and use the body for the text that we will train on. This should offer us a lot more data, but it was a struggle wrangling it all into the dataframes. I chose to go ahead and start from the text files themselves, and I will incode some snippets of the transformations below.

---

```python
import numpy as np
import json
import regex as re
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils
import seaborn as sns
import keras
import nltk
import random

from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from sklearn.naive_bayes import MultinomialNB
```

```python
# Load model
model_file = 'models/newsgroup_body_clean_model'
model = keras.models.load_model(model_file)
```

## Augmentation. Run 03 = Synonym Insertion


```python
df = pd.read_pickle('../data/dataframes/newsgroup_body_cleaned_exploded.pkl')
```


```python
def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words)
	return new_words

def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = utils.get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)
	
def insert_rejoin(x):
	if len(x) > 1:
		words = random_insertion(x.split(), 1)
		sentence = ' '.join(words)
		return sentence
```


```python
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
df['exploded_body'] = df['exploded_body'].apply(lambda x: insert_rejoin(x))
print(df.sample(frac=1).reset_index(drop=True).loc[:,['newsgroup', 'exploded_body']].head().to_markdown())
```


<style>.container { width:100% !important; }</style>


|    | newsgroup   | exploded_body                                                                                                                                   |
|---:|:------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | comp_elec   | fax modem best well use home dear offic bought around doe nt know data fax featur use voic mail box realli like -- - captain zod zod ncubecom   |
|  1 | sci_med     | ocd articl crnfg newshawaiiedu sharynk hawaiiedu write recent heard mental disord call obsess compuls disord caus could caus nervou breakdown ob|
|  2 | comp_elec   | ax max ax ax ax ax ax ax ax ax ax ax ax ax ax ax max ax ax ax ax ax ax ax ax ax ax ax ax axe ax ax max ax ax ax ax ax ax ax ax ax ax ax ax ax   |
|  3 | politics    | peopl live differ regim unit self-percept palestinian peopl identifi palestin territori entiti ethnic forethought religi entiti incorrect palest|
|  4 | religion    | object valu v scientif accuraci wa year say christian moral articl srusnewsww mantiscouk mathew mathew mantiscouk wrote lpzsml unicornnottacuk  |


These EDA methods can actually end up producing some NaN types which we need to discard. I think that comp_elec entry right there is also a testament to no matter how much cleaning and preprocessing you do, sometimes you just get bad data. Ideally we would drop that but I doubt it would make much of a difference here.


```python
nan_rows = df[df['exploded_body'].isnull()]
print(nan_rows.to_markdown())
```


```python
df = df.dropna()
```


```python
nan_rows = df[df['exploded_body'].isnull()]
print(nan_rows.to_markdown())
```

    | newsgroup   | body   | exploded_body   |
    |-------------|--------|-----------------|



```python
all_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
# We'll use all
target_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
```


```python
# container for sentences
X = np.array([s for s in df['exploded_body']])
# container for sentences
y = np.array([n for n in df['newsgroup']])
```


```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(df['newsgroup'])
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)

classes = np.unique(y_train)
mapping = dict(zip(classes, target_categories))

len(X_train), len(X_test), classes, mapping
```




    (51862,
     17288,
     array([0, 1, 2, 3, 4, 5, 6]),
     {0: 'sport',
      1: 'autos',
      2: 'religion',
      3: 'comp_elec',
      4: 'sci_med',
      5: 'seller',
      6: 'politics'})




```python
# model parameters
vocab_size = 1200
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
```


```python
# tokenize sentences
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# convert train dataset to sequence and pad sequences
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

# convert validation dataset to sequence and pad sequences
validation_sequences = tokenizer.texts_to_sequences(X_test)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)
```


```python
# fit model
num_epochs = 20
history = model.fit(train_padded, y_train, 
                    epochs=num_epochs, verbose=1,
                    validation_split=0.3)

# predict values
pred = model.predict(validation_padded)
```

    2023-06-21 00:38:23.510932: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 17425440 exceeds 10% of free system memory.


    Epoch 1/20
    1135/1135 [==============================] - 6s 4ms/step - loss: 1.9798 - accuracy: 0.4307 - val_loss: 1.2645 - val_accuracy: 0.5585
    Epoch 2/20
    1135/1135 [==============================] - 5s 4ms/step - loss: 1.0619 - accuracy: 0.6272 - val_loss: 0.9048 - val_accuracy: 0.6835
    Epoch 3/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.7877 - accuracy: 0.7285 - val_loss: 0.7124 - val_accuracy: 0.7545
    Epoch 4/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.6348 - accuracy: 0.7849 - val_loss: 0.6078 - val_accuracy: 0.7934
    Epoch 5/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.5457 - accuracy: 0.8173 - val_loss: 0.5460 - val_accuracy: 0.8153
    Epoch 6/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.4897 - accuracy: 0.8361 - val_loss: 0.5156 - val_accuracy: 0.8227
    Epoch 7/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.4503 - accuracy: 0.8503 - val_loss: 0.4832 - val_accuracy: 0.8362
    Epoch 8/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.4209 - accuracy: 0.8607 - val_loss: 0.4644 - val_accuracy: 0.8448
    Epoch 9/20
    1135/1135 [==============================] - 5s 4ms/step - loss: 0.3985 - accuracy: 0.8676 - val_loss: 0.4513 - val_accuracy: 0.8501
    Epoch 10/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3810 - accuracy: 0.8733 - val_loss: 0.4402 - val_accuracy: 0.8520
    Epoch 11/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3655 - accuracy: 0.8782 - val_loss: 0.4332 - val_accuracy: 0.8556
    Epoch 12/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.3534 - accuracy: 0.8817 - val_loss: 0.4306 - val_accuracy: 0.8589
    Epoch 13/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3431 - accuracy: 0.8854 - val_loss: 0.4314 - val_accuracy: 0.8551
    Epoch 14/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3327 - accuracy: 0.8903 - val_loss: 0.4239 - val_accuracy: 0.8610
    Epoch 15/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.3239 - accuracy: 0.8918 - val_loss: 0.4178 - val_accuracy: 0.8616
    Epoch 16/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3160 - accuracy: 0.8949 - val_loss: 0.4153 - val_accuracy: 0.8624
    Epoch 17/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.3094 - accuracy: 0.8966 - val_loss: 0.4175 - val_accuracy: 0.8625
    Epoch 18/20
    1135/1135 [==============================] - 5s 4ms/step - loss: 0.3037 - accuracy: 0.8985 - val_loss: 0.4109 - val_accuracy: 0.8641
    Epoch 19/20
    1135/1135 [==============================] - 5s 4ms/step - loss: 0.2971 - accuracy: 0.9011 - val_loss: 0.4110 - val_accuracy: 0.8650
    Epoch 20/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.2926 - accuracy: 0.9020 - val_loss: 0.4186 - val_accuracy: 0.8630
    541/541 [==============================] - 1s 2ms/step



```python
import os

file_name = 'run_03'
plot_type = 'history'
model_name = 'newsgroup_body_clean'
#####
os.makedirs(f"images/{plot_type}", exist_ok=True)
os.makedirs(f"images/{plot_type}/{model_name}", exist_ok=True)
save_path = f'images/{plot_type}/{model_name}/{file_name}.png' 

utils.plot_history_and_save(history, save_path)
```


![png](clean_run_03_files/clean_run_03_17_0.png)



```python
# TensorFlow SavedModel format => .keras
model_file = 'models/newsgroup_body_clean_model'
model.save(model_file)
```

