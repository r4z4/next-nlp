```python
import numpy as np
import json
import regex as re
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
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
model_file = 'models/newsgroup_clean_model'
model = keras.models.load_model(model_file)
```

## Augmentation. Run 03 = Synonym Insertion


```python
df = pd.read_pickle('data/dataframes/newsgroup_cleaned.pkl')
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
df['subject'] = df['subject'].apply(lambda x: insert_rejoin(x))
print(df.sample(frac=1).reset_index(drop=True).head().to_markdown(tablefmt="grid"))
```

    +----+-------------+--------------------------------------------------+
    |    | newsgroup   | subject                                          |
    +====+=============+==================================================+
    |  0 | comp_elec   | pgp system of rules idea ibm system              |
    +----+-------------+--------------------------------------------------+
    |  1 | comp_elec   | connect ten digitis x repost                     |
    +----+-------------+--------------------------------------------------+
    |  2 | autos       | execute carb cleaner - work perform carb rebuild |
    +----+-------------+--------------------------------------------------+
    |  3 | comp_elec   | need card x win popup menu packag                |
    +----+-------------+--------------------------------------------------+
    |  4 | politics    | nineteenth 19th centuri capit                    |
    +----+-------------+--------------------------------------------------+


These EDA methods can actually end up producing some NaN types which we need to discard.


```python
nan_rows = df[df['subject'].isnull()]
print(nan_rows.to_markdown(tablefmt="grid"))
```

    +-------+-------------+-----------+
    |       | newsgroup   | subject   |
    +=======+=============+===========+
    | 10590 | politics    |           |
    +-------+-------------+-----------+
    | 13180 | sport       |           |
    +-------+-------------+-----------+
    | 16268 | religion    |           |
    +-------+-------------+-----------+
    | 26798 | comp_elec   |           |
    +-------+-------------+-----------+
    | 27808 | comp_elec   |           |
    +-------+-------------+-----------+



```python
df = df.dropna()
```


```python
all_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
# We'll use all
target_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
```


```python
# container for sentences
X = np.array([s for s in df['subject']])
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




    (6167,
     2056,
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

    Epoch 1/20
    135/135 [==============================] - 2s 6ms/step - loss: 1.8546 - accuracy: 0.4076 - val_loss: 1.7949 - val_accuracy: 0.4138
    Epoch 2/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.7611 - accuracy: 0.4291 - val_loss: 1.7235 - val_accuracy: 0.4263
    Epoch 3/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.6952 - accuracy: 0.4372 - val_loss: 1.6794 - val_accuracy: 0.4354
    Epoch 4/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.6579 - accuracy: 0.4451 - val_loss: 1.6504 - val_accuracy: 0.4403
    Epoch 5/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.6294 - accuracy: 0.4525 - val_loss: 1.6280 - val_accuracy: 0.4516
    Epoch 6/20
    135/135 [==============================] - 1s 6ms/step - loss: 1.6094 - accuracy: 0.4601 - val_loss: 1.6131 - val_accuracy: 0.4495
    Epoch 7/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.5931 - accuracy: 0.4613 - val_loss: 1.5979 - val_accuracy: 0.4554
    Epoch 8/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.5752 - accuracy: 0.4655 - val_loss: 1.5856 - val_accuracy: 0.4571
    Epoch 9/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.5599 - accuracy: 0.4685 - val_loss: 1.5701 - val_accuracy: 0.4625
    Epoch 10/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.5433 - accuracy: 0.4727 - val_loss: 1.5556 - val_accuracy: 0.4657
    Epoch 11/20
    135/135 [==============================] - 1s 6ms/step - loss: 1.5278 - accuracy: 0.4773 - val_loss: 1.5416 - val_accuracy: 0.4706
    Epoch 12/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.5104 - accuracy: 0.4826 - val_loss: 1.5285 - val_accuracy: 0.4700
    Epoch 13/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.4941 - accuracy: 0.4854 - val_loss: 1.5136 - val_accuracy: 0.4738
    Epoch 14/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.4768 - accuracy: 0.4896 - val_loss: 1.5025 - val_accuracy: 0.4722
    Epoch 15/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.4612 - accuracy: 0.4947 - val_loss: 1.4870 - val_accuracy: 0.4841
    Epoch 16/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.4449 - accuracy: 0.4998 - val_loss: 1.4732 - val_accuracy: 0.4857
    Epoch 17/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.4262 - accuracy: 0.5030 - val_loss: 1.4591 - val_accuracy: 0.4943
    Epoch 18/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.4094 - accuracy: 0.5125 - val_loss: 1.4472 - val_accuracy: 0.4965
    Epoch 19/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.3929 - accuracy: 0.5165 - val_loss: 1.4351 - val_accuracy: 0.4943
    Epoch 20/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.3742 - accuracy: 0.5199 - val_loss: 1.4209 - val_accuracy: 0.4986
    65/65 [==============================] - 0s 1ms/step



```python
import os

file_name = 'run_03'
plot_type = 'history'
model_name = 'newsgroup_clean'
#####
os.makedirs(f"images/{plot_type}", exist_ok=True)
os.makedirs(f"images/{plot_type}/{model_name}", exist_ok=True)
save_path = f'images/{plot_type}/{model_name}/{file_name}.png' 

utils.plot_history_and_save(history, save_path)
```


![png](/images/news/clean_run_03.png)



```python
# reviews on which we need to predict
sentence = ["son of genuine vinyl records 4sale"]

# convert to a sequence
sequences = tokenizer.texts_to_sequences(sentence)

# pad the sequence
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# preict the label
print(model.predict(padded))
```

    1/1 [==============================] - 0s 35ms/step
    [[0.34423554 0.59764796 0.34918392 0.35519084 0.41514224 0.12569432
      0.41660988]]

```python
# TensorFlow SavedModel format => .keras
model_file = 'models/newsgroup_clean_model'
model.save(model_file)
```


