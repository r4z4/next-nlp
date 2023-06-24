# TREC Dataset with EDA - Easy Data Augmentation - Methods

---

This is another dataset where we have a relatively small dataset, and so we'll be using the EDA methods for some simple data augmentation, which will allow us to quickly and easily maximize the size of the data that we have. We will use the Synonym Repalcement, Random Insertion and Random Swap methods and see where that will take us.

```python
import pandas as pd
import random
```


```python
df = pd.read_pickle(r'data/dataframes/final_cleaned_normalized.pkl')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>entity</th>
      <th>question</th>
      <th>question_normalized</th>
      <th>question_cleaned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DESC</td>
      <td>How did serfdom develop in and then leave Russia</td>
      <td>how did serfdom develop in and then leave russia</td>
      <td>serfdom develop leav russia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENTY</td>
      <td>What films featured the character Popeye Doyle</td>
      <td>what films featured the character popeye doyle</td>
      <td>film featur charact popey doyl</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DESC</td>
      <td>How can I find a list of celebrities ' real names</td>
      <td>how can i find a list of celebrities real names</td>
      <td>find list celebr real name</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENTY</td>
      <td>What fowl grabs the spotlight after the Chines...</td>
      <td>what fowl grabs the spotlight after the chines...</td>
      <td>fowl grab spotlight chines year monkey</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABBR</td>
      <td>What is the full form of .com</td>
      <td>what is the full form of com</td>
      <td>full form com</td>
    </tr>
  </tbody>
</table>
</div>




```python
import random
from random import shuffle
random.seed(1)

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line
```


```python
from nltk.corpus import wordnet
```


```python
########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

import nltk
nltk.download('wordnet')

def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

def replace_rejoin_sr(x):
    words = synonym_replacement(x.split(), 1)
    sentence = ' '.join(words)
    return sentence
```

    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!



```python
replace_rejoin_sr('What is the total land mass of the continent of africa')
```




    'What is the add up land mass of the continent of africa'




```python
## Loop through and apply synonym_replacement for each headline
df['question_cleaned_sr'] = df['question_cleaned'].apply(lambda x: replace_rejoin_sr(x))
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>entity</th>
      <th>question</th>
      <th>question_normalized</th>
      <th>question_cleaned</th>
      <th>question_cleaned_sr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DESC</td>
      <td>How did serfdom develop in and then leave Russia</td>
      <td>how did serfdom develop in and then leave russia</td>
      <td>serfdom develop leav russia</td>
      <td>serfdom break leav russia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENTY</td>
      <td>What films featured the character Popeye Doyle</td>
      <td>what films featured the character popeye doyle</td>
      <td>film featur charact popey doyl</td>
      <td>moving picture show featur charact popey doyl</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DESC</td>
      <td>How can I find a list of celebrities ' real names</td>
      <td>how can i find a list of celebrities real names</td>
      <td>find list celebr real name</td>
      <td>find list celebr substantial name</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENTY</td>
      <td>What fowl grabs the spotlight after the Chines...</td>
      <td>what fowl grabs the spotlight after the chines...</td>
      <td>fowl grab spotlight chines year monkey</td>
      <td>fowl grab spotlight chine year monkey</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABBR</td>
      <td>What is the full form of .com</td>
      <td>what is the full form of com</td>
      <td>full form com</td>
      <td>full contour com</td>
    </tr>
  </tbody>
</table>
</div>



Note: I did need to add a check for word length - ```if len(words) > 1:``` - here -> Again, just a function of us using such a limited dataset. 


```python
########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		if len(words) > 1:
			add_word(new_words)
	return new_words

def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)
	
def replace_rejoin_ri(x):
    words = random_insertion(x.split(), 1)
    sentence = ' '.join(words)
    return sentence
```


```python
replace_rejoin_ri('What is the total land mass of the continent of africa')
```




    'What is the total represent land mass of the continent of africa'




```python
df['question_cleaned_ri'] = df['question_cleaned'].apply(lambda x: replace_rejoin_ri(x))
```


```python
########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		if len(words) > 1:
			new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

def replace_rejoin_rs(x):
    words = random_swap(x.split(), 1)
    sentence = ' '.join(words)
    return sentence
```


```python
replace_rejoin_rs('What is the total land mass of the continent of africa')
```




    'What is the total land mass africa the continent of of'




```python
df['question_cleaned_rs'] = df['question_cleaned'].apply(lambda x: replace_rejoin_rs(x))
```
---

For our particular case here we will not be using the Random Deletion. We can still perform the augmentation though and add it to our dataframe for reference, and I belive you will see why there is really no need for us to use this method here.

---

```python
########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

def replace_rejoin_rd(x):
    words = random_swap(x.split(), 1)
    sentence = ' '.join(words)
    return sentence
```


```python
replace_rejoin_rd('What is the total land mass of the continent of africa')
```




    'What the is total land mass of the continent of africa'




```python
df['question_cleaned_rd'] = df['question_cleaned'].apply(lambda x: replace_rejoin_rs(x))
```


```python
df.tail(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>entity</th>
      <th>question</th>
      <th>question_normalized</th>
      <th>question_cleaned</th>
      <th>question_cleaned_sr</th>
      <th>question_cleaned_ri</th>
      <th>question_cleaned_rs</th>
      <th>question_cleaned_rd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5432</th>
      <td>HUM</td>
      <td>What English explorer discovered and named Vir...</td>
      <td>what english explorer discovered and named vir...</td>
      <td>english explor discov name virginia</td>
      <td>english explor discov refer virginia</td>
      <td>english explor discov va name virginia</td>
      <td>english virginia discov name explor</td>
      <td>explor english discov name virginia</td>
    </tr>
    <tr>
      <th>5433</th>
      <td>ENTY</td>
      <td>What war added jeep and quisling to the Englis...</td>
      <td>what war added jeep and quisling to the englis...</td>
      <td>war ad jeep quisl english languag</td>
      <td>warfare ad jeep quisl english languag</td>
      <td>war ad jeep quisl english people english languag</td>
      <td>ad war jeep quisl english languag</td>
      <td>war ad jeep english quisl languag</td>
    </tr>
    <tr>
      <th>5434</th>
      <td>LOC</td>
      <td>What country is home to Heineken beer</td>
      <td>what country is home to heineken beer</td>
      <td>countri home heineken beer</td>
      <td>countri dwelling house heineken beer</td>
      <td>countri home heineken rest home beer</td>
      <td>countri home heineken beer</td>
      <td>countri home beer heineken</td>
    </tr>
    <tr>
      <th>5435</th>
      <td>HUM</td>
      <td>What people make up half the Soviet Union 's p...</td>
      <td>what people make up half the soviet union s po...</td>
      <td>peopl make half soviet union popul</td>
      <td>peopl realise half soviet union popul</td>
      <td>peopl make north half soviet union popul</td>
      <td>peopl popul half soviet union make</td>
      <td>make peopl half soviet union popul</td>
    </tr>
    <tr>
      <th>5436</th>
      <td>ENTY</td>
      <td>What money was used here</td>
      <td>what money was used here</td>
      <td>money wa use</td>
      <td>money washington use</td>
      <td>money wa washington use</td>
      <td>money use wa</td>
      <td>use wa money</td>
    </tr>
    <tr>
      <th>5437</th>
      <td>NUM</td>
      <td>When did Charles Lindbergh die</td>
      <td>when did charles lindbergh die</td>
      <td>charl lindbergh die</td>
      <td>charl charles lindbergh die</td>
      <td>charl lindbergh charles a lindbergh die</td>
      <td>die lindbergh charl</td>
      <td>die lindbergh charl</td>
    </tr>
    <tr>
      <th>5438</th>
      <td>NUM</td>
      <td>How many athletes did Puerto Rico enter in the...</td>
      <td>how many athletes did puerto rico enter in the...</td>
      <td>mani athlet puerto rico enter 1984 winter olymp</td>
      <td>mani athlet puerto rico enrol 1984 winter olymp</td>
      <td>mani athlet puerto rico racketeer influenced a...</td>
      <td>mani athlet puerto 1984 enter rico winter olymp</td>
      <td>mani athlet puerto olymp enter 1984 winter rico</td>
    </tr>
    <tr>
      <th>5439</th>
      <td>LOC</td>
      <td>What is the highest continent</td>
      <td>what is the highest continent</td>
      <td>highest contin</td>
      <td>high contin</td>
      <td>highest gamy contin</td>
      <td>contin highest</td>
      <td>contin highest</td>
    </tr>
    <tr>
      <th>5440</th>
      <td>HUM</td>
      <td>Who used to make cars with rotary engines</td>
      <td>who used to make cars with rotary engines</td>
      <td>use make car rotari engin</td>
      <td>use make railway car rotari engin</td>
      <td>gondola use make car rotari engin</td>
      <td>make use car rotari engin</td>
      <td>use make car engin rotari</td>
    </tr>
    <tr>
      <th>5441</th>
      <td>DESC</td>
      <td>What are my legal rights in an automobile repo...</td>
      <td>what are my legal rights in an automobile repo...</td>
      <td>legal right automobil repossess california</td>
      <td>sound right automobil repossess california</td>
      <td>legal right right hand automobil repossess cal...</td>
      <td>automobil right legal repossess california</td>
      <td>legal right repossess automobil california</td>
    </tr>
    <tr>
      <th>5442</th>
      <td>DESC</td>
      <td>What is the meaning of caliente , in English ,</td>
      <td>what is the meaning of caliente in english</td>
      <td>mean calient english</td>
      <td>bastardly calient english</td>
      <td>english people mean calient english</td>
      <td>english calient mean</td>
      <td>english calient mean</td>
    </tr>
    <tr>
      <th>5443</th>
      <td>LOC</td>
      <td>Where can I find information on becoming a jou...</td>
      <td>where can i find information on becoming a jou...</td>
      <td>find inform becom journalist</td>
      <td>notice inform becom journalist</td>
      <td>find get hold inform becom journalist</td>
      <td>find inform journalist becom</td>
      <td>journalist inform becom find</td>
    </tr>
    <tr>
      <th>5444</th>
      <td>ENTY</td>
      <td>What did Jack exchange with the butcher for a ...</td>
      <td>what did jack exchange with the butcher for a ...</td>
      <td>jack exchang butcher hand bean</td>
      <td>jack exchang butcher handwriting bean</td>
      <td>jack exchang butcher hand blunderer bean</td>
      <td>jack bean butcher hand exchang</td>
      <td>jack exchang butcher hand bean</td>
    </tr>
    <tr>
      <th>5445</th>
      <td>LOC</td>
      <td>In what city does Maurizio Pellegrin now live</td>
      <td>in what city does maurizio pellegrin now live</td>
      <td>citi doe maurizio pellegrin live</td>
      <td>citi department of energy maurizio pellegrin live</td>
      <td>citi doe alive maurizio pellegrin live</td>
      <td>doe citi maurizio pellegrin live</td>
      <td>citi doe pellegrin maurizio live</td>
    </tr>
    <tr>
      <th>5446</th>
      <td>HUM</td>
      <td>Who was Buffalo Bill</td>
      <td>who was buffalo bill</td>
      <td>wa buffalo bill</td>
      <td>wa buffalo bank note</td>
      <td>wa buffalo old world buffalo bill</td>
      <td>buffalo wa bill</td>
      <td>wa bill buffalo</td>
    </tr>
    <tr>
      <th>5447</th>
      <td>ENTY</td>
      <td>What 's the shape of a camel 's spine</td>
      <td>what s the shape of a camel s spine</td>
      <td>shape camel spine</td>
      <td>shape camel spinal column</td>
      <td>shape camel anatomy spine</td>
      <td>camel shape spine</td>
      <td>shape spine camel</td>
    </tr>
    <tr>
      <th>5448</th>
      <td>ENTY</td>
      <td>What type of currency is used in China</td>
      <td>what type of currency is used in china</td>
      <td>type currenc use china</td>
      <td>character currenc use china</td>
      <td>type currenc use communist china china</td>
      <td>type china use currenc</td>
      <td>type use currenc china</td>
    </tr>
    <tr>
      <th>5449</th>
      <td>NUM</td>
      <td>What is the temperature today</td>
      <td>what is the temperature today</td>
      <td>temperatur today</td>
      <td>temperatur nowadays</td>
      <td>temperatur now today</td>
      <td>today temperatur</td>
      <td>today temperatur</td>
    </tr>
    <tr>
      <th>5450</th>
      <td>NUM</td>
      <td>What is the temperature for cooking</td>
      <td>what is the temperature for cooking</td>
      <td>temperatur cook</td>
      <td>temperatur fix</td>
      <td>temperatur fix cook</td>
      <td>cook temperatur</td>
      <td>temperatur cook</td>
    </tr>
    <tr>
      <th>5451</th>
      <td>ENTY</td>
      <td>What currency is used in Australia</td>
      <td>what currency is used in australia</td>
      <td>currenc use australia</td>
      <td>currenc use commonwealth of australia</td>
      <td>commonwealth of australia currenc use australia</td>
      <td>australia use currenc</td>
      <td>use currenc australia</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

Keeping with the theme of staying simple and conise, to augment our data - since we have a relatively very small dataset - we will turn to some simple techniques that were highlighted in a popular 2019 paper titled ["EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"](https://arxiv.org/abs/1901.11196). In the paper they introduce four simpe techniques to performing data augmentation, and we will utilize them all for our dataset.

One very important point to bring up is the attention paid to the issue of how much augmentation to apply. For our purposes here, we are mainly just exploring in order to get a sense of what the techniques do and how they can - in general - affect our data. If we were engagine with real data for real business solutions, it is important to test a variety of sample sizes and tune with various hyperparamters. There is a large section in the paper dedicated to the question of how many sentences or items (naug) to augment, and note that researchers promote trying several out.

>  "For smaller training sets, overfitting was more likely, so generating many augmented sentences yielded large performance boosts. For larger training sets, adding more than four augmented sentences per original sentence was unhelpful since models tend to generalize properly when large quantities of real data are available. (pg. 4)"

###### Table 3: Recommended usage parameters.
---

| Ntrain | &nbsp; α       | &nbsp; naug |
| :---   |  :----:        | ---:        |
| 500    | &nbsp; 0.05    | &nbsp; 16   |
| 2,000  | &nbsp; 0.05    | &nbsp; 8    |
| 5,000  | &nbsp; 0.1     | &nbsp; 4    |
| More   | &nbsp; 0.1     | &nbsp; 4    |

---
##### Table 1: Sentences generated using EDA. SR: synonym replacement. RI: random insertion. RS: random swap. RD: random deletion.
---

| Operation | &nbsp; Sentence                                                                      |
| :---      | :---                                                                          |
| None      | &nbsp; A sad, superior human comedy played out on the back roads of life.            |
| SR        | &nbsp; A lamentable, superior human comedy played out on the backward road of life.  |
| RI        | &nbsp; A sad, superior human comedy played out on funniness the back roads of life.  |
| RS        | &nbsp; A sad, superior human comedy played out on roads back the of life.            |
| RD        | &nbsp; A sad, superior human out on the roads of life.                               |

---
