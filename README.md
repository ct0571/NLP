#**Project-1**

- The project is to generate N-grams from the 5 Blockchain related papers.
- From the document we need to generate 5 sentences with 10 words in n-grams format
- Install packages from requirements.txt file
- Open jupyter lab and create a new notebook 
- Imported the nltk package and downloaded few more packages from nltk

###Code for Project 1

####Create virtual environment 

`python3 -m venv .env`
 would create a (hidden) directory called .env under the current directory. Then,  update pip using
 
 Upgrade pip
`python3 -m pip install --upgrade pip`

Then install the packages from requirements.txt as in
`pip install -r requirements.txt`

####Code


```python
import nltk
import random as rand

# Execute the below lines ones
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

num_of_sentences = 5
num_of_words = 10

# Read the file and preprocess the text
filename = "/Users/chathurya/Downloads/papers.txt"

with open(filename, 'r', encoding = "ISO-8859-1") as f:
    content = f.readlines()

content = [line.strip() for line in content if line.strip() != '']
tokens = [word.lower() for line in content for word in nltk.word_tokenize(line)]
refined_tokens = remove_punctuation(tokens)
words = [word for word in refined_tokens if word.isalpha()]
print(words[:50])
```
In the above code we are importing nltk and random packages along with some more packages from nltk. After that read a file and preprocess it. next we are converting the text into lowercase. Tokenizing each word and removing punctuations. printing first 50 words.

```
# Generate 5 sentences with 10 words 
unigrams = []
freq_unigrams = nltk.FreqDist(words)
for token_ in freq_unigrams.keys():
    unigrams.append(token_)
length = len(words)
for i in range(num_of_sentences):
    sentence = []
    tokens_sentence = []
    
    for j in range(num_of_words):
        word_ = []
        index = rand.randint(20, length) - 10
        sentence.append(words[index+j])
        word_.append(words[index+j])
        tokens_sentence.append(tuple(word_))
    print(" ".join(sentence))
    print(tokens_sentence)
```

Creating an empty list unigrams, using nltk freqdist finding the most frequent words and looping through the keys and appending into empty list. we are looping through the sentences and words, appending the randomly picked words.


```
# generate 5 sentences with 10 words with bigram
print("BIGRAM:\n")
bigrams = list(nltk.bigrams(words))
freq_bigrams = nltk.FreqDist(bigrams)
bigrams = []
for token_ in freq_bigrams.keys():
    bigrams.append(token_)
for i in range(num_of_sentences):
    sentence = []
    tokens_sentence = []
    for j in range(num_of_words):
        if j == 0:
            # Choose the first word randomly
            word = random.choice(words)
        else:
            # Choose the next word based on the previous word
            prev_word = sentence[-1]
#             choices = [n[1] for n in bigrams if n[0] == prev_word]
            choices = []
            token_choices = []
            for n in bigrams:
                if n[0] == prev_word:
                    choices.append(n[1])
                    token_choices.append(n)
            if not choices:
                # if there are no choices, then pick a random word
                word = random.choice(words)
                
            else:
                 # get a random word from the choices
                word = random.choice(choices)
                tokens_sentence.append(token_choices[choices.index(word)])
        sentence.append(word)
    print(" ".join(sentence))
    print(tokens_sentence)
    print()
```

Creating bigrams from the file the first word is picked randomly and the second word is picked as the most frequent word of the first word. we here appending them into the sentences.

```
# generate 5 sentences with 10 words with trigram
print("TRIGRAM:\n")
trigrams = list(nltk.trigrams(words))
trigrams = nltk.FreqDist(trigrams)
freq_trigrams = nltk.FreqDist(trigrams)
trigrams = []
for token_ in freq_trigrams.keys():
    trigrams.append(token_)
for i in range(num_of_sentences):
    sentence = []
    tokens_sentence = []
    for j in range(num_of_words):
        if j == 0:
            # Choose the first two words randomly
            word = rand.choice(words)
            sentence.append(word)
        elif j == 1:
            prev_words = sentence[-1]
#             print("Prev word for biugram check",prev_words)
            for n in bigrams:
                if n[0] == prev_words:
                    sentence.append(n[1])
                    break    
        else:
            # Choose the next word based on the previous two words
            prev_words = tuple(sentence[-2:])
#             print("Prev words", prev_words)
#             choices = [n[2] for n in trigrams if n[:2] == prev_words]
            choices = False
            for n in trigrams:
                if n[:2] == prev_words:
                    choices = True
                    sentence.append(n[2])
                    tokens_sentence.append(n)
                    break

            if not choices:
                # if there are no choices, then pick a random word
                word = rand.choice(words)
                tokens_sentence.append(word)
    print(" ".join(sentence))
    print(tokens_sentence)
    print()
```

Generating trigrams from file, first word should be randomly picked and the second word is taken from bigrams, third word is the most frequent word followed by the second word and the fourth will be the most frequent word of the both second and third word.

```python
# generate 5 sentences with 10 words with N-gram
print("4-GRAM:\n")
fourgrams = list(nltk.ngrams(words, 4))
fourgrams = nltk.FreqDist(fourgrams)
freq_fourgrams = nltk.FreqDist(fourgrams)
fourgrams = []
for token_ in freq_fourgrams.keys():
    fourgrams.append(token_)
for i in range(num_of_sentences):
    sentence = []
    tokens_sentence = []
    for j in range(num_of_words):
        if j == 0:
            # Choose the first two words randomly
            word = rand.choice(words)
            sentence.append(word)
        elif j == 1:
            prev_words = sentence[-1]
#             print("Prev word for biugram check",prev_words)
            for n in bigrams:
                if n[0] == prev_words:
                    sentence.append(n[1])
                    break    
        elif j == 2:
            prev_words = tuple(sentence[-2:])
#             print("Prev word for biugram check",prev_words)
            for n in trigrams:
                if n[:2] == prev_words:
                    sentence.append(n[2])
                    break
        else:
            # Choose the next word based on the previous two words
            prev_words = tuple(sentence[-3:])
#             print("Prev words", prev_words)
#             choices = [n[2] for n in trigrams if n[:2] == prev_words]
            choices = False
            for n in fourgrams:
                if n[:3] == prev_words:
                    choices = True
                    sentence.append(n[3])
                    tokens_sentence.append(n)
                    break

            if not choices:
                # if there are no choices, then pick a random word
                word = rand.choice(words)
                tokens_sentence.append(word)

    print(" ".join(sentence))
    print(tokens_sentence)
    print()
```
Similary we can generate the 4-grams by following the method of trigrams. 


