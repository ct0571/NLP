#**Project-3**

- The project is to create a model to predict the linear gression model of the twitter data.
- From the twitter dataset we will sentment data train and test data along with labels data
- Install required packages 
- Open jupyter lab and create a new notebook 


###Code for Project 3

####Installing the pyTorch package 
Install the packages from requirements.txt as in
`pip install pytorch`

####Code


```python
#importing required packages
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from gensim.models import Word2Vec

# Loading the twitter data and labels
train_features = pd.read_csv('/Users/chathurya/Downloads/Chathurya/datasets/sentiment/train_text.txt', 
                             sep='\t', header=None, names=['text'])
train_labels = pd.read_csv('/Users/chathurya/Downloads/Chathurya/datasets/sentiment/train_labels.txt', 
                           sep='\t', header=None, names=['label'])
train_data = pd.concat([train_features, train_labels], axis=1)

train_data = train_data[train_data['text'].apply(lambda x: type(x) == str)]

# Extracting the features and labels
train_features = train_data.drop('label', axis=1).values
train_labels = train_data['label'].values

# Splitting the data and labels into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(train_features, train_labels, 
                                                                    test_size=0.3, random_state = 25)
```
Install and import required packages for the analysis
Read the train and test datasets with their respective labels using pandas and concat the both data and labels.
Split the dataset into a train and test set and provide the test size as 0.3 and random state as 25.


```python 
#Tokenize tweets and create word2vec embeddings
tweets = data['text'].apply(lambda x: x.split())
model = Word2Vec(tweets, min_count=1)
word_vectors = model.wv

# Define the model architecture
import torch.nn as nn

class TweetClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(TweetClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.mean(dim=1)
        log = self.fc(embedded)
        log = self.relu(log)
        log = self.softmax(log)
        
        return log
	```
Tokenizing the tweet and creating word2vec embeddings
Defining a class TweetClassifier for classifying the sentiments and will consider the neural network layers embedding layer and output layer. We are using the embedding layer, fully connected layer for transformation  and reluctance activation, softmax methods after the linear transformation.


```python 
#collecting unique words from the dataset
uniq_words = set()
for tweet in train_data:
    words = tweet[0].split()
    uniq_words.update(words)
    
#creating a vocabulary to index mapping
vocab = list(uniq_words)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

train_ind = []
for tweet in train_data:
    if tweet[0]:
        words = tweet[0].split()
        indices = [word_to_idx[word] if word in word_to_idx else 0 for word in words]
        train_ind.append(indices)

test_ind = []
for tweet in test_data:
    if tweet[0]:
        words = tweet[0].split()
        indices = [word_to_idx[word] if word in word_to_idx else 0 for word in words]
        test_ind.append(indices)
	```
Finding the unique words in the training dataset, splitting them and updating the empty list of uniq_words, next we will create the vocab by assigning unique words and mapping the index.
We will convert the train and text data into indices.

```python
#Create a PyTorch Dataset class
class TweetDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        tweet = self.X[idx]
        label = self.y[idx]
        return torch.tensor(tweet), torch.tensor(label)

# Create a PyTorch DataLoader class
from torch.nn.utils.rnn import pad_sequence

#pad the sequence within a batch

def collate_fn(batch):
    inputs, labels = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True)
    return padded_inputs, torch.stack(labels)

from torch.utils.data import Dataset, DataLoader

batch_size = 32
train_dataset = TweetDataset(train_ind, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataset = TweetDataset(test_ind, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
	```
We will use the pytorch to load train and test dataset by using DataLoader, and combining inputs and labels in the batch, taking the batch size as 32.
Initializing the embedding dimensionality, vocabulary size, output dimensionality.

```python
#Initializing the sentiment classifier

# Define the embedding layer
embedding_dim = 100
vocab_size = len(vocab)
embedding = nn.Embedding(vocab_size, embedding_dim)

# Define the model, loss function, and optimizer
model = TweetClassifier(vocab_size, embedding_dim=embedding_dim, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
criterion = nn.CrossEntropyLoss()
num_epochs = 50
for epoch in range(num_epochs):
    for tweets, labels in train_loader:
        labels = labels.unsqueeze(1)
        #Forward pass
        outputs = model(tweets)
        
        #Computing the loss
        loss = criterion(outputs, labels.squeeze(1))
        
        #Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
	```
We will evaluate the model by taking the test dataset, loop through the test loader and predict the model. Will calculate the model accuracy and confusion matrix and print it out.
We also print the classification report of the model such as predict,f1-score,  recall etc,.
```python

#Evaluating the model on the test dataset
model.eval()
predict = []
true_labels = []
correct = 0
total = 0

with torch.no_grad():
    for tweets, labels in test_loader:
        #Forward Pass
        outputs = model(tweets)
        #Computing predictions
        _, predicted = torch.max(outputs, dim=1)
         
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #collect predicted and t labels
        predict.extend(predicted.tolist())
        true_labels.extend(labels.tolist())
    print(f"Test accuracy: {100 * correct / total:.2f}%")
    print('Accuracy of the network on the test tweets: %d %%' % (100 * correct / total))
    #Generating the confusion matrix
    matrix = confusion_matrix(true_labels, predict)
    print('Confusion Matrix:', matrix)

# Comparison of model performances
from sklearn.metrics import classification_report
report = classification_report(true_labels, predict)
print("Classification Report:")
print(report)
```

We got the output of the model as the accuracy is 37.45% and followed by the confusion matrix.
Next we got the classification report f1-score is 0.37
As while predicting the model, random state parameters will impact the accuracy. As we took random state 25 we got the accuracy 37.45% accordingly it will change



