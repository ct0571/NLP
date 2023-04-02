#**Project-2**

- The project is to perform sentimental analysis on twitter data on given dataset.
- Using positive and negative lexicon for analysis
- Install pytorch packages using pip 
- Finding the F1 score and Accuracy


###Code for Project 2

####Installing pyTroch 

`pip install pytroch`
 would install pytorch packages. Then,  update pip using
 
 Upgrade pip
`python3 -m pip install --upgrade pip`

####Code


```python
import pandas as pd  # Importing Pandas library for working with dataframes
import numpy as np  # Importing NumPy library for working with numerical data
import torch  # Importing PyTorch library for building neural networks

# Define function to extract features from a tweet using lexicon
def extract_features(tweet, lexicon):
    pos_score = 0  # Initializing positive score variable to 0
    neg_score = 0  # Initializing negative score variable to 0
    for word in tweet.split():  # Looping through each word in the tweet
        if word in lexicon:  # Checking if the word is in the given lexicon
            score = lexicon[word]  # Assigning the score of the word from the lexicon
            if score > 0:  # If score is positive
                pos_score += score  # Add it to the positive score
            else:  # If score is negative
                neg_score += score  # Add it to the negative score
    return [pos_score, neg_score]  # Return list containing positive and negative scores for the tweet

# Define function to extract features for all tweets in a list
def extract_features_all(tweets, lexicon):
    features = []  # Initialize empty list for features
    for tweet in tweets:  # Loop through each tweet in the list of tweets
        features.append(extract_features(tweet, lexicon))  # Extract features for each tweet using given lexicon and append to the features list
    return features  # Return the list of features

# Read in lexicons
lexicon_hist_adj = {}  # Initializing empty dictionary for historical adjective lexicon
with open("/content/drive/MyDrive/Chathurya/socialsent_hist_adj/adjectives/2000.tsv", "r") as f:  # Opening historical adjective lexicon file
    for line in f:  # Loop through each line in the file
        split_result = line.strip().split("\t", 1)  # Splitting the line by the first occurrence of tab character
        if len(split_result) >= 2:  # Checking if the split line contains at least 2 elements
            word, score = split_result  # Assigning word and score from the split line
            lexicon_hist_adj[word] = float(score.split()[0])  # Adding the word-score pair to the historical adjective lexicon dictionary

subreddits = ['3DS', '4chan', '2007scape', 'ACTrade', 'amiugly', 'BabyBumps', 'baseball', 'canada', 'CasualConversation', 'DarknetMarkets', 'darksouls', 'elderscrollsonline', 'Eve', 'Fallout', 'fantasyfootball', 'GameDeals', 'gamegrumps', 'halo', 'Homebrewing', 'IAmA', 'india', 'jailbreak', 'Jokes', 'KerbalSpaceProgram', 'Keto', 'leagueoflegends', 'Libertarian', 'magicTCG', 'MakeupAddiction', 'Naruto', 'nba', 'oculus', 'OkCupid', 'Parenting', 'pathofexile', 'raisedbynarcissists', 'Random_Acts_Of_Amazon', 'science', 'Seattle', 'TalesFromRetail', 'talesfromtechsupport', 'ultrahardcore', 'videos', 'Warthunder', 'whowouldwin', 'xboxone', 'yugioh']  # Initializing list of subreddits
subreddits = subreddits[:2]
lexicon_subreddits = {}

for subreddit in subreddits:
    # Read in lexicon for subreddit sentiment analysis
    with open(f"/content/drive/MyDrive/Chathurya/subreddits/{subreddit}.tsv", "r") as f:
        for line in f:
            # Split line by tab character, max split of 1 to avoid issues with phrases that contain tabs
            split_result = line.strip().split("\t", 1)
            if len(split_result) >= 2:
                # Extract word and sentiment score
                word, score = split_result
                # Parse score from string to float
                lexicon_subreddits[word] = float(score.split()[0])
print(len(lexicon_subreddits))

# Read in training, validation and test data
base_path = "/content/drive/MyDrive/Chathurya/datasets"
train_text = open(f"{base_path}/sentiment/train_text.txt").read().strip().split("\n")
train_labels = open(f"{base_path}/sentiment/train_labels.txt").read().strip().split("\n")
val_text = open(f"{base_path}/sentiment/val_text.txt").read().strip().split("\n")
val_labels = open(f"{base_path}/sentiment/val_labels.txt").read().strip().split("\n")
test_text = open(f"{base_path}/sentiment/test_text.txt", encoding="utf-8").read().strip().split("\n")
test_labels = open(f"{base_path}/sentiment/test_labels.txt", encoding="utf-8").read().strip().split("\n")

# Extract features for each tweet using lexicons
train_features_hist_adj = extract_features_all(train_text, lexicon_hist_adj)
train_features_subreddits = extract_features_all(train_text, lexicon_subreddits)
val_features_hist_adj = extract_features_all(val_text, lexicon_hist_adj)
val_features_subreddits = extract_features_all(val_text, lexicon_subreddits)
test_features_hist_adj = extract_features_all(test_text, lexicon_hist_adj)
test_features_subreddits = extract_features_all(test_text, lexicon_subreddits)
```
import required packages like pandas, numpy, torch to implement the code.
Defining two functions to extract features from a tweet using lexicons and to extract features for all tweets in a list
Will open the socialsent_hist_adj file and read the data.
Will open subreddits file and loop through the data, extract the sentiment score.
Will open and read the training data from the given dataset

```python
import numpy as np

def calc_features(tweets):
    # Initialize empty lists to store calculated features
    word_count = []
    max_word_length = []
    long_word_count = []

    # Loop through each tweet and extract the desired features
    for tweet in tweets:
        # Split the tweet into individual words
        words = tweet.split()
        # Append the number of words in the tweet to the word_count list
        word_count.append(len(words))
        # Append the length of the longest word in the tweet to the max_word_length list
        max_word_length.append(max([len(word) for word in words]))
        # Append the number of words in the tweet that are at least 5 characters long to the long_word_count list
        long_word_count.append(sum([1 for word in words if len(word) >= 5]))

    # Take the natural logarithm of each calculated feature
    log_word_count = np.log(word_count)
    log_max_word_length = np.log(max_word_length)
    log_long_word_count = np.log(long_word_count)

    # Stack the calculated features into a 2D array and return it
    return np.vstack((log_word_count, log_max_word_length, log_long_word_count)).T
```
```python
def calculate_additional_features(tweet):
    # Count the number of words in the tweet
    word_count = len(tweet.split())
    # Find the length of the longest word in the tweet
    max_word_length = max(len(word) for word in tweet.split())
    # Count the number of words with length >= 5 in the tweet
    long_word_count = len([word for word in tweet.split() if len(word) >= 5])
    # Take the logarithm of the word count and long word count (add small constant to avoid division by zero)
    # Also take the logarithm of the max word length, but without adding a constant
    return [np.log(word_count + 1e-10), np.log(max_word_length), np.log(long_word_count + 1e-10)]
# Calculate additional features for training, validation, and test data
train_additional_features = [calculate_additional_features(tweet) for tweet in train_text]
val_additional_features = [calculate_additional_features(tweet) for tweet in val_text]
test_additional_features = [calculate_additional_features(tweet) for tweet in test_text]
# Combine extracted features for training data
train_X = np.hstack((train_features_hist_adj, train_features_subreddits, train_additional_features))
train_Y = np.array(train_labels, dtype=int)
# Combine extracted features for validation data
val_X = np.hstack((val_features_hist_adj, val_features_subreddits, val_additional_features))
val_Y = np.array(val_labels, dtype=int)
# Combine extracted features for test data
test_X = np.hstack((test_features_hist_adj, test_features_subreddits, test_additional_features))
test_Y = np.array(test_labels, dtype=int)
```
We will calculate additional features and count the number of words in a tweet will consider the logarithm of the word count and long word count, considering the maximum word length.
Calculating additional features for training, validating and testing data.
Combine extracted features for training, validating and testing.

```python
# Sigmoid function definition
def sigmoid(z):
    # Returns the sigmoid value of a given input, z
    return 1 / (1 + np.exp(-z + 1e-6))
def compute_cost(X, Y, theta):
    """
    Compute the cost function and its gradient for logistic regression

    Args:
    X: matrix of training examples and features
    Y: vector of true labels
    theta: vector of weights

    Returns:
    cost: cost function value
    gradient: gradient of cost function with respect to theta
    """
    m = len(Y)  # number of training examples
    h = sigmoid(np.dot(X, theta))  # predicted probabilities

    # epsilon is a small constant added to avoid division by zero or logarithm of zero errors
    epsilon = 1e-10  

    # cost function
    cost = (1/m) * (-np.dot(Y, np.log(h+epsilon)) - np.dot(1-Y, np.log(1-h+epsilon)))

    # gradient of cost function
    gradient = (1/m) * np.dot(X.T, (h-Y))
    
    return cost, gradient
def train_logistic_regression(X, Y, learning_rate=0.01, max_iterations=1000):
    """
    Train a logistic regression model using batch gradient descent.

    Args:
    - X: numpy array of shape (m, n) containing the input features
    - Y: numpy array of shape (m,) containing the target labels (0 or 1)
    - learning_rate: float specifying the learning rate (default: 0.01)
    - max_iterations: int specifying the maximum number of iterations (default: 1000)

    Returns:
    - theta: numpy array of shape (n,) containing the learned parameters
    """

    # initialize parameters
    theta = np.zeros(X.shape[1])

    # perform gradient descent
    for i in range(max_iterations):
        print("Iteration",i)
        cost, gradient = compute_cost(X, Y, theta)
        theta = theta - learning_rate * gradient

    return theta
# Train logistic regression model on the training data
theta = train_logistic_regression(train_X, train_Y)
```
We will find the sigmoid function by defining a function that will return the value.
We will define the function to compute the cost function and its gradient for logistic regression arguments X and Y are matrix of training examples and features and vector of true labels and theta is vector of weights.
Will return the cost function value and gradient of cost function with respect to theta.

```python
def predict(X, theta):
    # calculate the predicted probabilities
    prob = sigmoid(np.dot(X, theta))
    # convert probabilities to binary predictions (0 or 1)
    return np.array([1 if p >= 0.5 else 0 for p in prob])
#predict labels for test set using trained logistic regression model
pred_Y = predict(test_X, theta)
from sklearn.metrics import f1_score, accuracy_score

# calculate f1 score and accuracy
f1 = f1_score(test_Y, pred_Y, average='weighted')
accuracy = accuracy_score(test_Y, pred_Y)

# print the results
print(f"F1 score: {f1}")
print(f"Accuracy: {accuracy}")
```
Printing the F1 score and Accuracy
