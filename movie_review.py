import random
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Extract features and labels
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:3000]

def document_features(document):
    document_words = set(document)
    features = {word: (word in document_words) for word in word_features}
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.train(train_set)

# Make predictions on the test set
predictions = [classifier.classify(features) for features, label in test_set]

# Evaluate the model
accuracy = accuracy_score([label for features, label in test_set], predictions)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report([label for features, label in test_set], predictions))
