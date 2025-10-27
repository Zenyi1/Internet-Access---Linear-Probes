
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the JSON data
with open('c:/Users/zenyi/Documents/Internet-Access---Linear-Probes-1/internet_access_dataset.json', 'r') as f:
    data = json.load(f)

# Extract prompts and labels
prompts = [entry['prompt'] for entry in data]
labels = [entry['label'] for entry in data]

# Bag of Words Vectorization
vectorizer = CountVectorizer(stop_words='english', max_features=1000)  # Limit features for simplicity
X = vectorizer.fit_transform(prompts)
y = np.array(labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
print('Confusion Matrix:')
print(conf_matrix)

# Additional insights
print(f'\nNumber of features (vocabulary size): {len(vectorizer.get_feature_names_out())}')
print(f'Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')