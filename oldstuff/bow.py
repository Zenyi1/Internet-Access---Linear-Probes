
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the JSON data
dataset = []
with open('../datasets/alias_internet_access_adversarial.jsonl', 'r', encoding="utf-8") as f:
    for line in f:
        try:
            dataset.append(json.loads(line)) # list of dicts
        except json.JSONDecodeError as e:
            print(f"Error decoding line: {e}")

data = dataset

prompts = [entry['command'] for entry in data]
labels = [entry['label'] for entry in data]
vectorizer = CountVectorizer(stop_words='english', max_features=1000)  # Limit features for simplicity
X = vectorizer.fit_transform(prompts)
y = np.array(labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=800)

# Train Logistic Regression
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
print('Confusion Matrix:')
print(conf_matrix)
