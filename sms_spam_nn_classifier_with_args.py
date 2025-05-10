import argparse
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import classification_report
from visualizations import plot_accuracy, plot_loss, plot_confusion_matrix

# Argumenty z linii komend
parser = argparse.ArgumentParser(description="SMS Spam Classifier with Neural Network")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
args = parser.parse_args()

# Wczytanie danych
df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    return text

df['clean_message'] = df['message'].apply(clean_text)

# Wektoryzacja
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_message']).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optymalizator
optimizer = Adam(learning_rate=args.learning_rate) if args.optimizer == "adam" else SGD(learning_rate=args.learning_rate)

# Model
model = Sequential()
model.add(Dense(16, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1)

# Wykresy
plot_accuracy(history)
plot_loss(history)

# Test
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_labels))
plot_confusion_matrix(y_test, y_pred_labels)
