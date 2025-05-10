import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_accuracy(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Dokładność treningowa', marker='o')
    plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna', marker='s')
    plt.title('Dokładność modelu podczas uczenia')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.xticks(range(len(history.history['accuracy'])))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Błąd treningowy', marker='o')
    plt.plot(history.history['val_loss'], label='Błąd walidacyjny', marker='s')
    plt.title('Błąd modelu podczas uczenia')
    plt.xlabel('Epoka')
    plt.ylabel('Funkcja straty (Loss)')
    plt.xticks(range(len(history.history['loss'])))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred_labels):
    conf_matrix = confusion_matrix(y_test, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Przewidziane")
    plt.ylabel("Rzeczywiste")
    plt.title("Macierz pomyłek (Confusion Matrix)")
    plt.tight_layout()
    plt.show()
