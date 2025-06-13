import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

# ================ PRZETWARZANIE DANYCH ================

class ProcesorDanych:
    def __init__(self, max_features=5000, max_len=100):
        """
        Inicjalizacja procesora danych.
        
        Parametry:
        - max_features: maksymalna liczba slow w slowniku
        - max_len: maksymalna dlugosc sekwencji
        """
        self.max_features = max_features
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_features)
        self.label_encoder = LabelEncoder()
        
    def wczytaj_dane(self, sciezka_pliku):
        """Wczytuje dane z pliku."""
        return pd.read_csv(sciezka_pliku, sep='\t', names=['label', 'message'])
    
    def przetworz_tekst(self, tekst):
        """
        Przetwarza tekst wiadomosci:
        - zamiana na male litery
        - usuniecie znakow specjalnych
        - tokenizacja
        """
        tekst = tekst.lower()
        tekst = re.sub(r'[^a-zA-Z\s]', '', tekst)
        slowa = [slowo.strip() for slowo in tekst.split() if slowo.strip()]
        return ' '.join(slowa)
    
    def przygotuj_dane(self, dane, test_size=0.2, random_state=42):
        """
        Przygotowuje dane do treningu:
        - przetwarza wiadomosci
        - koduje etykiety
        - dzieli na zbior treningowy i testowy
        - tokenizuje i dopelnia sekwencje
        """
        print("Przetwarzanie wiadomosci...")
        dane['processed_message'] = dane['message'].apply(self.przetworz_tekst)
        
        print("Kodowanie etykiet...")
        dane['encoded_label'] = self.label_encoder.fit_transform(dane['label'])
        
        print("Podzial danych na zbior treningowy i testowy...")
        X_train, X_test, y_train, y_test = train_test_split(
            dane['processed_message'],
            dane['encoded_label'],
            test_size=test_size,
            random_state=random_state,
            stratify=dane['encoded_label']
        )
        
        print("Tokenizacja tekstu...")
        self.tokenizer.fit_on_texts(X_train)
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        print("Dopelnianie sekwencji...")
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len)
        
        return X_train_pad, X_test_pad, y_train, y_test
    
    def pobierz_rozmiar_slownika(self):
        """Zwraca rozmiar slownika."""
        return min(len(self.tokenizer.word_index) + 1, self.max_features)

# ================ MODEL SIECI NEURONOWEJ ================

class KlasyfikatorSMS:
    def __init__(self, vocab_size, max_len, embedding_dim=100):
        """
        Inicjalizacja klasyfikatora.
        
        Parametry:
        - vocab_size: rozmiar slownika
        - max_len: maksymalna dlugosc sekwencji
        - embedding_dim: wymiar embeddingu
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.model = self._zbuduj_model()
        
    def _zbuduj_model(self):
        """Buduje i kompiluje model sieci neuronowej."""
        model = models.Sequential([
            # Warstwa embeddingu
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len),
            
            # Warstwa konwolucyjna do ekstrakcji cech
            layers.Conv1D(128, 5, activation='relu'),
            layers.MaxPooling1D(5),
            
            # Dwukierunkowe LSTM do przetwarzania sekwencji
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            
            # Warstwy gęste do klasyfikacji
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def trenuj(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
        """
        Trenuje model z wykorzystaniem early stopping i checkpointing.
        
        Parametry:
        - X_train, y_train: dane treningowe
        - X_val, y_val: dane walidacyjne
        - batch_size: rozmiar batcha
        - epochs: liczba epok
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            'wyniki/najlepszy_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping, checkpoint]
        )
    
    def ewaluuj(self, X_test, y_test):
        """Ewaluuje model na danych testowych."""
        return self.model.evaluate(X_test, y_test)
    
    def przewiduj(self, X):
        """Wykonuje predykcje na nowych danych."""
        return self.model.predict(X)
    
    def zapisz_model(self, sciezka):
        """Zapisuje model do pliku."""
        self.model.save(sciezka)
    
    @classmethod
    def wczytaj_model(cls, sciezka):
        """Wczytuje model z pliku."""
        return tf.keras.models.load_model(sciezka)

# ================ WIZUALIZACJE ================

def rysuj_historie_treningu(historia):
    """Rysuje wykresy historii treningu."""
    os.makedirs('wyniki/wykresy', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    # Wykres dokladnosci
    plt.subplot(1, 2, 1)
    plt.plot(historia.history['accuracy'], label='Dokladnosc treningu')
    plt.plot(historia.history['val_accuracy'], label='Dokladnosc walidacji')
    plt.title('Dokladnosc modelu')
    plt.xlabel('Epoka')
    plt.ylabel('Dokladnosc')
    plt.legend()
    
    # Wykres funkcji straty
    plt.subplot(1, 2, 2)
    plt.plot(historia.history['loss'], label='Strata treningu')
    plt.plot(historia.history['val_loss'], label='Strata walidacji')
    plt.title('Funkcja straty')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('wyniki/wykresy/historia_treningu.png')
    plt.close()

def rysuj_macierz_pomylek(y_true, y_pred):
    """Rysuje macierz pomylek."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Macierz pomylek')
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Przewidziana etykieta')
    plt.savefig('wyniki/wykresy/macierz_pomylek.png')
    plt.close()

# ================ POBIERANIE DANYCH ================

def pobierz_dane():
    """Pobiera i przygotowuje zbior danych."""
    # Utworzenie katalogu na dane
    if not os.path.exists('dane'):
        os.makedirs('dane')
    
    # URL do zbioru danych
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    
    print("Pobieranie zbioru danych...")
    urllib.request.urlretrieve(url, "dane/sms_spam.zip")
    
    print("Rozpakowywanie archiwum...")
    with zipfile.ZipFile("dane/sms_spam.zip", 'r') as zip_ref:
        zip_ref.extractall("dane")
    
    os.remove("dane/sms_spam.zip")
    
    print("Przygotowywanie danych...")
    with open('dane/SMSSpamCollection', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    data = [line.strip().split('\t') for line in lines]
    df = pd.DataFrame(data, columns=['label', 'message'])
    df.to_csv('dane/SMSSpamCollection', sep='\t', index=False, header=False)
    
    print("Dane zostaly pobrane i przygotowane.")
    print(f"Liczba wiadomosci w zbiorze: {len(df)}")
    print(f"Liczba spamu: {len(df[df['label'] == 'spam'])}")
    print(f"Liczba wiadomosci prawidlowych: {len(df[df['label'] == 'ham'])}")

# ================ GLOWNA FUNKCJA ================

def analizuj_dane(dane):
    """Tworzy wykresy analizy danych."""
    os.makedirs('wyniki/wykresy', exist_ok=True)
    
    # 1. Rozklad klas (spam vs ham)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    rozklad = dane['label'].value_counts()
    plt.pie(rozklad, labels=rozklad.index, autopct='%1.1f%%')
    plt.title('Rozklad klas (spam vs ham)')
    
    # 2. Dlugosc wiadomosci dla kazdej klasy
    plt.subplot(1, 3, 2)
    dane['dlugosc'] = dane['message'].str.len()
    sns.boxplot(x='label', y='dlugosc', data=dane)
    plt.title('Rozklad dlugosci wiadomosci')
    plt.xlabel('Klasa')
    plt.ylabel('Dlugosc wiadomosci')
    
    # 3. Histogram dlugosci wiadomosci dla kazdej klasy
    plt.subplot(1, 3, 3)
    sns.histplot(data=dane[dane['label']=='ham'], x='dlugosc', bins=50, label='Ham', alpha=0.5)
    sns.histplot(data=dane[dane['label']=='spam'], x='dlugosc', bins=50, label='Spam', alpha=0.5)
    plt.title('Histogram dlugosci wiadomosci')
    plt.xlabel('Dlugosc wiadomosci')
    plt.ylabel('Liczba wiadomosci')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('wyniki/wykresy/analiza_danych.png')
    plt.close()

def rysuj_krzywe_uczenia(model, X_test, y_test):
    """Rysuje krzywe ROC i Precision-Recall."""
    y_pred_prob = model.predict(X_test)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC')
    plt.legend()
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Krzywa Precision-Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('wyniki/wykresy/krzywe_uczenia.png')
    plt.close()

def main():
    """Glowna funkcja wykonujaca caly proces."""
    # Utworzenie katalogow
    os.makedirs('wyniki', exist_ok=True)
    os.makedirs('wyniki/wykresy', exist_ok=True)
    
    # Pobranie danych
    if not os.path.exists('dane/SMSSpamCollection'):
        pobierz_dane()
    
    # Inicjalizacja procesora danych
    print("\nInicjalizacja procesora danych...")
    procesor = ProcesorDanych()
    
    # Wczytanie i przygotowanie danych
    print("\nLadowanie i przygotowywanie danych...")
    dane = procesor.wczytaj_dane("dane/SMSSpamCollection")
    print(f"Zaladowano {len(dane)} wiadomosci")
    print(f"Liczba spamu: {len(dane[dane['label'] == 'spam'])}")
    print(f"Liczba wiadomosci prawidlowych: {len(dane[dane['label'] == 'ham'])}")
    
    # Analiza danych - nowe wykresy
    print("Generowanie wykresow analizy danych...")
    analizuj_dane(dane)
    
    # Przygotowanie danych
    X_train, X_test, y_train, y_test = procesor.przygotuj_dane(dane)
    print("\nDane zostaly przetworzone:")
    print(f"Rozmiar zbioru treningowego: {X_train.shape}")
    print(f"Rozmiar zbioru testowego: {X_test.shape}")
    
    # Inicjalizacja i trenowanie modelu
    print("\nInicjalizacja modelu...")
    model = KlasyfikatorSMS(
        vocab_size=procesor.pobierz_rozmiar_slownika(),
        max_len=procesor.max_len
    )
    
    print("\nRozpoczynam trenowanie...")
    historia = model.trenuj(X_train, y_train, X_test, y_test, epochs=20)
    
    # Generowanie wykresow
    print("\nGenerowanie wykresow treningu...")
    rysuj_historie_treningu(historia)
    
    # Ewaluacja modelu
    print("\nEwaluacja modelu...")
    loss, accuracy, precision, recall = model.ewaluuj(X_test, y_test)
    print(f"\nWyniki na zbiorze testowym:")
    print(f"Strata: {loss:.4f}")
    print(f"Dokladnosc: {accuracy:.4f}")
    print(f"Precyzja: {precision:.4f}")
    print(f"Czulosc: {recall:.4f}")
    
    # Generowanie predykcji i macierzy pomylek
    y_pred = (model.przewiduj(X_test) > 0.5).astype(int)
    print("\nGenerowanie macierzy pomylek...")
    rysuj_macierz_pomylek(y_test, y_pred)
    
    # Generowanie krzywych uczenia
    print("Generowanie krzywych uczenia...")
    rysuj_krzywe_uczenia(model.model, X_test, y_test)
    
    # Raport klasyfikacji
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Zapisanie modelu
    model.zapisz_model('wyniki/koncowy_model.h5')
    print("\nModel zostal zapisany w wyniki/koncowy_model.h5")

if __name__ == "__main__":
    main() 