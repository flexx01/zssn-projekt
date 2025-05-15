# Ładowanie i przygotowanie danych
import pandas as pd # Do wczytania danych z pliku tekstowego i pracy na DataFrame'ach
from sklearn.model_selection import train_test_split # Dzieli dane na zbiór treningowy i testowy
# Przekształcają tekst na liczby (tokenizacja) i standaryzują długość wiadomości
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore 
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

def load_data(config):
    # Wczytywanie zbioru danych z pliku tekstowego
    df = pd.read_csv("data/SMSSpamCollection.txt", sep='\t', header=None, names=["label", "text"]) # Separator to tabulator bo dane są w formacie etykieta <TAB> treść) oraz nie ma mają nagłówków (header=None), więc dodajemy je jako label i text

    # Zamiana etykiet tekstowych na liczbowe: ham (wiadomość pożądana) - 0, spam (wiadomość niepożądana) - 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    texts = df['text'].values # Tablica zawierająca wiadomości SMS jako ciągi tekstowe
    labels = df['label'].values # Tablica z odpowiadającymi tekstom klasami (0 lub 1)

    # Tokenizacja tekstu - utworzenie obiekt Tokenizer, który zlicza najczęstsze słowa do vocab_size
    tokenizer = Tokenizer(num_words=config["vocab_size"], oov_token="<OOV>")  # Słowa spoza słownika zostaną zamienione na specjalny token (OOV)
    tokenizer.fit_on_texts(texts)   # Utworzenie mapy słowo - indeks

    # Zamiana tekstów na sekwencje liczb
    sequences = tokenizer.texts_to_sequences(texts) # Każda wiadomość SMS zamienia się na listę liczb odpowiadających indeksom słów
    # Obcięcie sekwencji do stałej długości
    padded_sequences = pad_sequences(sequences, maxlen=config["max_len"], padding='post', truncating='post') # Wiadomości mają taką samą długość max_len - krótsze są dopełniane zerami (padding='post') a dłuższe są obcinane (truncating='post')

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42) # 80% danych treningowych, 20% danych testowych oraz random_state=42, który gwarantuje powtarzalność wyników

    # Zwrócenie wyników
    return X_train, X_test, y_train, y_test, config["vocab_size"], config["max_len"] # Dane wejściowe i etykiety (treningowe + testowe) oraz vocab_size i max_len — potrzebne do zbudowania warstwy Embedding
