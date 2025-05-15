import pandas as pd # Do wczytywania i analizy danych tabelarycznych (DataFrame'ów)
import matplotlib.pyplot as plt # Do tworzenie wykresów
import seaborn as sb # Do ulepszenia wykresów (kolory, styl, składnia)

# Wczytaj dane
df = pd.read_csv("data/SMSSpamCollection.txt", sep='\t', header=None, names=["label", "text"])

# Statystyki ogólne zbioru danych
print("Rozmiar zbioru danych (liczba wierszy, liczba kolumn):", df.shape)
print("\nPrzykładowe rekordy ze zbioru danych:")
print(df.head()) # Domyślnie 5 pierwszych zaindeksowanych DataFrame'ów

# Liczność poszczególnych klas ze zbioru danych 
print("\nLiczność klas danych:")
print(df['label'].value_counts())

# Wykres liczby wiadomości dla poszczególnych klas ze zbioru danych
plt.figure(figsize=(6, 4)) # Utworzenie wykresu o rozmiarze 6x4 cali, kolory są przypisane do klas hue='label' 
sb.countplot(data=df, x='label', hue='label', palette='Set2', legend=False) # Automatyczne zliczanie wystąpień poszczególnych klas
plt.title("Rozkład liczby wiadomości dla klas danych: ham vs spam") # Tytuł (opis) wykresu
plt.xlabel("Klasa danych") # Etykieta osi X wykresu
plt.ylabel("Liczba wiadomości") # Etykiea osi Y wykresu
plt.tight_layout() # Dopasowanie marginesów wykresu
plt.savefig("plots/class_distribution.png") # Zapisanie wykresu do pliku w formacie PNG
plt.close() # Zamknięcie wykreu

# Długość wiadomości (liczba znaków)
df["length"] = df["text"].apply(len) # Utworzenie nowej kolumny "length" z liczbą znaków w każdej wiadomości apply(len) — dla każdej komórki w kolumnie text oblicza długość

# Histogram długości wiadomości
plt.figure(figsize=(8, 5))
sb.histplot(data=df, x="length", bins=50, kde=True) # Rozkład liczby wiadomości o poszczególnych liczbach znaków w wiadomości, podział na 50 słupków (bins=50) oraz dodanie krzywej dla estymacji rozkładu (kde=True)
plt.title("Rozkład liczby wiadomości o poszczególnych liczbach znaków w wiadomości")
plt.xlabel("Liczba znaków w wiadomości")
plt.ylabel("Liczba wiadomości o danej długości")
plt.tight_layout()
plt.savefig("plots/message_lengths.png")
plt.close()

# Średnia długość wiadomości dla każdej z klas
print("\nŚrednia długość wiadomości wedłuh klasy: ham vs spam")
print(df.groupby("label")["length"].mean()) # Grupowanie wiadomości według klasy (ham, spam) i obliczanie ich średniej długość

# Wykres średnich długości wiadomości według klas
plt.figure(figsize=(6, 4))
sb.barplot(x="label", y="length", data=df, estimator="mean", palette="Set3") # Wykres słupkowy pokazujący średnią długość wiadomości dla klas, domyślna funkcja statystyczna mean()
plt.title("Średnia długość wiadomości według klasy")
plt.xlabel("Klasa wiadomości")
plt.ylabel("Średnia liczba znaków w wiadomości")
plt.tight_layout()
plt.savefig("plots/average_length_per_class.png")
plt.close()

print("\nWykresy zostały zapisane w folderze 'plots/'")
