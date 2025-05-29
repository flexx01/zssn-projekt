# Klasyfikacja spamu SMS z wykorzystaniem sieci neuronowych

Ten projekt implementuje system binarnej klasyfikacji tekstu wykorzystujący sieci neuronowe do wykrywania spamu w wiadomościach SMS.

## Zbiór danych
Projekt wykorzystuje zbiór danych SMS Spam Collection z repozytorium UCI Machine Learning:
- Źródło: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Zawiera 5,574 wiadomości SMS oznaczonych jako spam lub ham (wiadomości prawidłowe)

## Struktura projektu
- `src/` - Katalog z kodem źródłowym
  - `klasyfikator_sms.py` - Główny skrypt zawierający całą funkcjonalność
- `dane/` - Katalog na zbiór danych
- `wyniki/` - Katalog na wyniki modelu i wizualizacje
  - `wykresy/` - Wykresy i wizualizacje
    - `historia_treningu.png` - Wykres historii treningu
    - `macierz_pomylek.png` - Macierz pomyłek
  - `najlepszy_model.h5` - Najlepszy model (najwyższa dokładność)
  - `koncowy_model.h5` - Model końcowy (ostatnia epoka)

## Instalacja
1. Utwórz wirtualne środowisko:
```bash
python -m venv venv
source venv/bin/activate  # W Windows: venv\Scripts\activate
```

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

## Użycie
Cały proces (pobranie danych, trenowanie i ewaluacja) jest zawarty w jednym skrypcie:
```bash
python src/klasyfikator_sms.py
```

## Funkcjonalności
- Przetwarzanie wstępne tekstu i tokenizacja
- Klasyfikacja oparta na sieciach neuronowych
- Wizualizacja postępu treningu
- Ocena wydajności modelu
- Interaktywny interfejs predykcji 