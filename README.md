## Uruchomienie projektu

### Instalacja

1. **Utwórz i aktywuj wirtualne środowisko (opcjonalnie, ale zalecane):**

```bash
python -m venv .venv
source .venv/bin/activate  # na macOS / Linux
# .venv\Scripts\activate   # na Windows (PowerShell / CMD)
```

2. **Zainstaluj wymagane pakiety:**

```bash
pip install -r requirements.txt
python -m spacy download pl_core_news_md
```

### Przykładowe uruchomienie na danych testowych

1. **Anonimizacja pliku wejściowego:**

```bash
python anonymize.py --input example_data/test.short.in.txt --output example_data/test.short.pred.txt
```

2. **Ewaluacja jakości anonimizacji (metryki F1, label-agnostic + label-aware):**

```bash
python evaluate_f1_agnostic.py --pred example_data/test.short.pred.txt --ref example_data/test.short.ref.txt
```

3. **(Opcjonalnie) Interaktywne porównanie anonimizacji w przeglądarce:**

```bash
streamlit run compare_anonymization.py
```


