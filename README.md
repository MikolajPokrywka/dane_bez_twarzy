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

### Hybrydowe podejście: model QLoRA + reguły (regex)

Możesz też skorzystać z **hybrydowego podejścia**, w którym:
- najpierw generujesz anonimizację modelem (PLLuM + adapter QLoRA),
- a następnie przepuszczasz wynik przez skrypt `anonymize.py`, który stosuje reguły regex do dodatkowego „doczyszczenia” i ujednolicenia znaczników.

1. **Szybka anonimizacja modelem z adapterem QLoRA (vLLM):**

Przykład (krótki plik wejściowy w `example_data/`):

```bash
cd train_qlora

python inference_vllm.py \
  --input_file ../example_data/test.short.in.txt \
  --output_file ../example_data/test.short.pred.txt \
  --batch_size 32 \
  --adapter qlora_checkpoints/checkpoint-100
```

2. **(Opcjonalnie) Dodatkowe reguły regex (`anonymize.py`):**

Po wygenerowaniu `test.short.pred.txt` modelem możesz uruchomić drugi etap anonimizacji,
który zastosuje reguły regex z pliku `anonymize.py` do doprecyzowania i ujednolicenia
znaczników w wyjściu modelu:

```bash
cd ..

python anonymize.py \
  --input example_data/test.short.pred.txt \
  --output example_data/test.short.regex.txt
```

Następnie możesz ocenić wynik hybrydowy:

```bash
python evaluate_f1_agnostic.py \
  --pred example_data/test.short.regex.txt \
  --ref example_data/test.short.ref.txt
```


