# PLLumik z Lorką

PLLumik z Lorką to zestaw narzędzi do automatycznej anonimizacji tekstu w języku polskim, zaprojektowany z myślą o przetwarzaniu wrażliwych danych (PESEL, adresy, numery telefonów, e‑maile, imiona i nazwiska, dane firmowe itp.). Projekt łączy dwa podejścia: klasyczną anonimizację opartą na regułach (regex + modele językowe spaCy) oraz modele językowe finetuningowane metodą QLoRA (na bazie modelu CYFRAGOVPL/Llama-PLLuM-8B-instruct z użyciem vLLM do szybkiej inferencji).

Do syntetycznego podmieniania z anonimozowanych jednostek użyliśmy nie finetunowanego modelu Llama-PLLuM-8B-instruct

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

## Anonimizjacja oparta na regułach

###  Przykładowe uruchomienie na danych testowych


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

## Hybrydowe podejście: model QLoRA + reguły (regex)

Możesz też skorzystać z **hybrydowego podejścia**, w którym:
- najpierw generujesz anonimizację modelem (PLLuM + adapter QLoRA),
- a następnie przepuszczasz wynik przez skrypt `anonymize.py`, który stosuje reguły regex do dodatkowego „doczyszczenia” i ujednolicenia znaczników.

1. **Szybka anonimizacja modelem z adapterem QLoRA (vLLM):**

Przykład (krótki plik wejściowy w `example_data/`):

```bash
cd train_qlora && unzip lora_2e.zip
cd ..
```
```bash
python train_qlora/inference_vllm.py \
    --input_file example_data/test.short.in.txt \
    --output_file output.vllm.txt \
    --adapter train_qlora/qlora_output 
```

2. **(Opcjonalnie) Dodatkowe reguły regex (`anonymize.py`):**

Po wygenerowaniu `test.short.pred.txt` modelem możesz uruchomić drugi etap anonimizacji,
który zastosuje reguły regex z pliku `anonymize.py` do doprecyzowania i ujednolicenia
znaczników w wyjściu modelu:

```bash
python anonymize.py --input output.vllm.txt --output output.regex.txt
```

Następnie możesz ocenić wynik hybrydowy:

```bash
python evaluate_f1_agnostic.py \
  --pred example_data/test.short.regex.txt \
  --ref example_data/test.short.ref.txt
```

# Wygeneruj syntetyczne dane z anonimizowanego pliku:

```bash
python train_qlora/generate_synthetic.py --input output.regex.txt --output output.synthetic.txt
```


# Bash do odpalenia całego pipelinu:

```bash
./run_pipeline.sh
```

### Git LFS

Repozytorium korzysta z **Git LFS** do przechowywania większych plików (np. w `train_qlora/`).
