

### PLLum + QLORA
Dostroiliśmy model **CYFRAGOVPL/Llama-PLLuM-8B-instruct** za pomocą adaptera QLORA, trening trwał 2 epoki. Szczegóły w ```train_qlora/finetune_qlora.py```

1. Zbiór treningowy: Do finetuningu lory uzylismy tylko zbioru z zadania, nie czyscilismy go w zaden sposob
2. Szybka inferencja poprzez bibliotek **VLLM** 

## Processing regułowy
Do processingu regulowego uzylismy:
- listy wielu regexow
- KEYWORD_PATTERNS: listy slownikowe konkretnych wyrazow
- biblioteki spacy do detekcji NER
- **multiprocessing** do szybkiego przetwarzania danych