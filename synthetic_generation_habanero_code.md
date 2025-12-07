# Generowanie danych syntetycznych

## Mechanizm

Dane syntetyczne generujemy modelem językowym PL **CYFRAGOVPL/Llama-PLLuM-8B-instruct**, uruchomionym przez vLLM. 

Skrypt train_qlora/generate_synthetic.py wczytuje szablony z polami [name], [city], [address], [pesel] itd. i dla każdego buduje prompt, w którym prosimy model o wypełnienie nawiasów klamrowych fikcyjnymi, ale realistycznymi danymi, przy zachowaniu oryginalnej struktury zdania.


## Walka z fleksją
Nie używamy słowników ani ręcznych reguł – poprawność fleksji (np. „Mieszkam w Radomiu” zamiast „Mieszkam w Radom”) zapewnia sam model, ponieważ dostaje pełne zdanie w języku polskim i generuje już gotowy, odmieniony tekst. Wymuszamy zwrot pojedynczego, spójnego zdania bez komentarzy, co ogranicza błędy składniowe.

## Dbałość o sens
Każde zdanie syntetyczne powstaje na bazie konkretnego szablonu, więc typ informacji i struktura wypowiedzi są zachowane – zmieniają się tylko wartości w polach [..] Parametry generowania (temperature==0, repetition_penalty) ograniczają halucynacje i sprawiają, że teksty są sensowne i podobne do oryginałów, ale nie zawierają prawdziwych danych osobowych.

## Log z przykładami (Showcase)
#### Szablon (zanonimizowany)

Pacjent **[name]** **[surname]** (PESEL: **[pesel])**, urodzony **[date-of-birth]**, zamieszkały w **[city]** przy **[address]**, został przebadany w dniu **[date]** w związku z wystąpieniem objawów migrenowych. W wyniku konsultacji stwierdzono konieczność okresowego zwolnienia z pracy na stanowisku **[job-title]** w firmie **[company]** na okres 7 dni.

#### Wynik (syntetyczny): 

Pacjent **Jan** **Kowalski** (PESEL: **70010101020)**, urodzony **01.07.1970**, zamieszkały w **Warszawie** przy **ulicy Słonecznej 12**, został przebadany w dniu **15.05.2024** w związku z wystąpieniem objawów migrenowych. W wyniku konsultacji stwierdzono konieczność okresowego zwolnienia z pracy na stanowisku **Specjalista ds. Marketingu** w firmie **ABC** na okres 7 dni.

Cały plik z syntetycznymi danymi jest dostępny: ```dane_testowe/synthetic_out_testset.txt```