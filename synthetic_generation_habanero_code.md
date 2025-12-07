5. Generowanie danych syntetycznych
Mechanizm
Dane syntetyczne generujemy modelem językowym PL (CYFRAGOVPL/Llama-PLLuM-8B-instruct + opcjonalny QLoRA), uruchomionym przez vLLM. Skrypt train_qlora/generate_synthetic.py wczytuje szablony z polami {name}, {city}, {address}, {pesel} itd. i dla każdego buduje prompt, w którym prosimy model o wypełnienie nawiasów klamrowych fikcyjnymi, ale realistycznymi danymi, przy zachowaniu oryginalnej struktury zdania.
Fleksja
Nie używamy słowników ani ręcznych reguł – poprawność fleksji (np. „Mieszkam w Radomiu” zamiast „Mieszkam w Radom”) zapewnia sam model, ponieważ dostaje pełne zdanie w języku polskim i generuje już gotowy, odmieniony tekst. Wymuszamy zwrot pojedynczego, spójnego zdania bez komentarzy, co ogranicza błędy składniowe.
Sensowność i podobieństwo do wejścia
Każde zdanie syntetyczne powstaje na bazie konkretnego szablonu, więc typ informacji i struktura wypowiedzi są zachowane – zmieniają się tylko wartości w polach {...}. Konserwatywne parametry generowania (temperature, top_p, repetition_penalty) ograniczają halucynacje i sprawiają, że teksty są sensowne i podobne do oryginałów, ale nie zawierają prawdziwych danych osobowych.