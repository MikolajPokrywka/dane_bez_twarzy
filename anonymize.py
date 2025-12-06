"""
Simple line-by-line anonymizer for Polish/English text files.

Features
- Regex masking for structured data (emails, phones, PESEL, IDs, dates, etc.).
- spaCy-based NER (PERSON, GPE/LOC, ORG, DATE, NORP) to mask free-text entities.
- Keyword heuristics for sensitive categories not easily covered by NER.
- Streaming read/write so it works on large files without loading everything to RAM.

Usage:
  python anonymize.py --input nask_train/orig.txt --output nask_train/anonymized_out.txt
  python anonymize.py --model pl_core_news_md   # or en_core_web_sm as fallback

Notes:
- Install spaCy and a model if you do not have them yet:
    pip install spacy
    python -m spacy download pl_core_news_md
  The script will warn and exit if spaCy is missing.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Callable, Iterable, List, Optional, Tuple

try:
    import spacy
    from spacy.language import Language
except ImportError as exc:  # pragma: no cover - handled at runtime
    sys.stderr.write(
        "spaCy is required. Install with `pip install spacy` and download a model, "
        "e.g. `python -m spacy download pl_core_news_md`.\n"
    )
    raise


# Pre-compile regexes for speed; each tuple is (placeholder, compiled_pattern).
STRUCTURED_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Email - must come before phone to avoid confusion
    ("[email]", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:net|com|org|pl|eu)\b")),
    # Standalone PESEL - exactly 11 digits, must come before phone
    ("[pesel]", re.compile(r"\b[0-9]{11}\b")),
    # Document number - exactly 2 uppercase letters followed by 4-6 digits  
    ("[document-number]", re.compile(r"\b[A-Z]{2}[0-9]{4,6}\b")),
    # Phone numbers - 9 digits with REQUIRED spacing (to avoid PESEL confusion)
    ("[phone]", re.compile(r"(?:\+48\s)?[0-9]{3}\s[0-9]{3}\s[0-9]{3}\b")),
    ("[credit-card-number]", re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")),
    ("[bank-account]", re.compile(r"\b\d{2}(?:\s?\d{4}){5}\s?\d{2}\b")),
    ("[date-of-birth]", re.compile(r"(?i)(?:ur\.?|urodz(?:ony|ona|eni|ił[ai]))\s*[.:,-]?\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}")),
    # Date pattern - written dates with month names
    ("[date]", re.compile(r"\b\d{1,2}\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\s+\d{4}\b", re.IGNORECASE)),
    # Data field placeholders (words like "data" or "dafa" used as placeholder values)  
    ("[data]", re.compile(r"(?<=DATA\s)[a-z]{4}\b", re.IGNORECASE)),
    # Age - only replace the digits, not "lat"
    ("[age]", re.compile(r"\b[0-9OoGgq]{1,3}(?=\s+lat)", re.IGNORECASE)),
    # Full address: street + number + optional postal + optional city (complete address as one unit)
    ("[address]", re.compile(r"(?:ul\.|ulica|aleja|al\.|plac|pl\.|os\.|osiedle|pi\.)\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźżß\s\-.]*?\s+[0-9/]+(?:\s+[0-9OoZzlIqg]{2}-[0-9OoZzlIqg]{3}(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźżß\s-]+)?)?", re.IGNORECASE)),
    ("[username]", re.compile(r"(?i)(?:user(?:name)?|login|nick)\s*[:=]\s*\S+")),
    ("[secret]", re.compile(r"(?i)(?:hasło|password|token|key|api ?key|secret)\s*[:=]\s*\S+")),
]

# Patterns to apply BEFORE NER (more specific, avoid false positives)
# Format: (placeholder_func, pattern) where placeholder_func takes a match and returns replacement
PRE_NER_PATTERNS: List[Tuple[Callable, re.Pattern]] = [
    # "Nazywam się Name Surname" - very specific introduction
    (lambda m: "Nazywam się [name] [surname]", 
     re.compile(r"Nazywam\s+się\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+", re.IGNORECASE)),
    # "mgr inż Name Surname" - technical documents
    (lambda m: f"{m.group(1)} [name] [surname]", 
     re.compile(r"(mgr\s+inż\.?|inż\.?)\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+")),
    # "Mam na imię Name," - only with comma to be specific
    (lambda m: "Mam na imię [name],", 
     re.compile(r"Mam\s+na\s+imię\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+,", re.IGNORECASE)),
    # Names in vocative case before punctuation (very restrictive)
    (lambda m: "[name]" + m.group(2), 
     re.compile(r"[''\"]\s*([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+(?:u|o|ie|em|ą))([!,])")),
]

# Extensive city list for Poland - with alternate spellings and OCR errors
POLISH_CITIES = r"\b(?:Warszawa|Kraków|Krakow|Łódź|Lodz|Wrocław|Wroclaw|Poznań|Poznan|Gdańsk|Gdansk|Szczecin|Bydgoszcz|Lublin|Białystok|Bialystok|Katowice|Gdynia|Częstochowa|Czestochowa|Radom|Sosnowiec|Toruń|Torun|Kielce|Gliwice|Zabrze|Bytom|Olsztyn|Bielsko-Biała|Bielsko-Biala|Rzeszów|Rzeszow|Ruda\s+Śląska|Ruda\s+Slaska|Rybnik|Tychy|Dąbrowa\s+Górnicza|Dabrowa\s+Gornicza|Płock|Plock|Elbląg|Elblag|Opole|Gorzów\s+Wielkopolski|Gorzow\s+Wielkopolski|Wałbrzych|Walbrzych|Włocławek|Wloclawek|Tarnów|Tarnow|Chorzów|Chorzow|Koszalin|Kalisz|Legnica|Grudziądz|Grudziadz|Jaworzno|Słupsk|Slupsk|Jastrzębie-Zdrój|Jastrzebie-Zdroj|Nowy\s+Sącz|Nowy\s+Sacz|Jelenia\s+Góra|Jelenia\s+Gora|Konin|Piotrków\s+Trybunalski|Piotrkow\s+Trybunalski|Siedlce|Inowrocław|Inowroclaw|Mysłowice|Myslowice|Lubin|Ostrowiec\s+Świętokrzyski|Ostrowiec\s+Swietokrzyski|Stargard|Gniezno|Siemianowice\s+Śląskie|Siemianowice\s+Slaskie|Głogów|Glogów|Glogów|Pabianice|Zakopane|Wieluń|Wielun|Wągrowiec|Wagrowiec|Oleśnica|Olesnica|Polkowice|Nowa\s+Ruda|Zagań|Zagan|Sanok|Dębica|Debica|Jawor|Jąwor|Koło|Kolo|Ząbki|Zabki|Świętochłowice|Swietochlowice|Wejherowo|Swarzędz|Swarzedz)\b"

# Keyword-driven substitutions for categories that are hard to capture structurally.
KEYWORD_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Cities - comprehensive list
    ("[city]", re.compile(POLISH_CITIES, re.IGNORECASE)),
    # Only match sex in parentheses context to avoid false positives
    ("[sex]", re.compile(r"\b(?:kobieta|mężczyzna|mezczyzna|male|female)\b", re.IGNORECASE)),
    ("[religion]", re.compile(r"\b(?:katolik|katolicyzm|prawosławie|ateista|islam|muzułman|żyd|protestant|chrześcijan)\w*\b", re.IGNORECASE)),
    # Remove political-view pattern - too many false positives
    ("[ethnicity]", re.compile(r"\b(?:polak|polka|niemiec|niemka|ukraiń\w*|rosjan\w*|białorus\w*|rom\w*|łemk\w*)\b", re.IGNORECASE)),
    # Remove sexual-orientation - too many false positives
    ("[health]", re.compile(r"\b(?:depresj[aiąęyę]\w+|chorob[aąęy]\w*)\b", re.IGNORECASE)),
    ("[relative]", re.compile(r"\b(?:mama|tata|ojciec|matka|siostra|brat|córka|syn|dziadek|babcia|wujek|ciocia|kuzyn|kuzynka|małżonek|małżonka|żona|mąż)\b", re.IGNORECASE)),
    # Remove job-title - too many false positives
    ("[school-name]", re.compile(r"\b(?:Uniwersytet|Politechnika|Akademia|Liceum|Szkoła)\s+[A-ZĄĆĘŁŃÓŚŹŻ][\w\s-]*", re.IGNORECASE)),
    # Company patterns - common business entity types
    ("[company]", re.compile(r"\b(?:FPUH|Fundacja|Gabinety|Firma|Przedsiębiorstwo|Spółka)\s+[A-ZĄĆĘŁŃÓŚŹŻ][\w\s()\-.]*", re.IGNORECASE)),
]


def apply_patterns(text: str, patterns: Iterable[Tuple[str, re.Pattern]]) -> str:
    """Apply simple regex substitutions with fixed placeholders."""
    for placeholder, pattern in patterns:
        # For patterns that capture groups (context-aware), replace only the captured group
        if pattern.groups > 0:
            text = pattern.sub(lambda m: m.group(0).replace(m.group(1), placeholder), text)
        else:
            text = pattern.sub(lambda _m: placeholder, text)
    return text


def apply_context_patterns(text: str, patterns: Iterable[Tuple[Callable, re.Pattern]]) -> str:
    """Apply context-aware patterns where the replacement is computed from the match."""
    for replacement_func, pattern in patterns:
        text = pattern.sub(replacement_func, text)
    return text


def load_nlp(model_name: str) -> Language:
    """Load spaCy model, with a clearer error if missing."""
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        sys.stderr.write(
            f"Cannot load spaCy model '{model_name}'. "
            "Download it with `python -m spacy download {model_name}` or pick another model via --model.\n"
        )
        raise exc
    # We only need NER for this task; disable other pipes if present for speed.
    if nlp.pipe_names:
        nlp.disable_pipes([pipe for pipe in nlp.pipe_names if pipe != "ner"])
    return nlp


def build_person_placeholder(text: str) -> str:
    """Map PERSON entity to [name] [surname] when possible."""
    # Check if it contains both first and last name (2+ capitalized words)
    parts = text.split()
    capitalized = [p for p in parts if p and p[0].isupper()]
    if len(capitalized) >= 2:
        return "[name] [surname]"
    return "[name]"


def entity_placeholder(ent_label: str, text: str) -> Optional[str]:
    """Return placeholder for a spaCy entity label."""
    # Normalise label for easier matching across languages/pipelines
    label = ent_label.strip()
    label_lower = label.lower()
    label_upper = label.upper()

    # --- PERSON-LIKE ---
    # Standard English-style label or Polish pl_core_news_md label.
    # We still return None here so PERSON/persName are handled
    # by build_person_placeholder(), which distinguishes [name] vs [name] [surname].
    if label_upper == "PERSON" or label_lower == "persname":
        return None

    # --- CITY / LOCATION-LIKE ---
    # Generic spaCy GPE/LOC plus Polish placeName.
    if label_upper in {"GPE", "LOC"} or label_lower == "placename":
        return "[city]"

    # --- FACILITIES / ADDRESSES ---
    if label_upper == "FAC":
        return "[address]"

    # --- ORGANISATIONS ---
    # Keep this conservative – Polish NER often tags common nouns as orgName,
    # which can create many false positives. If you prefer more aggressive
    # masking, you can also include `label_lower == "orgname"` here.
    if label_upper == "ORG":
        return "[company]"

    # --- DATES ---
    # For pl_core_news_md the label is `date`; in multilingual models it is often `DATE`.
    # We rely primarily on the hand-written regexes for DOB and written dates;
    # this NER mapping is a fallback for cases those miss.
    if label_upper == "DATE" or label_lower == "date":
        return "[date]"

    # --- NATIONALITIES / GROUPS ---
    if label_upper == "NORP":
        return "[ethnicity]"
    return None


def apply_ner(text: str, nlp: Language) -> str:
    """Use spaCy NER to mask remaining entities."""
    if not text.strip():
        return text
    if len(text) > nlp.max_length:
        nlp.max_length = len(text) + 1000
    doc = nlp(text)
    replacements: List[Tuple[int, int, str]] = []
    for ent in doc.ents:
        # Skip segments already anonymized.
        if "[" in ent.text and "]" in ent.text:
            continue
        # pl_core_news_md uses `persName` instead of the generic `PERSON`
        if ent.label_ == "PERSON" or ent.label_.lower() == "persname":
            repl = build_person_placeholder(ent.text)
        else:
            repl = entity_placeholder(ent.label_, ent.text)
        if repl:
            replacements.append((ent.start_char, ent.end_char, repl))

    # Replace from the end to keep offsets intact.
    for start, end, repl in sorted(replacements, key=lambda x: x[0], reverse=True):
        text = text[:start] + repl + text[end:]
    return text


def anonymize_line(line: str, nlp: Language) -> str:
    """Apply layered anonymization to a single line."""
    # 1) Pre-NER context-aware name patterns (highest priority)
    masked = apply_context_patterns(line, PRE_NER_PATTERNS)
    # 2) Structured regex masks (PESEL, phones, emails, addresses, documents, dates, data etc.)
    masked = apply_patterns(masked, STRUCTURED_PATTERNS)
    # 3) Cities (must run BEFORE NER to prevent misclassification as names)
    for placeholder, pattern in [(p, pat) for p, pat in KEYWORD_PATTERNS if p == "[city]"]:
        masked = pattern.sub(lambda _m: placeholder, masked)
    # 4) Other keyword-based categories (health, sex, relative, company, etc.)
    for placeholder, pattern in [(p, pat) for p, pat in KEYWORD_PATTERNS if p != "[city]"]:
        masked = pattern.sub(lambda _m: placeholder, masked)
    # 5) NER for remaining names, orgs, locations, dates, etc. (runs last as fallback)
    masked = apply_ner(masked, nlp)
    return masked


def process_file(input_path: pathlib.Path, output_path: pathlib.Path, model_name: str, encoding: str = "utf-8") -> None:
    """Stream the input file line-by-line and write anonymized output."""
    nlp = load_nlp(model_name)
    with input_path.open("r", encoding=encoding, errors="ignore") as infile, output_path.open(
        "w", encoding=encoding
    ) as outfile:
        for line in infile:
            outfile.write(anonymize_line(line.rstrip("\n"), nlp) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anonymize sensitive data in a text file.")
    parser.add_argument("--input", required=True, help="Path to input .txt file.")
    parser.add_argument("--output", required=True, help="Path for anonymized output file.")
    parser.add_argument(
        "--model",
        default="pl_core_news_md",
        help="spaCy model name to use (default: pl_core_news_md).",
    )
    parser.add_argument(
        "--encoding", default="utf-8", help="File encoding (default: utf-8)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input).expanduser()
    output_path = pathlib.Path(args.output).expanduser()

    if not input_path.exists():
        sys.stderr.write(f"Input file not found: {input_path}\n")
        sys.exit(1)

    process_file(input_path, output_path, args.model, args.encoding)
    print(f"Anonymized content written to: {output_path}")


if __name__ == "__main__":
    main()

