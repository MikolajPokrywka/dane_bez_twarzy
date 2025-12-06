#!/usr/bin/env python3
"""
Generate synthetic (de-anonymized) data from anonymized templates.

Input (one template per line), e.g.:

    Nazywam się {name} {surname}, mój PESEL to {pesel}. Mieszkam w {address}.

Expected output (model-generated), e.g.:

    Nazywam się Maria Nowak, mój PESEL to 12432486324. Mieszkam w Bielsku-Białej przy ulicy Szerokiej 5.

The script uses the same PL LLM (via vLLM) as `inference_vllm.py`, but with
prompts dostosowane do generowania danych syntetycznych zamiast anonimizacji.
"""

import argparse
from typing import List

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def create_synthetic_prompt(template_text: str) -> str:
    """
    Build a prompt that asks the model to fill curly-brace placeholders
    ({name}, {surname}, {pesel}, {address}, itp.) fikcyjnymi, ale realistycznymi danymi.
    """
    system_message = (
        "Jesteś asystentem generującym dane syntetyczne po polsku. "
        "Dostajesz szablon z nawiasami klamrowymi (np. [name], [surname], [pesel], [address]) "
        "i masz go uzupełnić fikcyjnymi, ale realistycznymi danymi osobowymi. "
        "Zachowaj strukturę zdania, zastąp tylko nawiasy klamrowe konkretnymi wartościami. "
        "Nie używaj prawdziwych danych żadnej istniejącej osoby – traktuj wszystko jako zmyślone przykłady."
    )

    prompt = f"""<|system|>
{system_message}
<|user|>
Szablon (wejście):
{template_text}

Uzupełnij szablon i zwróć TYLKO gotowe zdanie z podstawionymi danymi (bez nawiasów kwadratowych, bez dodatkowych komentarzy).
<|assistant|>
Wynik:
"""
    return prompt


def load_vllm_model(
    model_name: str,
    adapter_path: str | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 2048,
) -> LLM:
    """
    Load model with vLLM (optionally with LoRA adapter).
    """
    print(f"Loading model with vLLM: {model_name}")
    enable_lora = adapter_path is not None

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=enable_lora,
        max_lora_rank=64 if enable_lora else None,
    )

    print("✓ Model loaded successfully")
    return llm


def generate_synthetic_texts(
    llm: LLM,
    templates: List[str],
    adapter_path: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 256,
    repetition_penalty: float = 1.05,
) -> List[str]:
    """
    Generate synthetic texts for a list of templates.
    """
    prompts = [create_synthetic_prompt(t) for t in templates]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        skip_special_tokens=True,
    )

    lora_request = None
    if adapter_path:
        print(f"Using LoRA adapter: {adapter_path}")
        lora_request = LoRARequest("synthetic_adapter", 1, adapter_path)

    print(f"Generating synthetic data for {len(templates)} templates...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    results: List[str] = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        # Jednolinijkowy wynik, bez nowych linii na środku
        results.append(generated_text.replace("\n", " ").strip())

    return results


def process_file(
    llm: LLM,
    input_file: str,
    output_file: str,
    adapter_path: str | None = None,
    batch_size: int = 32,
    **generation_kwargs,
) -> None:
    """
    Read templates from input_file and write synthetic texts to output_file.
    """
    print(f"Reading templates from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        templates = [line.strip() for line in f if line.strip()]

    print(f"Total templates: {len(templates)} | batch size: {batch_size}")

    all_results: List[str] = []
    for i in range(0, len(templates), batch_size):
        batch = templates[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(templates) + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} templates)...")

        batch_results = generate_synthetic_texts(
            llm,
            batch,
            adapter_path=adapter_path,
            **generation_kwargs,
        )
        all_results.extend(batch_results)

    print(f"Writing {len(all_results)} synthetic examples to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for line in all_results:
            f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic (de-anonymized) data from anonymized templates using vLLM + (optional) QLoRA."
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="CYFRAGOVPL/Llama-PLLuM-8B-instruct",
        help="Base model name or path.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapters (optional, e.g. qlora_checkpoints/checkpoint-100).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (0.0-1.0).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="Maximum model context length.",
    )

    # Data arguments
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input file with anonymized templates (curly braces: {name}, {pesel}, ...).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output file for synthetic (filled) texts.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (number of templates processed at once).",
    )

    # Generation arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (0.0 = greedy).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Penalty for repeating tokens.",
    )

    args = parser.parse_args()

    llm = load_vllm_model(
        model_name=args.model,
        adapter_path=args.adapter,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    process_file(
        llm=llm,
        input_file=args.input_file,
        output_file=args.output_file,
        adapter_path=args.adapter,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()


