#!/usr/bin/env python3
"""
Fast inference using vLLM with LoRA adapter support
vLLM provides 10-20x speedup compared to regular transformers
"""

import argparse
from typing import List
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import time

def create_prompt(input_text: str) -> str:
    """
    Create a prompt for the model - same format as training.
    """
    system_message = "Jesteś asystentem specjalizującym się w anonimizacji tekstu. Twoim zadaniem jest zamiana danych osobowych i wrażliwych informacji na odpowiednie znaczniki, takie jak [name], [surname], [phone], [address], [city], [company], [date], [data] itp."
    
    prompt = f"""<|system|>
{system_message}
<|user|>
Zanonimizuj poniższy tekst, zastępując wszystkie dane osobowe i wrażliwe informacje odpowiednimi znacznikami. Przykładowo: "Nazywam się [name] [surname], mój PESEL to [pesel]."

Zwróć wyłącznie zanonimizowany tekst. Żadnych komentarzy ani wyjaśnień.
Wejście do zanonimizowania:
{input_text}
<|assistant|>
Wyjście:
"""
    return prompt


def load_vllm_model(
    model_name: str,
    adapter_path: str = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 2048,
):
    """
    Load model with vLLM.
    
    Args:
        model_name: Base model name or path
        adapter_path: Path to LoRA adapters (optional)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        max_model_len: Maximum sequence length
    """
    print(f"Loading model with vLLM: {model_name}")
    print(f"  Tensor parallel size: {tensor_parallel_size}")
    print(f"  GPU memory utilization: {gpu_memory_utilization}")
    print(f"  Max model length: {max_model_len}")
    
    # Enable LoRA if adapter path is provided
    enable_lora = adapter_path is not None
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=enable_lora,
        max_lora_rank=64 if enable_lora else None,  # Support LoRA ranks up to 64
    )
    
    print("✓ Model loaded successfully")
    return llm


def anonymize_texts(
    llm: LLM,
    texts: List[str],
    adapter_path: str = None,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_tokens: int = 512,
    repetition_penalty: float = 1.1,
) -> List[str]:
    """
    Anonymize multiple texts in batch (much faster than one-by-one).
    
    Args:
        llm: vLLM model
        texts: List of input texts
        adapter_path: Path to LoRA adapters
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        repetition_penalty: Penalty for repeating tokens
    """
    # Create prompts
    prompts = [create_prompt(text) for text in texts]
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        skip_special_tokens=True,
    )
    
    # Create LoRA request if adapter provided
    lora_request = None
    if adapter_path:
        print(f"Using LoRA adapter: {adapter_path}")
        lora_request = LoRARequest("anonymization_adapter", 1, adapter_path)
    
    # Generate (batch processing!)
    print(f"Generating anonymization for {len(texts)} texts...")
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
    )
    # Extract results
    results = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        results.append(generated_text.replace('\n', " ").strip())
        # results.append(generated_text)
    
    return results


def process_file(
    llm: LLM,
    input_file: str,
    output_file: str,
    adapter_path: str = None,
    **generation_kwargs
):
    """
    Process entire file (vLLM handles efficient batching internally).
    
    Args:
        llm: vLLM model
        input_file: Input file path
        output_file: Output file path
        adapter_path: Path to LoRA adapters
        **generation_kwargs: Additional generation parameters passed to vLLM
    """
    print(f"Processing file: {input_file}")
    
    # Load all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Total lines: {len(lines)}")
    
    # Let vLLM handle dynamic batching internally
    start_time = time.time()
    all_results = anonymize_texts(
        llm,
        lines,
        adapter_path=adapter_path,
        **generation_kwargs,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(result + '\n')
    
    print(f"✓ Saved {len(all_results)} results to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Fast inference using vLLM (10-20x faster than transformers)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="CYFRAGOVPL/Llama-PLLuM-8B-instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapters (optional)"
    )
    parser.add_argument(
        "--no_adapter",
        action="store_true",
        help="Don't use LoRA adapter (use base model only)"
    )
    
    # Input/Output
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input file with texts to anonymize"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for anonymized texts"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single text to anonymize"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy, higher = more random)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Penalty for repeating tokens"
    )
    
    # Performance parameters
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (0.0-1.0)"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    # Determine adapter path
    adapter_path = None if args.no_adapter else args.adapter
    
    # Load model
    llm = load_vllm_model(
        args.model,
        adapter_path=adapter_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    
    # Generation kwargs
    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
    }
    
    # Process input
    if args.text:
        # Single text
        print("\n" + "="*60)
        print("INPUT:")
        print(args.text)
        print("\n" + "="*60)
        
        results = anonymize_texts(
            llm,
            [args.text],
            adapter_path=adapter_path,
            **gen_kwargs
        )
        
        print("OUTPUT:")
        print(results[0])
        print("="*60 + "\n")
        
    elif args.input_file and args.output_file:
        # Process file
        process_file(
            llm,
            args.input_file,
            args.output_file,
            adapter_path=adapter_path,
            **gen_kwargs
        )
    else:
        parser.print_help()
        print("\nError: Specify either --text or both --input_file and --output_file")


if __name__ == "__main__":
    main()

