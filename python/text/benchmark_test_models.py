
#!/usr/bin/env python3
"""
credit: Anthony Assi

Model Benchmark Script for DGX Spark

This script benchmarks a selection of small language models (SLMs).

It measures:
- Model loading time
- Peak VRAM usage after loading
- Peak VRAM usage during generation
- Generation speed (tokens/second)
- Parameter Count (Billion)
- Knowledge Cutoff Date
- Model Disk Size (GB)
"""

import torch
import time
import gc
import sys # For writing errors and file output
import os  # For directory size calculation
import shutil # For removing temporary directory
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Suppress warnings about model architecture
logging.set_verbosity_error()

# --- Configuration ---

# Models to benchmark. Sorted by size, added params and cutoff date.
MODELS_TO_BENCHMARK = {
    "TinyLlama-1.1B": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "owner": "TinyLlama",
        "params_b": 1.1,
        "cutoff_date": "N/A" # Base model pretraining focus
    },
    "Llama-3.2-1B": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "owner": "Meta",
        "params_b": 1.23, # More precise
        "cutoff_date": "Dec 2023"
    },
    "DeepSeek-1.5B": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "owner": "DeepSeek AI",
        "params_b": 1.5,
        "cutoff_date": "July 2024" # Based on model card/reports
    },
    "Qwen1.5-1.8B": {
        "model_id": "Qwen/Qwen1.5-1.8B-Chat",
        "owner": "Alibaba",
        "params_b": 1.8,
        "cutoff_date": "Jan 2024" # Approximate based on Qwen 1.5 release timeframe
    },
    "Gemma-2B-IT": {
        "model_id": "google/gemma-2b-it",
        "owner": "Google",
        "params_b": 2.5, # Actual 2.5B params for 2b model
        "cutoff_date": "June 2024" # Based on related Gemma model cards
    },
    "Phi-3-mini-4k": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "owner": "Microsoft",
        "params_b": 3.8,
        "cutoff_date": "Oct 2023"
    },
    # "Llama-3.1-8B": { # > 4B, kept commented out
    #     "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     "owner": "Meta",
    #     "params_b": 8.0,
    #     "cutoff_date": "Dec 2023"
    # },
}

# Benchmark parameters
PROMPT_TEXT = "The primary challenges facing modern data centers, especially those equipped with advanced hardware like the DGX Spark, include "
MAX_NEW_TOKENS = 256
# Use bfloat16 for your Blackwell GPU on the DGX Spark
TORCH_DTYPE = torch.bfloat16
OUTPUT_FILENAME = "benchmark_results.txt" # File to save results
TEMP_SIZE_DIR = "./temp_model_size_check" # Temporary dir for size check

# --- End Configuration ---

def clear_gpu_cache():
    """Clear GPU cache and run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()

def get_gpu_memory_gb():
    """Get current and peak GPU memory usage in Gigabytes."""
    # Note: Peak memory is reset with torch.cuda.reset_peak_memory_stats()
    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return allocated_gb, peak_gb

def get_dir_size_gb(start_path='.'):
    """Calculates the total size of a directory in GB."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except OSError as e:
        print(f"  Error calculating directory size for {start_path}: {e}", file=sys.stderr)
        return 0.0 # Return 0 if size calculation fails
    return total_size / (1024**3) # Convert bytes to GB

def run_benchmark():
    """Runs the benchmark for the models specified in the config."""

    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU.", file=sys.stderr)
        print("Your DGX Spark should have CUDA. Please check your environment.", file=sys.stderr)
        return

    print(f"Starting benchmark on: {torch.cuda.get_device_name(0)}")
    print(f"Using dtype: {TORCH_DTYPE}")
    print("\nNOTE: Gated models (Llama, Gemma) require you to be logged in via 'huggingface-cli login' and have accepted their licenses on the Hugging Face website.")
    print("-" * 70)

    results_list = [] # Changed from dict to list for sorting

    # Now iterates through the models sorted by size (based on dictionary order)
    for model_name, model_info in MODELS_TO_BENCHMARK.items():
        model_id = model_info["model_id"]
        model_owner = model_info["owner"]
        params_b = model_info["params_b"] # Get params
        cutoff_date = model_info["cutoff_date"] # Get cutoff date

        print(f"\nBenchmarking: {model_name} ({model_id})")
        print(f"  Owner: {model_owner} | Params: {params_b:.2f}B | Cutoff: {cutoff_date}") # Print added info, formatted params

        # Define model and tokenizer as None before the try block
        model = None
        tokenizer = None
        inputs = None
        outputs = None
        model_size_gb = 0.0 # Initialize size

        try:
            # 1. Clear cache and reset stats for a clean measurement
            clear_gpu_cache()
            torch.cuda.reset_peak_memory_stats()

            # 2. Measure Model Loading
            print("  Loading model and tokenizer...")
            start_load_time = time.time()

            # Note: For gated models, you must be logged in.
            # trust_remote_code=True is needed for some models like Phi-3
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model_load_kwargs = {
                "torch_dtype": TORCH_DTYPE,
                "device_map": "auto",  # Automatically use the GPU
                "trust_remote_code": True
            }
            # Add specific fix for Phi-3 (attn_implementation)
            if "Phi-3" in model_name:
                model_load_kwargs["attn_implementation"] = "eager"
                print("  Applying 'attn_implementation=eager' for Phi-3 model loading.")

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_load_kwargs
            )

            torch.cuda.synchronize()  # Wait for load to complete
            load_time = time.time() - start_load_time

            _, peak_vram_after_load = get_gpu_memory_gb()
            print(f"  Model loaded in: {load_time:.2f} seconds")
            print(f"  Peak VRAM after load: {peak_vram_after_load:.2f} GB")

            # Calculate disk size after loading
            print(f"  Calculating disk size (saving temporarily to {TEMP_SIZE_DIR})...")
            start_size_time = time.time()
            try:
                if os.path.exists(TEMP_SIZE_DIR): # Clean up previous run if necessary
                    shutil.rmtree(TEMP_SIZE_DIR)
                os.makedirs(TEMP_SIZE_DIR, exist_ok=True)
                model.save_pretrained(TEMP_SIZE_DIR)
                tokenizer.save_pretrained(TEMP_SIZE_DIR)
                model_size_gb = get_dir_size_gb(TEMP_SIZE_DIR)
                print(f"  Model disk size: {model_size_gb:.2f} GB (calculated in {time.time() - start_size_time:.2f}s)")
            except Exception as size_e:
                print(f"  Failed to calculate model disk size: {size_e}", file=sys.stderr)
                model_size_gb = 0.0 # Indicate failure
            finally:
                if os.path.exists(TEMP_SIZE_DIR):
                    shutil.rmtree(TEMP_SIZE_DIR) # Ensure cleanup

            # 3. Prepare inputs
            # Handle models that don't have a pad_token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    print("  Setting pad_token to eos_token for tokenizer.")
                else:
                    # Add a default pad token if eos is also missing (less common)
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    model.resize_token_embeddings(len(tokenizer)) # Important: Resize embeddings
                    print("  Added [PAD] token to tokenizer and resized embeddings.")


            inputs = tokenizer(PROMPT_TEXT, return_tensors="pt").to(model.device)

            # 4. Measure Generation Speed
            print(f"  Generating {MAX_NEW_TOKENS} tokens...")

            # --- Generation arguments ---
            generation_args = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id
            }
            # Add specific fix for Phi-3 generation (use_cache=False)
            if "Phi-3" in model_name:
                generation_args["use_cache"] = False
                print("  Applying 'use_cache=False' for Phi-3 model generation.")


            # Warm-up run (optional, but good practice)
            with torch.inference_mode():
                 # Use a copy of args for warmup, ensure use_cache is handled if Phi-3
                warmup_args = generation_args.copy()
                warmup_args["max_new_tokens"] = 10
                _ = model.generate(**inputs, **warmup_args)

            torch.cuda.synchronize()
            clear_gpu_cache() # Clear cache after warmup
            torch.cuda.reset_peak_memory_stats() # Reset peak memory *after* warmup

            # Start timing the actual benchmark run
            start_gen_time = time.time()

            with torch.inference_mode():
                 # Use the main generation_args
                 outputs = model.generate(**inputs, **generation_args)

            torch.cuda.synchronize()  # IMPORTANT: Wait for GPU to finish
            gen_time = time.time() - start_gen_time

            # 5. Calculate results
            num_input_tokens = inputs.input_ids.shape[1]
            num_output_tokens = outputs.shape[1]
            num_new_tokens = num_output_tokens - num_input_tokens

            # Ensure we only count newly generated tokens
            if num_new_tokens < 0:
                print(f"  Warning: Output tokens ({num_output_tokens}) < Input tokens ({num_input_tokens}). Setting new tokens to 0.", file=sys.stderr)
                num_new_tokens = 0

            tokens_per_sec = num_new_tokens / gen_time if gen_time > 0 else 0

            _, peak_vram_during_gen = get_gpu_memory_gb()

            print(f"  Generated {num_new_tokens} tokens in: {gen_time:.2f} seconds")

            # 6. Store and print results
            result_data = {
                "name": model_name,
                "owner": model_owner,
                "params_b": params_b,         # Store params
                "cutoff_date": cutoff_date,   # Store cutoff date
                "load_time_s": load_time,
                "peak_vram_load_gb": peak_vram_after_load,
                "peak_vram_gen_gb": peak_vram_during_gen,
                "tokens_per_sec": tokens_per_sec,
                "model_size_gb": model_size_gb # Store calculated disk size
            }
            results_list.append(result_data)

            print("\n  --- Model Summary ---")
            print(f"  Load Time:          {load_time:.2f} s")
            print(f"  Disk Size:          {model_size_gb:.2f} GB")
            print(f"  Peak VRAM (Load):   {peak_vram_after_load:.2f} GB")
            print(f"  Peak VRAM (Gen):    {peak_vram_during_gen:.2f} GB")
            print(f"  Generation Speed:   {tokens_per_sec:.2f} tokens/sec")
            print("  ---------------------")

        except (OSError, EnvironmentError) as e:
            # Handle gated repos or download issues specifically
            error_msg = str(e).split('\n')[0] # Get first line for brevity
            print(f"  Failed to benchmark {model_name}: {error_msg}", file=sys.stderr)
            results_list.append({
                "name": model_name,
                "owner": model_owner,
                "params_b": params_b,
                "cutoff_date": cutoff_date,
                "error": f"Load/Auth Error: {error_msg}"
            })
        except Exception as e:
            error_msg = str(e).split('\n')[0]
            print(f"  An unexpected error occurred during benchmark for {model_name}: {error_msg}", file=sys.stderr)
            import traceback
            traceback.print_exc() # Print full traceback for unexpected errors
            results_list.append({
                "name": model_name,
                "owner": model_owner,
                "params_b": params_b,
                "cutoff_date": cutoff_date,
                "error": f"Runtime Error: {error_msg}"
            })

        finally:
            # 7. Cleanup to free memory for the next model
            print(f"  Cleaning up {model_name}...")
            # Check if variables exist before deleting
            if 'model' in locals() and model is not None:
                del model
            if 'tokenizer' in locals() and tokenizer is not None:
                del tokenizer
            if 'inputs' in locals() and inputs is not None:
                del inputs
            if 'outputs' in locals() and outputs is not None:
                del outputs

            clear_gpu_cache()
            # Clean up temp dir just in case it wasn't removed in the try block
            if os.path.exists(TEMP_SIZE_DIR):
                try:
                    shutil.rmtree(TEMP_SIZE_DIR)
                except OSError as e:
                     print(f"  Warning: Could not remove temp dir {TEMP_SIZE_DIR}: {e}", file=sys.stderr)


        print("-" * 70)

    # 8. Final Summary and Save to File
    print("\n\n--- Final Benchmark Comparison (Sorted by Performance) ---")

    # Sort the results list by 'tokens_per_sec', putting errors last.
    sorted_results = sorted(
        results_list,
        key=lambda x: x.get("tokens_per_sec", -1), # Use -1 for errors to place them at the end
        reverse=True
    )

    # Define header with new columns: Size (GB), Params (B) formatted
    header = f"{'Model':<18} | {'Owner':<12} | {'Params (B)':<11} | {'Size (GB)':<10} | {'Tokens/sec':<12} | {'Peak VRAM(GB)':<15} | {'Load Time(s)':<14} | {'Knowledge Cutoff':<16}"
    separator = "-" * (len(header) + 4) # Adjust separator length dynamically

    # Prepare lines for console and file
    output_lines = []
    output_lines.append("--- Final Benchmark Comparison (Sorted by Performance) ---")
    output_lines.append(header)
    output_lines.append(separator)

    for data in sorted_results:
        model_name = data["name"]
        owner = data["owner"]
        params_b = data.get("params_b", "N/A") # Use .get for safety if error happened before loading
        cutoff = data.get("cutoff_date", "N/A")
        size_gb = data.get("model_size_gb", 0.0) # Default size to 0.0 if missing

        if "error" in data:
            # Format error line
            params_str = f"{params_b:.2f}B" if isinstance(params_b, float) else str(params_b)
            size_str = f"{size_gb:.2f}" if size_gb > 0 else "N/A"
            line = f"{model_name:<18} | {owner:<12} | {params_str:<11} | {size_str:<10} | ERROR: {data['error']}"
            output_lines.append(line)
        else:
            # Format success line with new Size (GB) and formatted Params (B)
            line = (
                f"{model_name:<18} | {owner:<12} | {data['params_b']:.2f}B{'':<8} | " # Format params with B
                f"{data['model_size_gb']:<10.2f} | " # Add model size
                f"{data['tokens_per_sec']:<12.2f} | {data['peak_vram_gen_gb']:<15.2f} | "
                f"{data['load_time_s']:<14.2f} | {data['cutoff_date']:<16}"
            )
            output_lines.append(line)

    output_lines.append(separator)

    # Print to console
    for line in output_lines:
        print(line)

    # Save results to file
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + "\n")
        print(f"\nResults also saved to: {OUTPUT_FILENAME}")
    except IOError as e:
        print(f"\nError saving results to file '{OUTPUT_FILENAME}': {e}", file=sys.stderr)

if __name__ == "__main__":
    run_benchmark()