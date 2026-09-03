"""
build_dataset.py

Runs the full dataset-building pipeline in one command:
    1. extract_dialogue_dataset.py   (real Q&A pairs from Plato-style dialogues)
    2. generate_synthetic_dataset.py (LLM-generated pairs from essay-style texts)
    3. Combines both into dataset/combined.jsonl, ready to upload to
       finetune_colab.ipynb

This does NOT run the actual fine-tuning -- that step requires selecting
a GPU runtime and uploading a file through Colab's UI, which can't be
scripted. See the README for that step.

Usage:
    python3 build_dataset.py                  # run everything
    python3 build_dataset.py --skip-synthetic  # skip the Ollama-based step
                                                # (faster, no Ollama required,
                                                # but less training data)
"""

import os
import sys
import argparse
import subprocess

TEXTS_DIR = "texts"
DATASET_DIR = "dataset"
DIALOGUE_FILE = os.path.join(DATASET_DIR, "dialogue_pairs.jsonl")
SYNTHETIC_FILE = os.path.join(DATASET_DIR, "synthetic_pairs.jsonl")
COMBINED_FILE = os.path.join(DATASET_DIR, "combined.jsonl")


def run_step(description, script_name):
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}\n")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"\n[warning] {script_name} exited with an error (code {result.returncode}).")
        print("Continuing pipeline, but check the output above.")
    return result.returncode == 0


def combine_datasets():
    files_to_combine = [f for f in (DIALOGUE_FILE, SYNTHETIC_FILE) if os.path.exists(f)]

    if not files_to_combine:
        print("\nNo dataset files found to combine. Did the earlier steps run successfully?")
        return 0

    total_lines = 0
    with open(COMBINED_FILE, "w", encoding="utf-8") as out:
        for filepath in files_to_combine:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                out.write(content)
                if content and not content.endswith("\n"):
                    out.write("\n")
                total_lines += sum(1 for line in content.splitlines() if line.strip())

    return total_lines


def main():
    parser = argparse.ArgumentParser(description="Build the fine-tuning dataset in one step.")
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Skip the Ollama-based synthetic generation step (faster, no Ollama needed, less data)"
    )
    args = parser.parse_args()

    if not os.path.isdir(TEXTS_DIR) or not os.listdir(TEXTS_DIR):
        print(f"No texts found in {TEXTS_DIR}/. Run 'python3 download_texts.py' first.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)

    run_step("Step 1/2: Extracting real dialogue pairs", "extract_dialogue_dataset.py")

    if not args.skip_synthetic:
        print("\nStep 2/2 uses your local Ollama model -- make sure Ollama is")
        print("running (ollama serve) with a model pulled, or rerun this with")
        print("--skip-synthetic to skip it.")
        run_step("Step 2/2: Generating synthetic pairs from essay-style texts", "generate_synthetic_dataset.py")
    else:
        print("\nSkipping synthetic generation (--skip-synthetic was passed).")

    print(f"\n{'=' * 60}")
    print("  Combining datasets")
    print(f"{'=' * 60}\n")
    total = combine_datasets()

    if total:
        print(f"Done. Wrote {total} total training pairs to {COMBINED_FILE}")
        print(f"\nNext step: open finetune_colab.ipynb in Google Colab, upload")
        print(f"{COMBINED_FILE}, and run the notebook (see README for details).")
    else:
        print("Combine step produced no output -- check the errors above.")


if __name__ == "__main__":
    main()