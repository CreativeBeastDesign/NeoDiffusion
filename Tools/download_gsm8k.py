#!/usr/bin/env python3
import json
import urllib.request
import re
import os
import ssl

def main():
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    print(f"Downloading GSM8K test set from {url}...")
    try:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response:
            lines = response.read().decode('utf-8').splitlines()
    except Exception as e:
        print(f"Failed to download: {e}")
        exit(1)

    prompts = []
    for i, line in enumerate(lines[:100]):
        data = json.loads(line)
        question = data["question"]
        answer_text = data["answer"]
        # Extract ground truth number after ####
        match = re.search(r"####\s*(-?\d+)", answer_text)
        answer = match.group(1) if match else ""
        prompts.append({
            "id": f"gsm8k-{i:03d}",
            "user": question,
            "answer": answer
        })

    os.makedirs("Tools/diffusion-bench/PromptSuites", exist_ok=True)
    output_path = "Tools/diffusion-bench/PromptSuites/gsm8k_100.json"
    with open(output_path, "w") as f:
        json.dump({"name": "gsm8k_100", "prompts": prompts}, f, indent=2)

    print(f"Successfully wrote 100 prompts to {output_path}")

if __name__ == "__main__":
    main()
