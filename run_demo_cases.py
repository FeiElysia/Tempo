import json
import subprocess

def main():
    model_path = "./checkpoints/Tempo-6B"
    cases_file = "./examples/demo_cases.json"
    
    with open(cases_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        
    print(f"🚀 Loaded {len(cases)} demo cases. Starting sequential inference...\n" + "="*50)
    
    for i, case in enumerate(cases, 1):
        print(f"\n▶️  [Case {i}/{len(cases)}] Processing Video: {case['video_path']}")
        print(f"❓  Query: {case['query'][:50]}... (truncated)")
        
        command = [
            "python", "infer.py",
            "--model_path", model_path,
            "--video_path", case["video_path"],
            "--query", case["query"]
        ]
        
        subprocess.run(command)
        print("-" * 50)

if __name__ == "__main__":
    main()