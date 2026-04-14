import json
from collections import defaultdict


def calculate_accuracy_by_duration(jsonl_file):
    results = defaultdict(lambda: {"correct": 0, "total": 0})
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            score = data["videomme_perception_score"]

            duration = score["duration"]
            pred = score["pred_answer"]
            answer = score["answer"]

            results[duration]["total"] += 1
            if pred == answer:
                results[duration]["correct"] += 1

    print("=" * 50)
    print("Accuracy by Duration")
    print("=" * 50)

    accuracies = []
    for duration in ["short", "medium", "long"]:
        if results[duration]["total"] > 0:
            acc = results[duration]["correct"] / results[duration]["total"] * 100
            accuracies.append(acc)
            print(
                f"{duration:8s}: {acc:6.1f}% ({results[duration]['correct']}/{results[duration]['total']})"
            )

    avg_acc = sum(accuracies) / len(accuracies)
    print("-" * 50)
    print(f"{'Average':8s}: {avg_acc:6.1f}%")


if __name__ == "__main__":
    calculate_accuracy_by_duration(
        "./results/Tempo-6B-4K/checkpoints__Tempo-6B/***_samples_videomme.jsonl"
    )
