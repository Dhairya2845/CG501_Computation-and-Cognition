import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from model import PMFCNetwork


OUTPUT = Path(__file__).resolve().parent


def make_pair(rng, separated=True):
    sample = tuple(int(x) for x in rng.choice(16, size=4, replace=False))
    if separated:
        remaining = [x for x in range(16) if x not in sample]
        novel = tuple(int(x) for x in rng.choice(remaining, size=4, replace=False))
    else:
        novel = tuple(int(x) for x in rng.choice(16, size=4, replace=False))
    return sample, novel


def initialize_network(seed, lesioned):
    network = PMFCNetwork(seed=seed, lesioned=lesioned, size=20)
    network.pretrain(cycles=100)
    return network


def experiment_delay(network_seeds=range(6), trials_per_condition=4):
    rng = np.random.default_rng(100)
    results = {False: [], True: []}
    conditions = [0, 200, 400, 600, 800]
    pairs = [make_pair(rng) for _ in range(trials_per_condition)]
    for lesioned in (False, True):
        for delay in conditions:
            scores = []
            for seed in network_seeds:
                network = initialize_network(seed + (1000 if lesioned else 0), lesioned)
                for sample, novel in pairs:
                    network.encode_object(sample)
                    network.interfere(delay)
                    scores.append(network.recognition_score(sample, novel))
            results[lesioned].append(float(np.mean(scores)))
    return conditions, results


def experiment_list_length(network_seeds=range(6), sets_per_length=4):
    rng = np.random.default_rng(200)
    results = {False: [], True: []}
    lengths = [1, 6, 12, 18]
    for lesioned in (False, True):
        for length in lengths:
            scores = []
            for seed in network_seeds:
                network = initialize_network(seed + (2000 if lesioned else 0), lesioned)
                for _ in range(sets_per_length):
                    pairs = [make_pair(rng) for _ in range(length)]
                    for sample, _ in pairs:
                        network.encode_object(sample)
                    for sample, novel in pairs:
                        scores.append(network.recognition_score(sample, novel))
            results[lesioned].append(float(np.mean(scores)))
    return lengths, results


def experiment_trial_unique(network_seeds=range(6), pair_count=30):
    rng = np.random.default_rng(300)
    results = {}
    unique_pairs = [make_pair(rng) for _ in range(pair_count)]
    repeated_pair = make_pair(rng)
    repeated_pairs = [
        repeated_pair if i % 2 == 0 else (repeated_pair[1], repeated_pair[0])
        for i in range(pair_count)
    ]
    for condition, pairs in (
        ("trial-unique", unique_pairs),
        ("repeated", repeated_pairs),
    ):
        results[condition] = {}
        for lesioned in (False, True):
            scores = []
            for seed in network_seeds:
                network = initialize_network(seed + (3000 if lesioned else 0), lesioned)
                for sample, novel in pairs:
                    network.encode_object(sample)
                    network.interfere(200)
                    scores.append(network.recognition_score(sample, novel))
            results[condition][lesioned] = float(np.mean(scores))
    return results


def save_plot(x, control, lesion, xlabel, title, filename):
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, control, marker="o", linewidth=2, label="Control")
    plt.plot(x, lesion, marker="o", linestyle="--", linewidth=2, label="Lesion")
    plt.xlabel(xlabel)
    plt.ylabel("Recognition score")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=180)
    plt.close()


def main():
    conditions, delay = experiment_delay()
    lengths, list_results = experiment_list_length()
    trial_results = experiment_trial_unique()

    save_plot(
        conditions,
        delay[False],
        delay[True],
        "Delay (interfering stimuli)",
        "Delay-dependent recognition",
        "figure_delay.png",
    )
    save_plot(
        lengths,
        list_results[False],
        list_results[True],
        "List length (number of pairs)",
        "Effect of list length",
        "figure_list_length.png",
    )

    labels = ["Trial-unique", "Repeated"]
    control = [
        trial_results["trial-unique"][False],
        trial_results["repeated"][False],
    ]
    lesion = [
        trial_results["trial-unique"][True],
        trial_results["repeated"][True],
    ]
    x = np.arange(len(labels))
    plt.figure(figsize=(7, 4.5))
    width = 0.35
    plt.bar(x - width / 2, control, width, label="Control")
    plt.bar(x + width / 2, lesion, width, label="Lesion")
    plt.xticks(x, labels)
    plt.ylabel("Recognition score")
    plt.title("Trial-unique versus repeated stimuli")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "figure_trial_unique.png", dpi=180)
    plt.close()

    with open(OUTPUT / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "condition", "control", "lesion"])
        for i, condition in enumerate(conditions):
            writer.writerow(
                ["delay", condition, delay[False][i], delay[True][i]]
            )
        for i, length in enumerate(lengths):
            writer.writerow(
                ["list_length", length, list_results[False][i], list_results[True][i]]
            )
        for label in ("trial-unique", "repeated"):
            writer.writerow(
                [
                    "stimulus_type",
                    label,
                    trial_results[label][False],
                    trial_results[label][True],
                ]
            )

    print("Saved figures and results.csv")


if __name__ == "__main__":
    main()
