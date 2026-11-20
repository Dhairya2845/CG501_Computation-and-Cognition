import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from model import PMFCNetwork
from scipy import stats
N_NETS, SIZE = 6, 200              # set SIZE = 40 for a quick smoke test only
DELAYS = [0, 200, 400, 600, 800]
LENGTHS = [1, 6, 12, 18]
KS = [0, 1, 2, 3]
EXT_DELAYS = [0, 400, 800]

OUTPUT = Path(__file__).resolve().parent


def make_pair(rng, separated=True):
    sample = tuple(int(x) for x in rng.choice(16, size=4, replace=False))
    if separated:
        remaining = [x for x in range(16) if x not in sample]
        novel = tuple(int(x) for x in rng.choice(remaining, size=4, replace=False))
    else:
        novel = tuple(int(x) for x in rng.choice(16, size=4, replace=False))
    return sample, novel


def make_pair_shared(rng, k):
    """Sample/novel share k features in the same positions; the rest are disjoint."""
    sample = rng.choice(16, 4, replace=False)
    keep = rng.choice(4, k, replace=False)
    novel = sample.copy()
    pool = [f for f in range(16) if f not in sample]
    others = [i for i in range(4) if i not in keep]
    novel[others] = rng.choice(pool, len(others), replace=False)
    return tuple(int(x) for x in sample), tuple(int(x) for x in novel)


def initialize_network(seed, size=SIZE):
    net = PMFCNetwork(seed=seed, size=size)
    net.pretrain(cycles=100)
    return net, net.snapshot()

def experiment_delay(n_nets=N_NETS, trials=12, alpha=0.05):
    ctl = np.zeros((n_nets, len(DELAYS)))
    les = np.zeros((n_nets, len(DELAYS)))
    for n in range(n_nets):
        rng = np.random.default_rng(100 + n)                  # different pairs per network
        pairs = [make_pair(rng) for _ in range(trials)]
        net, snap = initialize_network(n)
        c = [[] for _ in DELAYS]
        l = [[] for _ in DELAYS]
        for sample, novel in pairs:
            net.restore(snap)                                 # reset every trial
            net.encode_object(sample)
            done = 0
            for j, d in enumerate(DELAYS):                    # test has no learning, so checkpoints are valid
                net.interfere(d - done, alpha)
                done = d
                r_les, r_ctl = net.scores(sample, novel)
                c[j].append(r_ctl)
                l[j].append(r_les)
        ctl[n] = [np.mean(x) for x in c]
        les[n] = [np.mean(x) for x in l]
    return DELAYS, ctl, les


def experiment_list_length(n_nets=N_NETS, sets=4):
    rng = np.random.default_rng(200)
    lists = {L: [[make_pair(rng) for _ in range(L)] for _ in range(sets)] for L in LENGTHS}
    ctl = np.zeros((n_nets, len(LENGTHS)))
    les = np.zeros((n_nets, len(LENGTHS)))
    for n in range(n_nets):
        net, snap = initialize_network(n)
        for j, L in enumerate(LENGTHS):
            c, l = [], []
            for pairs in lists[L]:
                net.restore(snap)                             # reset every list
                for sample, _ in pairs:
                    net.encode_object(sample)
                for sample, novel in pairs:
                    r_les, r_ctl = net.scores(sample, novel)
                    c.append(r_ctl)
                    l.append(r_les)
            ctl[n, j], les[n, j] = np.mean(c), np.mean(l)
    return LENGTHS, ctl, les


def experiment_trial_unique(n_nets=N_NETS, pair_count=30):
    rng = np.random.default_rng(300)
    unique = [make_pair(rng) for _ in range(pair_count)]
    base = make_pair(rng)
    repeated = [base if rng.random() < 0.5 else (base[1], base[0]) for _ in range(pair_count)]
    out = {}
    for name, pairs in (("trial-unique", unique), ("repeated", repeated)):
        ctl, les = np.zeros(n_nets), np.zeros(n_nets)
        for n in range(n_nets):
            net, _ = initialize_network(n)                    # NO restore between trials
            c, l = [], []
            for sample, novel in pairs:
                net.encode_object(sample)
                net.interfere(200)
                r_les, r_ctl = net.scores(sample, novel)
                c.append(r_ctl)
                l.append(r_les)
            ctl[n], les[n] = np.mean(c), np.mean(l)
        out[name] = (ctl, les)
    return out


def experiment_similarity(n_nets=10, pairs_per_cell=10):
    rng = np.random.default_rng(400)
    pairs = {k: [make_pair_shared(rng, k) for _ in range(pairs_per_cell)] for k in KS}
    ctl = np.zeros((n_nets, len(KS), len(EXT_DELAYS)))
    les = np.zeros_like(ctl)
    for n in range(n_nets):
        net, snap = initialize_network(n)
        for a, k in enumerate(KS):
            c = [[] for _ in EXT_DELAYS]
            l = [[] for _ in EXT_DELAYS]
            for sample, novel in pairs[k]:
                net.restore(snap)
                net.encode_object(sample)
                done = 0
                for b, d in enumerate(EXT_DELAYS):
                    net.interfere(d - done)
                    done = d
                    r_les, r_ctl = net.scores(sample, novel)
                    c[b].append(r_ctl)
                    l[b].append(r_les)
            ctl[n, a] = [np.mean(x) for x in c]
            les[n, a] = [np.mean(x) for x in l]
    return ctl, les


# ---------- statistics and plotting ----------
def sem(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))


def paired(ctl, les):
    """Paired t-test across networks plus Cohen's d_z for control - lesion."""
    t, p = stats.ttest_rel(ctl, les)
    diff = ctl - les
    return t, p, diff.mean() / diff.std(ddof=1)


def save_plot(x, ctl, les, xlabel, title, filename):
    plt.figure(figsize=(7, 4.5))
    for data, style, label in ((ctl, "-", "Control"), (les, "--", "Lesion")):
        plt.errorbar(x, data.mean(0), yerr=[sem(data[:, j]) for j in range(data.shape[1])],
                     marker="o", linestyle=style, linewidth=2, capsize=3, label=label)
    plt.xlabel(xlabel)
    plt.ylabel("Recognition score")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=180)
    plt.close()


def main():
    rows = [["experiment", "condition", "network", "control", "lesion"]]

    x, c, l = experiment_delay()
    save_plot(x, c, l, "Delay (interfering stimuli)", "Delay-dependent recognition", "figure_delay.png")
    for j, d in enumerate(x):
        print("delay", d, "t=%.2f p=%.4f d=%.2f" % paired(c[:, j], l[:, j]))
        rows += [["delay", d, n, c[n, j], l[n, j]] for n in range(len(c))]

    x, c, l = experiment_list_length()
    save_plot(x, c, l, "List length (number of pairs)", "Effect of list length", "figure_list_length.png")
    for j, L in enumerate(x):
        print("list", L, "t=%.2f p=%.4f d=%.2f" % paired(c[:, j], l[:, j]))
        rows += [["list_length", L, n, c[n, j], l[n, j]] for n in range(len(c))]

    tu = experiment_trial_unique()
    labels = list(tu)
    xs = np.arange(len(labels))
    plt.figure(figsize=(7, 4.5))
    for off, idx, name in ((-0.175, 0, "Control"), (0.175, 1, "Lesion")):
        plt.bar(xs + off, [tu[k][idx].mean() for k in labels], 0.35,
                yerr=[sem(tu[k][idx]) for k in labels], capsize=3, label=name)
    plt.xticks(xs, labels)
    plt.ylabel("Recognition score")
    plt.title("Trial-unique versus repeated stimuli")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "figure_trial_unique.png", dpi=180)
    plt.close()
    for k in labels:
        print(k, "t=%.2f p=%.4f d=%.2f" % paired(*tu[k]))
        rows += [["stimulus_type", k, n, tu[k][0][n], tu[k][1][n]] for n in range(len(tu[k][0]))]

    c, l = experiment_similarity()
    imp = c - l                                               # impairment = control - lesion
    plt.figure(figsize=(7, 4.5))
    for b, d in enumerate(EXT_DELAYS):
        plt.errorbar(KS, imp[:, :, b].mean(0), yerr=[sem(imp[:, a, b]) for a in range(len(KS))],
                     marker="o", capsize=3, label=f"delay {d}")
        slopes = [np.polyfit(KS, imp[n, :, b], 1)[0] for n in range(len(imp))]
        print("similarity slope, delay", d, "t=%.2f p=%.4f" % stats.ttest_1samp(slopes, 0))
        for a, k in enumerate(KS):
            rows += [["similarity_d%d" % d, k, n, c[n, a, b], l[n, a, b]] for n in range(len(c))]
    plt.xlabel("Shared features (k)")
    plt.ylabel("Impairment (control - lesion)")
    plt.title("Effect of sample-novel similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "figure_similarity.png", dpi=180)
    plt.close()

    with open(OUTPUT / "results.csv", "w", newline="", encoding="utf-8") as h:
        csv.writer(h).writerows(rows)
    print("Saved figures and results.csv")


if __name__ == "__main__":
    main()
