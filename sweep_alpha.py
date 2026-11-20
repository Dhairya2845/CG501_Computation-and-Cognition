import csv
import matplotlib.pyplot as plt
import numpy as np
import run_experiments as r

ALPHAS = [0.02, 0.05, 0.10]
rows = [["alpha", "delay", "network", "control", "lesion"]]

plt.figure(figsize=(7, 4.5))
for a in ALPHAS:
    x, c, l = r.experiment_delay(alpha=a)
    gap = c - l
    plt.errorbar(x, gap.mean(0), yerr=[r.sem(gap[:, j]) for j in range(gap.shape[1])],
                 marker="o", capsize=3, label=f"alpha = {a}")
    rows += [[a, d, n, c[n, j], l[n, j]] for j, d in enumerate(x) for n in range(len(c))]
    print("alpha", a, "control", np.round(c.mean(0), 3), "lesion", np.round(l.mean(0), 3), flush=True)

plt.axhline(0, color="gray", linewidth=0.8)
plt.xlabel("Delay (interfering stimuli)")
plt.ylabel("Control - lesion recognition score")
plt.title("Sensitivity to interference learning rate")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(r.OUTPUT / "figure_alpha_sweep.png", dpi=180)

with open(r.OUTPUT / "results_alpha_sweep.csv", "w", newline="", encoding="utf-8") as h:
    csv.writer(h).writerows(rows)
print("Saved figure_alpha_sweep.png and results_alpha_sweep.csv")
