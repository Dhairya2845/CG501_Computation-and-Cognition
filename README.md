# Replication and extension of Cowell, Bussey and Saksida (2006)

NumPy implementation of the hierarchical Kohonen (self-organising map) model of object recognition memory in perirhinal cortex (PRh), from: Cowell, R. A., Bussey, T. J., & Saksida, L. M. (2006). Why does brain damage impair memory? A connectionist model of object recognition memory in perirhinal cortex. Journal of Neuroscience, 26(47), 12186-12197.

The model has four caudal feature grids and one PRh conjunction grid. The lesioned network is the control network's caudal layer alone. Familiarity is read out as the selectivity of a grid's activation map, and recognition is scored as R = (S_sample - S_novel) / (S_sample + S_novel).

## Experiments

1. Delay: 0, 200, 400, 600, 800 interfering stimuli
2. List length: 1, 6, 12, 18 pairs
3. Trial-unique vs repeated stimuli
4. Extension: sample-novel feature similarity (k = 0, 1, 2, 3 shared features,
   crossed with delays 0, 400, 800)
5. Sensitivity analysis: delay experiment at interference learning rates
   0.02, 0.05 (paper value) and 0.10

## Files

- `model.py`: Kohonen grid, PMFC network, readout and recognition scores
- `run_experiments.py`: runs experiments 1-4, saves figures and `results.csv`
- `sweep_alpha.py`: runs experiment 5, saves its figure and CSV
- `test_model.py`: unit tests
- `results.csv`: per-network results for experiments 1-4
- `results_alpha_sweep.csv`: per-network results for experiment 5
- `figure_delay.png`, `figure_list_length.png`, `figure_trial_unique.png`, `figure_similarity.png`, `figure_alpha_sweep.png`: figures used in the report

## Requirements

Python 3, NumPy, SciPy and Matplotlib:

    pip install -r requirements.txt

No external dataset is used. All stimuli are simulated.

## Run

From this folder:

    python -m unittest
    python run_experiments.py
    python sweep_alpha.py

These regenerate every figure and CSV listed above. Seeds are fixed in the code, so repeated runs give the same numbers. Full-size runs (200 x 200 grids) take about 1 hour; set `SIZE = 40` at the top of `run_experiments.py` for a quick smoke test (its numbers are for debugging only).

## Settings used for the reported results

- Grid size: 200 x 200 (torus), city-block neighbourhood distance
- Pretraining: 100 cycles, alpha(t) = t^-0.2, G(t) = 0.5 + 10 t^-0.4, then
  noise in [-1, 1] and rescaling (w + 1) / 3
- Encoding: 20 cycles, alpha = 0.35, G = 2.085; interference: 1 cycle each,
  alpha = 0.05, G = 2.085
- Experiments 1-3: [6] networks; [4] trials per network for delay; [4] lists per
  length; 30 pairs per condition for trial-unique vs repeated
- Extension: [10] networks, [10] pairs per cell
- Readout: RMS distance normalised by input dimension, activation ln(1/dist)
  capped at 9.210, peak = winner plus 8 neighbours (3 x 3 block)

## Implementation choices not given in the paper

Feature values (16 features on a 4 x 4 grid of levels), the RMS form of the distance, the neighbourhood cutoff at 3G, and averaging the two layers' recognition scores in the control network. See the report for details.

