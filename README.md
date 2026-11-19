# Replication of Cowell, Bussey, and Saksida (2006)

This project implements the PMFC-style hierarchical Kohonen network and reproduces the three recognition-memory simulations in Cowell, Bussey, and Saksida (2006):

1. Delay-dependent recognition impairment
2. Recognition impairment across list lengths
3. Trial-unique versus repeated stimuli

## Files

- `model.py`: PMFC-style Kohonen network and recognition scoring
- `run_experiments.py`: runs all simulations and creates figures and `results.csv`
- `results.csv`: numerical simulation outputs
- `figure_*.png`: experiment figures
- `report.tex`: research report source
- `Cowell_PMFC_Project_Report.pdf`: compiled report

## Run

From this folder:

```powershell
python run_experiments.py
pdflatex -interaction=nonstopmode report.tex
```

## Requirements

The implementation requires Python 3, NumPy, and Matplotlib. PDF compilation requires a LaTeX installation with `pdflatex`.
