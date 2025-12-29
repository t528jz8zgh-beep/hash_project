# hash_project

Unified Python benchmark for open addressing without reordering:
Uniform / Funnel / Elastic hashing.

## Requirements
- Python 3.10+ (recommended)
- No extra deps (or: numpy, pandas, matplotlib if used)

## Run
### Alpha sweep (fixed n, vary alpha = 1-delta)
python3 bench_csv.py --algos uniform elastic funnel --n_list 50000 --delta_list 0.2 0.1 0.05 0.02 --trials 3 --q 3000 --out_csv alpha_sweep.csv

### N sweep (fixed alpha, vary n)
python3 bench_csv.py --algos uniform elastic funnel --n_list 16384 65536 262144 1048576 --delta_list 0.05 --trials 3 --q 800 --out_csv n_sweep_pow2.csv

## Outputs
- CSV: `alpha_sweep.csv`, `n_sweep_pow2.csv`
- Figures/Tables: `analysis_out/` 
