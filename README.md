# TVAE_release
TVAE release version.

## Instructions for Reproducing Results:

### Required Libraries and Packages:

CUDA 10.2

torch 1.6.0

pyro 1.4.0

pandas 1.0.5

numpy 1.18.5

argparse 1.1

scipy 1.5.0

sklearn 0.23.1

os

pytz 2020.1

### TVAESynth:

```
python3 main_TVAEsynth.py --tl_weight 0.1 --latent_dim_o 1 --latent_dim_c 2 --latent_dim_t 2 --latent_dim_y 2 --hidden_dim 20 --num_layers 2 --num_epochs 40 --batch_size 200 --learning_rate 0.0005 --reps 100
```
### IHDP:
```
python3 main_IHDP.py --tl_weight 0.4 --latent_dim_o 5 --latent_dim_c 15 --latent_dim_t 10 --latent_dim_y 10 --hidden_dim 500 --num_layers 4 --num_epochs 200 --batch_size 200 --learning_rate 0.00005 --reps 100 --jobs_version 1
```

### JOBS:
```
python3 main_JOBS.py --tl_weight 0.1 --latent_dim_o 4 --latent_dim_c 8 --latent_dim_t 6 --latent_dim_y 6 --hidden_dim 200 --num_layers 2 --num_epochs 150 --batch_size 200 --learning_rate 0.00001 --reps 100
```





