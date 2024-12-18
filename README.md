# Exclusion-simulation

This is the source code to simulate the spreading of a conjugative plasmid in a 2D static population of bacteria.

Two versions are available :

- Single plasmid : simulates the spreading of a single plasmid depending on the exclusion index (EI)
- Two plasmids : simulates the spreading of two competing plasmids depending on 4 separate EIs : $EI_{1}^{self}$, $EI_{1}^{comp}$, $EI_{2}^{self}$, $EI_{2}^{comp}$. These 4 variables define whether each plasmid excludes itself or competing plasmids when carried by a recipient cell.


## Dependencies
```
python==3.9.19
numpy==1.26.4
matplotlib==3.8.4
numba==0.59.1
```

## Prepare environment
```shell
conda new -n Exc_sim python==3.9
conda activate Exc_sim
pip install numpy matplotlib numba
```

## Run
```
python Exc_sim_one_plasmid.py [-h] [-m MODE] [-ei EXCLUSION_INDEX] [-r REPEAT] [-d DEAD_CUTOFF]
python Exc_sim_two_plasmid.py [-h] [-m MODE] [-ei EXCLUSION_INDICES] [-r REPEAT] [-d DEAD_CUTOFF]
```