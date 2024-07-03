# TraT-sim

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

## Run
```
python TraT_sim_one_donor.py [-h] [-m MODE] [-ei EXCLUSION_INDEX] [-r REPEAT] [-d DEAD_CUTOFF]

Simulates the spreading of a single plasmid depending on the exclusion index (EI)

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  'm' (movie) or 'b' (batch). Default = 'm'
  -ei EXCLUSION_INDEX, --exclusion-index EXCLUSION_INDEX
                        Exclusion index. Default = 1 (no exclusion)
  -r REPEAT, --repeat REPEAT
                        Number of repeats in batch mode. Default = 50
  -d DEAD_CUTOFF, --dead-cutoff DEAD_CUTOFF
                        Dead cutoff. Minimum number of simultaneous mating into the same recipient which would result in the death of the recipient (lethal zygosis). Default = 5
```
