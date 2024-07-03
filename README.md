# TraT-sim

This is the source code to simulate the spreading of a conjugative plasmid in a 2D static population of bacteria.

Two versions are available :

- Single plasmid : simulates the spreading of a single plasmid depending on the exclusion index (EI)
- Two plasmids : simulates the spreading of two competing plasmids depending on 4 separate EIs : $EI_{1}^{self}$, $EI_{1}^{comp}$, $EI_{2}^{self}$, $EI_{2}^{comp}$. These 4 variables define whether each plasmid excludes itself or competing plasmids when carried by a recipient cell.


## Dependencies
```
numpy
matplotlib
```

## Run
```
python TraT_sim_one_donor.py [-h] [options]
	-m, --mode		[required] 'm' (movie) or 'b' (batch)
	-ei					Exclusion index. By default set to 1 (no exclusion).
	-r, --repeat		Number of repeats in batch mode. By default set to 100.
	-d					Dead cutoff. Minimum number of simultaneous mating into
						the same recipient which would result in the death of
						the recipient (lethal zygosis).
```
