# Plasmid exclusion simulation

## Description

This program simulates the spreading of a conjugative plasmid in a population of bacteria on plane. During conjugation, a conjugative plasmid mediates its own transfer from one bacterial cell ("donor") to another ("recipient"). However if the recipient already carries the plasmid, the transfer is disrupted, in a process known as "plasmid exclusion". This program aims to evaluate the effect of plasmid exclusion on the evolutionary fitness of conjugative plasmids.

## Versions

Two versions of the program are available :

In its simplest form (one plasmid), the simulation follows a number of rules:
- The population of bacteria is represented by a 2D array of integers, with each element in the array representing a single bacterial cell.
- At the beginning of the simulation, a random cell is turned into a donor cell, which can spread the plasmid into neighboring cells at each timestep of the simulation.
- When the target (recipient) cell already carries the plasmid, the success rate of the conjugation event is reduced by a factor called "exclusion index" (EI).
- Multiple donors can conjugate with the same recipient cell in the same timestep, and when the number of successful conjugations with the same recipient exceeds a threshold (d), that recipient cell dies and can no longer act as donor or recipient.

In the more advanced version, two distinct plasmids compete with each other. Each plasmid can exclude itself ($EI^{self}$) and/or the competing plasmid ($EI^{comp}$). The evolutionary fitness of each plasmid can be estimated by the percentage of cells that carry it at the end of the simulation.

## Features

- User-friendly GUI (✨new!✨)
- Customizable simulation parameters
- Automatic graph plotting
- Movie mode to track the evolution of simulations over time
- Data can to be saved for future analysis

## Installation

```shell
git clone https://github.com/nypchen/Exclusion-sim.git
cd Exclusion-sim
conda env create -f environment.yml
```

## Running

```shell
conda activate exc_sim
python one_plasmid_GUI.py
python two_plasmids_GUI.py
```