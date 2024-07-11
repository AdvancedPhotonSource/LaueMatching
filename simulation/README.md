# Simulation

This folder consists of two files to do simulation and then indexing using LaueMatching.

There are two steps:

## Generation of simulated image:

This involves using the orientations in `simulationOrientationMatrices.csv` file and experimental configuration in `params_sim.txt` file to generate the simulated image using the following command:

    ../GenerateSimulation.py -configFile params_sim.txt -orientationFile simulationOrientationMatrices.csv -outputFile simulated_1.h5

This will generate the following files:

`simulated_1.h5`: hdf5 file containing the simulated image as a `uint16_t` array.
`simulated_1.h5.tif`: TIFF image of the simulation.
`simulated_1.h5_simulated_recips.txt`: File containing the reciprocal space array of each orientation.

## Processing of simulated image to generate orientations:
