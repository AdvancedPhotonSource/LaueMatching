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

The orientations present in the `simulated_1.h5` file can be determined using the following command:

    ../RunImage.py -configFile params_sim.txt -imageFile simulated_1.h5 -nCPUs NCPUS -computeType COMPUTE

Here `NCPUS` should be the number of CPU cores you want to use and `COMPUTE` can either be `GPU` or `CPU`.

**NOTE:** Even when using a GPU, please use multiple CPU cores because refinement of orientations is still run on the CPU.

This will generate results in `results_simulation` folder. This parameter is set in the `params_sim.txt` file as `ResultDir`.

> If you want to reduce initialization times (from a few seconds per image to a few microseconds), your `OrientationFile` and `ForwardFile` should be in `/dev/shm`. You can either just provide the path to these files and `LaueMatching` will generate these, or copy already generated files to `/dev/shm`. Any files of `/dev/shm` can be directly memory mapped instead of being read, and this process is millions of times faster. ***Note:*** This will not work on MAC.