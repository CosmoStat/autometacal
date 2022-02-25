# autometacal

[![CI](https://github.com/CosmoStat/autometacal/actions/workflows/main.yml/badge.svg)](https://github.com/CosmoStat/autometacal/actions/workflows/main.yml)

Metacalibration and shape measurement by automatic differentiation

Project led by [@andrevitorelli](https://github.com/andrevitorelli)


## Requirements

This project relies on the [GalFlow](https://github.com/DifferentiableUniverseInitiative/GalFlow) library as well as
[GalSim](https://github.com/GalSim-developers/GalSim). To install GalFlow:
```bash
$ pip install git+https://github.com/DifferentiableUniverseInitiative/GalFlow.git
```
And we are also assuming that TensorFlow & TensorFlow-addons is installed.

To use quintic interpolation in TensorFlow-addons, follow these additional interim install instructions:

 - Clone tensorflow addons from [andrevitorelli/addons](https://github.com/andrevitorelli/addons)
 - Switch to /new_kernels branch
 - Compile as instructed
 - clone [andrevitorelli/GalFlow](https://github.com/andrevitorelli/GalFlow)
 - Switch to u/andrevitorelli/interpolation testing branch
 - install it with ` pip install . ` 


