This repository provides the script to reproduce the examples from the Bioptim paper (https://www.biorxiv.org/content/10.1101/2021.02.27.432868v1)

# Dependencies

The following instruction were tested on Linux Ubuntu 20.04. While bioptim is now Windows compatible, it was not at the time of the paper. The bioptim API slightly changed since the release of the paper. It is therefore expected to fail if one wants to run the example with the latest version of bioptim. 

In order to make these examples work you must have the following libraries=versions installed:
ipopt=3.13.3
casadi=3.5.5
eigen=3.3.9
biorbd=1.4.3
bioviz=2.0.1
bioptim=1.1.0
cmake=3.19.7  # Only required to build ACADOS

The easiest way to install the right versions in one go is to run the following command, assuming a conda environment is loaded:

```bash
conda install ipopt=3.13.3 casadi=3.5.5 biorbd=1.4.3 bioviz=2.0.1 bioptim=1.1.0 cmake=3.19.7 -cconda-forge
```

Moreover, to run the ACADOS examples, one must compile ACADOS=0.1.4. To automatically download and install it (again assuming the conda environment is loaded) the script `external/acados_install.sh` can be run. 

Please note that the Ipopt solver used is `ma57` that must be manually downloaded and installed from the hsl website [https://www.hsl.rl.ac.uk/](https://www.hsl.rl.ac.uk/). The default solver (`mumps`) will be a bit slower and will solve in more iterations.

# Running the examples

Once everything is properly installed, to generate the table of the paper, you can run the `table_generation.py` script. You can also explore each individual optimal control problems by having a look a `main.py` in each folder. 

Please note that the `generate_table.py` files use `ma57`, while `main.py` files use `mumps`, that is so one which does not have `ma57` still can run the examples.

Thanks for using Bioptim!
