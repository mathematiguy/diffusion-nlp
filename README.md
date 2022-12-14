# diffusion-nlp

This project attempts to reproduce the paper "Diffusion-LM Improves Controllable Text Generation" by Li, X. L., Thickstun, J., Gulrajani, I., Liang, P., & Hashimoto, T. B. (2022).

## Directory structure

There are 3 significant subfolders of this repository:

- `diffusion_lm` - contains code towards a from scratch reproduction of the authors' work. It includes a `model.py` model definition file in PyTorch, which implements the forward pass of the model as closely as I could figure out from the paper and also by looking through their source code. It is supported by `notebooks`, which contains my investigations of the model design, and also `tests` where I implemented some tests for testing the model code.
- `Diffusion-LM` - contains a fork of the original source code for the paper at github.com/XiangLi1999/Diffusion-LM. There I have containerized the project so it can be run reliably on other computers. The full details of the fork are documented there.
- `MLRC-2022-Report` - is a latex project containing a report written by myself for the completion of a Class Project for Comp-599 Natural Language Understanding at McGill University, fall 2022 semester.

# How to get started

The only software dependencies for this repository is GNU Make and Singularity. On Ubuntu systems, make can be installed simply via `sudo apt install make`. Instructions for how to install Singularity are available here: https://docs.sylabs.io/guides/3.5/user-guide/quick_start.html

If you are interested in running `diffusion_lm`, then you will need to build the singularity container in this directory.

```
# Build the singularity container for this project
make container
```

Then once you have done that, you can start a local Jupyterlab server via:

```
# Start local jupyterlab server
make jupyter
```

The server will be listening at `localhost:8888` and has a default password of "jupyter".

You can also run other `make` commands, such as:

```
# Build the latex report at MLRC-2022-Report/article.pdf
make report

# Run pytest unit tests
make test

# Attempt to train the diffusion_lm model (not working)
make train
```

This is everything you would need to know to get around this repository. Building the singularity container does take time, so if you insist on not using it you can still install the python requirements for the project with `pip install -r requirements.txt`, although it is recommended to do this inside of a python environment of some sort.

You can still run the make commands outside of the singularity container with `make <command> RUN=` - this suppresses the `singularity exec` command.
