## The Cerebrum library and engine

![Logo](/v1.1/logo.png)

The **Cerebrum library** can be used to easily train a **first** "[NNUE](https://www.chessprogramming.org/NNUE)-like" neural network for a chess engine. It was originally designed and built for the [Orion UCI chess engine](https://www.orionchess.com/).

Its originality lies in using only game results, parsed from pgn files provided by the user, and material values, computed on the fly, as targets for prediction. Both predicted values (game result and material) can be used for board evaluation.

Inference code is provided for embedding and using the trained network in a C/C++ or Python project, in two alternatives: standard (for accuracy) or quantized (for speed).

Do not hesitate to adapt the library to your own needs, and/or to use newer/better NNUE libraries for more flexibility/performance (e.g. [Bullet](https://github.com/jw1912/bullet/tree/main))!

<br/>

## Changes in 1.1

- **Lower memory requirements** for data preparation
- **Better training convergence** with improved feedback
- **Configurable network architecture** with 0, 1 or 2 hidden layers
- **New default (simpler) architecture**: `2x(768→128)→32→2` (1 hidden layer)

<br/>

## Changes in 1.0

- Training now relies on game results (from which a win ratio is deduced for each position during a game) and material only !
- Data preparation scripts are provided to automate the preparation of training data (using one or several pgn files)
- Network quantization is performed at the end of each training epoch, allowing the choice between better accuracy or increased inference speed
- A basic UCI chess engine is provided in two versions (standard or quantized) to demonstrate how to load and use the network
- Inference C code is now also available in two versions (standard or quantized)

<br/>

## Content and prerequisites (Windows)

The library consists of four main parts:

1. Data preparation code (Python scripts)
2. Training code (Python script)
3. Inference code (C files)
4. A basic UCI chess engine for demonstration purposes (Python script)

<br/>

To use the library, you will first need to:

- Download the `v1.1` folder of this repository
- Install a Python runtime: https://www.python.org/
- Install some Python librairies: `pip install torch tqdm chess`
- Download the [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) tool and put the `pgn-extract.exe` file in the folder `./1. data preparation/`

<br/>

Optionally (for better results):

- Download the [3, 4, 5 pieces](http://tablebase.sesse.net/syzygy/3-4-5/) endgame Syzygy tablebases and put them in the folder `./1. data preparation/syzygy/3-4-5/`
- Download the [6 pieces](http://tablebase.sesse.net/syzygy/6-WDL/) endgame Syzygy tablebases and put them in the folder `./1. data preparation/syzygy/6-pieces/`

Optionally (for faster training, if you have an NVIDIA GPU):

- `pip install torch --index-url https://download.pytorch.org/whl/cu124`

<br/>

## Usage (Windows)

### Data preparation

Prepare one or several pgn files containing full games and put it/them in the folder `./1. data preparation/pgn/`.

Then launch the script `prepare.bat` in the folder `./1. data preparation/` to obtain a file named `positions-shuffled.txt` which will be stored in the same folder.

This script will parse games and compute the average win ratio for each encountered position in all the games. It will also add some other statistical information (popcount, number of occurences of each position).

Copy the `positions-shuffled.txt` file to the folder `./2. training/positions/`.

<br/>

### Network architecture

You can configure the network architecture by modifying the script `train.py` in the folder `./2. training/`.

Supported architectures are:
- `2x(768→A)→2` (no hidden layer)
- `2x(768→A)→B→2` (one hidden layer)
- `2x(768→A)→B→C→2` (two hidden layers)

_(where A, B and C are mutliples of 32, e.g. `2x(768→128)→32→2` for `A=128` and `B=32`)_

<br/>

### Training

Launch the script `train.bat` in the folder `./2. training/`.

This script will parse the `positions-shuffled.txt` file in the folder `./2. training/positions/`, split it in batches of training data (this step can be long), and then use these data to train the neural network.

<br/>

### Output

Trained networks will be located in the folder `./2. training/networks/`. One network will be saved at the end of each training epoch.

By default:

- `epoch-11.txt` will be the last standard network (i.e. full precision: weights and biases are stored as `float`)
- `epoch-11-q.txt` will be the last quantized network (i.e. less precision, but high inference speed: weights and biases are stored as `int8`)

<br/>

### How to use trained networks

These networks can now be used in your own engine, using your own code, or:

- using the provided inference C code in `./3. inference/1. standard/` or `./3. inference/2. quantized/` folders
- using the provided inference Python code located in the `./4. engine/1. standard/` or `./4. engine/2. quantized/` folders

<br/>

In order to use your own trained network with the provided Cerebrum UCI chess engine:

- Copy the `epoch-11.txt` (resp. the `epoch-11-q.txt`) file in the folder `4. engine/1. standard/` (resp. `4. engine/2. quantized/`)
- Rename it to `network.txt`
- Launch the engine

<br/>

## How to configure name and author

You can adjust the name and author of the trained networks:

- Before training, by modifying the `NN_NAME` (default = "Cerebrum 1.1") and `NN_AUTHOR` (default = "David Carteau") variables in the script `train.py` located in the folder `./2. training/`
- After training, by modifying the first two lines of the generated networks (default = "name=Cerebrum 1.1" and "author=David Carteau")

<br/>

## Contribute

If you want to help me improve the library, do not hesitate to contact me via the [talkchess.com](https://www.talkchess.com) forum !

<br/>

## Copyright, license

Copyright 2025 by David Carteau. All rights reserved.

The Cerebrum library is licensed under the **MIT License** (see "LICENSE" and "/v1.1/license.txt" files).
