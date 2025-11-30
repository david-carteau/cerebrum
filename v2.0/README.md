## The Cerebrum library

![Logo](/v2.0/logo.png)

The **Cerebrum library** can be used to easily train a "[NNUE](https://www.chessprogramming.org/NNUE)-like" neural network for a chess engine. It was originally designed and built for the [Orion UCI chess engine](https://www.orionchess.com/).

It is composed of a few Python scripts for data preparation (optional), one Python script for **training**, and C code for **inference**.

Default network architecture is perspective-based with one hidden layer. Network weights are quantised to maximise inference speed.

Code is also provided to train a **first** network using only game results, parsed from PGN files provided by the user, and material values, computed on the fly (optional).

Feel free to adapt the library to your own needs and/or use newer/better NNUE libraries for greater flexibility and performance (e.g. [Bullet](https://github.com/jw1912/bullet/tree/main))!

<br/>

## Changes in 2.0

- **Change in network outputs**: networks now directly predict scores in centipawns → _breaking change!_
- **Tiny change to the data format** for data preparation → _breaking change!_
- **Lower disk usage** for data preparation
- **New default architecture**: `2x(768→256)→32→1` (1 hidden layer)

<br/>

## Changes in 1.1

- **Lower memory requirements** for data preparation
- **Better training convergence** with improved feedback
- **Configurable network architecture** with 0, 1 or 2 hidden layers
- **New default (simpler) architecture**: `2x(768→128)→32→2` (1 hidden layer)

<br/>

## Changes in 1.0

- Training now relies on game results (from which a win ratio is deduced for each position during a game) and material only!
- Data preparation scripts are provided to automate the preparation of training data (using one or several pgn files)
- Network quantization is performed at the end of each training epoch, allowing the choice between better accuracy or increased inference speed
- A basic UCI chess engine is provided in two versions (standard or quantized) to demonstrate how to load and use the network
- Inference C code is now also available in two versions (standard or quantized)

<br/>

## Content and prerequisites (Windows)

To use the library, you will first need to:

- Download the `v2.0` folder of this repository
- Install a Python runtime: https://www.python.org/
- Install some Python librairies: `pip install tqdm chess`
- Install PyTorch librairy: `pip install torch` or, if you have an NVIDIA GPU, `pip install torch --index-url https://download.pytorch.org/whl/cu128`

<br/>

Optionally, if you want to train a **first** network from PGN files:

- Download the [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) tool and put the `pgn-extract.exe` file in the folder `./1. data preparation (optional)/`

<br/>

## Usage (Windows)

### Data preparation (standard)

Prepare a file containing positions and evaluations. Each line of the file must contain a fenstring followed by its evaluation (in pawns), separated with a comma, e.g.:

Example:
- _r5k1/5pp1/pR2p3/1p1rP3/7P/R3P3/P6P/6K1 w - -,-4.5000_
- _6k1/ppp5/8/3P1p2/PP1b4/5pPp/5P1K/8 b - -,6.5000_
- _3r3k/p4pp1/4p2p/2pRq3/8/PP2P2P/2Q2PP1/2R3K1 b - -,-2.5000_
- _1r4k1/q2pbp1p/4n1p1/p1pQP2P/Rr1nB1N1/4B1P1/5PK1/2R5 w - -,3.5000_

Copy the `positions-shuffled.txt` file to the folder `./2. training/positions/`.

<br/>

### Data preparation (alternative)

Prepare one or several pgn files containing full games and put it/them in the folder `./1. data preparation (optional)/pgn/`.

Then launch the script `prepare.bat` in the folder `./1. data preparation (optional)/` to obtain a file named `positions-shuffled.txt`, which will be stored in the same folder.

Copy the `positions-shuffled.txt` file to the folder `./2. training/positions/`.

<br/>

### Network architecture

You can configure the network architecture by modifying the script `train.py` in the folder `./2. training/`.

Supported architectures are:
- `2x(768→A)→1` (no hidden layer)
- `2x(768→A)→B→1` (one hidden layer)
- `2x(768→A)→B→C→1` (two hidden layers)

_(where A, B and C are mutliples of 32, e.g. `2x(768→128)→32→2` for `A=128` and `B=32`)_

<br/>

### Training

Launch the script `train.bat` in the folder `./2. training/`.

This script will parse the `positions-shuffled.txt` file in the folder `./2. training/positions/`, split it in batches of training data (this step can be long), and then use these data to train the neural network.

<br/>

### Output

Trained networks will be located in the folder `./2. training/networks/`. One network will be saved at the end of each training epoch.

By default, `epoch-11-q.txt` will be the last quantized network.

<br/>

### How to use trained networks

Trained networks can now be used in your own engine, using your own code, or using the provided inference C code, provided in the `./3. inference/` folder.

<br/>

## How to configure name and author

You can adjust the name and author of the trained networks:

- Before training, by modifying the `NN_NAME` (default = "Cerebrum 2.0") and `NN_AUTHOR` (default = "David Carteau") variables in the script `train.py` located in the folder `./2. training/`
- After training, by modifying the first two lines of the generated networks (default = "name=Cerebrum 2.0" and "author=David Carteau")

<br/>

You can adjust more parameters: open and inspect the provided Python scripts!

<br/>

## Contribute

If you want to help me improve the library, do not hesitate to contact me via the [talkchess.com](https://www.talkchess.com) forum!

<br/>

## Copyright, license

Copyright 2025 by David Carteau. All rights reserved.

The Cerebrum library is licensed under the **MIT License** (see "LICENSE" and "/v2.0/license.txt" files).
