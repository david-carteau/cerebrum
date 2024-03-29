## The Cerebrum library and engine

![Logo](/v1.0/logo.png)

The **Cerebrum library** can be used to train and utilize "[NNUE](https://www.chessprogramming.org/NNUE)-like" neural networks for chess engines. It was initially designed and created for the [Orion UCI chess engine](https://www.orionchess.com/).

<br/>

## Changes in 1.0

- **Training now relies on game results** (from which a win ratio is deduced for each position during a game) and **material only** !
- **Data preparation** scripts are provided to automate the preparation of training data (using one or several pgn files)
- **Network quantization** is performed at the end of each training epoch, allowing the choice between better accuracy or increased inference speed
- A **basic UCI chess engine** is provided in two versions (standard or quantized) to demonstrate how to load and use the network
- Inference C code is now also available in two versions (standard or quantized)

<br/>

## Content and prerequisites (Windows)

The library consists of four main parts:

1. **Data preparation code** (Python scripts)
2. **Training code** (Python script)
3. **Inference code** (C files)
4. A **basic UCI chess engine** for demonstration purposes (Python script)

<br/>

To use the library, you will first need to:

- Download the `v1.0` folder of this repository
- Install a Python runtime: https://www.python.org/
- Install some Python librairies: `pip install torch numpy scipy tqdm chess`
- Download the [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) tool and put the `pgn-extract.exe` file in the folder `./1. data preparation/`

<br/>

Note that, **by default, training is forced to use the CPU**. If you have an NVIDIA GPU, you can:

- Install the appropriate version of the torch library (see links below) to improve training speed
- Set `FORCE_CPU_DEVICE` to `False` in the script `train.py` (line 53) to enable GPU usage

<br/>

## Usage (Windows)

### Input (1<sup>st</sup> alternative)

You can choose to provide one or several pgn files containing full games in the folder `./1. data preparation/pgn/`.

Then launch the script `prepare.bat` in the folder `./1. data preparation/` to obtain a file named `positions-shuffled.txt` which will be stored in the same folder.

This script will parse games and compute the average win ratio for each encountered position in all the games. It will also add some other statistical information (popcount, number of occurences of each position).

Copy the `positions-shuffled.txt` file to the folder `./2. training/positions/`.

Note that, **by default, all games after 31.12.2023 will be ignored**. You can easily modify this by editing the script `prepare.bat` (see source to discover how).

<br/>

### Input (2<sup>nd</sup> alternative)

Prepare your own handcrafted file containing position (fenstring), side to move, popcount, number of occurences of this position, win ratio:

```
2r3k1/1q3p1p/6p1/3p4/1r3N2/2n5/2PQ1PPP/R2R2K1 w 18 1 1.0
1n1r1rk1/1pp3p1/p2bppp1/1q6/1P1P1P1P/P3PBP1/1Q3NK1/2R2R2 w 26 1 1.0
r7/pp2bNpr/2n3Rp/4pk1P/2n5/2N1B3/PP3P2/2KR4 w 21 1 0.5
rn2qr2/6b1/2pkBp1p/p1N4P/1p1P4/PQ3P2/1P6/2R1K2R b 22 1 0.5
6k1/3n1pp1/pb6/3Pp2p/4N3/4P1P1/2r2PKP/1R2B3 w 18 1 0.5
3k1n2/4q3/4pp2/1Np1p1p1/2P1P3/3P2P1/1PK2P2/7Q w 17 1 1.0
r5k1/pb3rb1/3Rn1p1/Rp2p2p/4PP1P/2P1BNP1/4NK2/8 b 22 1 0.5
8/3k4/7R/8/5PK1/8/r7/8 w 5 1 0.5
5rk1/5pbp/1qN1b1p1/1PQnP3/r3B3/3p1N2/p2P1PPP/2R2RK1 w 25 1 0.5
2b1k3/2p5/1pP5/4rp2/2B1pNpP/1P6/PKP5/3R4 w 17 1 1.0
```

<br/>

This alternative can be useful when you already have a good idea of the win ratio associated to a (large) bunch of positions.

The provided file must be called `positions-shuffled.txt` and be placed in the folder `./2. training/positions/`.

<br/>

### Training

Launch the script `train.bat` in the folder `./2. training/`.

This script will parse the `positions-shuffled.txt` file in the folder `./2. training/positions/`, split it in batches of training data (this step can be long), and then use these data to train the neural network.

<br/>

### Output

Trained networks will be located in the folder `./2. training/networks/`. One network will be saved at the end of each training epoch.

By default:

- `epoch-5.txt` will be the last standard network (i.e. full precision: weights and biases are stored as `float`)
- `epoch-5-q.txt` will be the last quantized network (i.e. less precision, but high inference speed: weights and biases are stored as `int8`)

<br/>

### How to use trained networks

These networks can now be used in your own engine, using your own code, or:

- using the provided inference C code in `./3. inference/1. standard/` or `./3. inference/2. quantized/` folders
- using the provided inference Python code located in the `./4. engine/1. standard/` or `./4. engine/2. quantized/` folders

<br/>

In order to use your own trained network with the provided Cerebrum UCI chess engine:

- Copy the `epoch-5.txt` (resp. the `epoch-5-q.txt`) file in the folder `4. engine/1. standard/` (resp. `4. engine/2. quantized/`
- Rename it to `network.txt`
- Launch the engine

<br/>

In order to use your own trained network with Orion UCI chess engine (assuming you did not change the network architecture):

- Install a copy of Orion in a distinct folder of the regular Orion
- Remove any `orion64-v1.0.nn` or `network.txt` existing file in that folder
- Copy the `epoch-5-q.txt` file in the folder
- Rename it to `network.txt`
- Launch the copy of Orion: Orion will not find any `orion64-v1.0.nn` file, but will find a `network.txt` file, and then will convert it to a new `orion64-v1.0.nn` file
- Rename `orion64-v1.0.nn` as you want

Note: this is not super user-friendly, and will be enhanced in a next version ;-)

<br/>

## How to configure name and author

You can adjust the name and author of the trained networks:

- Before training, by modifying the `NN_NAME` (default = "Orion 1.0") and `NN_AUTHOR` (default = "David Carteau") variables in the script `train.py` located in the folder `./2. training/` (see lines 59 and 60)
- After training, by modifying the first two lines of the generated networks (default = "name=Orion 1.0" and "author=David Carteau")

<br/>

## How to replicate Orion 1.0's neural network

If you want to obtain the exact same neural network used in Orion 1.0, additional steps are required:

- Download the [3, 4, 5 pieces](http://tablebase.sesse.net/syzygy/3-4-5/) endgame **Syzygy tablebases** and put them in the folder `./1. data preparation/syzygy/3-4-5/`
- Download the [6 pieces](http://tablebase.sesse.net/syzygy/6-WDL/) endgame Syzygy tablebases and put them in the folder `./1. data preparation/syzygy/6-pieces/`
- Download [CCRL 40/4 archive](https://computerchess.org.uk/ccrl/402.archive/games.html) + [CCRL BLITZ](https://computerchess.org.uk/ccrl/404/games.html) + [CCRL 40/15](https://computerchess.org.uk/ccrl/4040/games.html) games, unzip the 3 files to the folder `./1. data preparation/pgn/`

<br/>

You will also need to use these specific versions of Python and Python librairies:

- `Python 3.11.7`
- `torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu`
- or, if you have an **NVIDIA GPU** `torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121`
- `numpy==1.26.3`
- `scipy==1.11.4`
- `tqdm==4.66.1`
- `chess==1.10.0`

<br/>

Important: to obtain the exact same neural network, let the `FORCE_CPU_DEVICE` variable set to `True` in the script `train.py` (line 53).

<br/>

Additionnal information can be found on the [Orion's website](https://www.orionchess.com/).

<br/>

## Copyright, license

Feel free to adapt the library to meet your specific needs, and do not hesitate to provide feedback ! 🌟

<br/>

The Cerebrum library is licensed under the **MIT License** (see "LICENSE" and "/v1.0/license.txt" files).

Copyright 2024 by David Carteau. All rights reserved.
