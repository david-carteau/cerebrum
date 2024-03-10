## The Cerebrum library and engine

![Logo](/v1.0/logo.png)

The **Cerebrum library** can be used to train and utilize "[NNUE](https://www.chessprogramming.org/NNUE)-like" neural networks for chess engines. It was initially designed and created for the [Orion UCI chess engine](https://www.orionchess.com/).

<br/>

The library consists of three main parts:

1. **Training code** (Python script)
2. **Inference code** (C files)
3. A **basic UCI chess engine** for demonstration purposes (Python script).

<br/>

To use the library, you will need:

- A **Python** runtime: https://www.python.org/
- Some Python librairies: `pip install torch numpy scipy tqdm chess`

<br/>

Note that, **by default, training is forced to use the CPU**. If you have an NVIDIA GPU, you can:

- Install the appropriate version of the torch library (see links below) to improve training speed
- Set `FORCE_CPU_DEVICE` to `False` in the script `train.py` (line 53) to enable GPU usage

<br/>

If you want to obtain the exact same neural network used in Orion 1.0, additional steps are required (here, for Windows):

- Download the [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) tool and put the `pgn-extract.exe` file in the folder `./1. training/1. data preparation`
- Download the [3, 4, 5 pieces](http://tablebase.sesse.net/syzygy/3-4-5/) endgame **Syzygy tablebases** and put them in the folder `./1. training/1. data preparation/syzygy/3-4-5/`
- Download the [6 pieces](http://tablebase.sesse.net/syzygy/6-WDL/) endgame Syzygy tablebases and put them in the folder `./1. training/1. data preparation/syzygy/6-pieces/`

<br/>

You will also need to install these specific versions of Python librairies:

- `torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu`
- or, if you have an **NVIDIA GPU** `torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121`
- `numpy==1.26.3`
- `scipy==1.11.4`
- `tqdm==4.66.1`
- `chess==1.10.0`

<br/>

Important: to obtain the exact same neural network, let the `FORCE_CPU_DEVICE` variable set to `True` in the script `train.py` (line 53).

<br/>

Feel free to adapt the library to meet your specific needs! 🌟

<br/>

## Copyright, license

Copyright 2024 by David Carteau. All rights reserved.

The Cerebrum library is licensed under the **MIT License** (see "LICENSE" and "/v1.0/license.txt" files).
