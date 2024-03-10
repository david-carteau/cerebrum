## The Cerebrum library and engine

![Logo](/v1.0/logo.png)

The **Cerebrum library** can be used to train and utilize **“NNUE-like” neural networks** for chess engines. It was initially designed and created for the Orion UCI chess engine.

The library consists of three main parts:

    **Training code** (Python script)
    **Inference code** (C files)
    A **basic UCI chess engine** for demonstration purposes (Python script).

To use the library, you will need:

    a **Python** runtime: https://www.python.org/
    some Python librairies: `pip install torch numpy scipy tqdm chess`

If you want to obtain the exact same neural network used in Orion 1.0, additional steps are required (here, for Windows):

    download the **pgn-extract** tool: https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/
    put the `pgn-extract.exe` file in the folder "./cerebrum/v1.0/1. training/1. data preparation"
    download the 3, 4, 5 pieces endgame **Syzygy tablebases**: http://tablebase.sesse.net/syzygy/3-4-5/
    put the downloaded files in the folder "./cerebrum/v1.0/1. training/1. data preparation/syzygy/3-4-5/"
    download the 6 pieces endgame **Syzygy tablebases**: http://tablebase.sesse.net/syzygy/6-WDL/
    put the downloaded files in the folder "./cerebrum/v1.0/1. training/1. data preparation/syzygy/6-pieces/"

You will also need to install these specific versions of Python librairies:

    `torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu` (or `torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121` if you have a **NVIDA GPU**)
    `numpy==1.26.3`
    `scipy==1.11.4`
    `tqdm==4.66.1`
    `chess==1.10.0`

Note that, by default, training is forced to use the CPU. If you have a NVIDIA GPU, be sure to install the correct version of the **torch** library and set `FORCE_CPU_DEVICE` to `False` in the script `train.py` (line 53).

Feel free to adapt the library to meet your specific needs! 🌟

## Copyright, license

Copyright 2024 by David Carteau. All rights reserved.

The Cerebrum library is licensed under the **MIT License** (see "LICENSE" and "/v1.0/license.txt" files).
