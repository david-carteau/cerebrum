@echo off

REM NAME: prepare.bat
REM AUTHOR: David Carteau, France, March 2024
REM LICENSE: MIT (see "license.txt" file content)
REM PURPOSE: Prepare training data for Orion UCI chess engine's neural network

cd /d %~dp0

python combine.py

REM [IMPORTANT]
REM remove '-Tdb2024' if you don't want to exclude games played after 31.12.2023
pgn-extract.exe -s -Tdb2024 -Wepd "games.pgn" | python split.py

python merge.py
python syzygy.py
python select.py

pause
