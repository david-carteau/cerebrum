@echo off

REM NAME: prepare.bat
REM AUTHOR: David Carteau, France, November 2025
REM LICENSE: MIT (see "license.txt" file content)
REM PURPOSE: Prepare training data

python games_select.py

REM [TIP] add --gamelimit 1024
REM [TIP] add '-Tda2024' if you want to only keep games played after 2024
REM [TIP] add '-Tdb2025' if you want to only keep games played before 2025

pgn-extract.exe -s -Wepd games.pgn | python positions_split.py

python positions_merge.py
python positions_select.py
python positions_shuffle.py

pause
