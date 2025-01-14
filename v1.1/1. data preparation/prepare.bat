@echo off

REM NAME: prepare.bat
REM AUTHOR: David Carteau, France, January 2025
REM LICENSE: MIT (see "license.txt" file content)
REM PURPOSE: Prepare training data

python combine.py

REM [TIP] add '-Tdb2025' if you want to exclude games played after 31.12.2024
pgn-extract.exe -s -Wepd "games.pgn" | python split.py

python merge.py
python syzygy.py
python select.py

pause
