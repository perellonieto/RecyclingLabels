#!/bin/sh

python -m unittest discover recybels
python -m unittest discover experiments
python -m unittest discover wlc
python wlc/WLweakener.py
