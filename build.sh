#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt
python nltk_setup.py
python train_model.py
