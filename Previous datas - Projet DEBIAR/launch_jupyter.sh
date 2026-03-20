#!/bin/bash

echo "[1/3] ⏱ Chargement des variables d'environnement..."
export PYTHONUSERBASE=$HOME/.local
export PYTHONHOME=$HOME/python310
export PATH=$PYTHONHOME/bin:$PATH
export LD_LIBRARY_PATH=$PYTHONHOME/lib:$PYTHONHOME/lib64:$HOME/openssl-1.1.1w:$LD_LIBRARY_PATH

echo "[2/3] 🐍 Activation de l’environnement Python..."
source ~/gpuenv/bin/activate

echo "[3/3] 🚀 Lancement de Jupyter Notebook (en arrière-plan)..."
nohup jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --NotebookApp.token='' > jupyter.log 2>&1 &