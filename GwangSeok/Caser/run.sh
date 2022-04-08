#!/bin/sh

pwd=$(readlink -f .)

/opt/conda/bin/python /opt/ml/paper/RecSys/Caser/train.py
/opt/conda/bin/python /opt/ml/paper/RecSys/Caser/inference.py