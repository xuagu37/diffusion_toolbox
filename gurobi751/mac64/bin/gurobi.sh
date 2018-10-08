#!/bin/sh

export PYTHONPATH=/Library/gurobi751/mac64/lib
export PYTHONSTARTUP=/Library/gurobi751/mac64/lib/gurobi.py

/usr/bin/python2.7 "$@"
