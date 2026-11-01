"""Vendored, unmodified model code from the official CBraMod repository.

Source: https://github.com/wjq-learning/CBraMod (commit b9e961003214326972c567eff390e75b0287e32a),
files models/cbramod.py and models/criss_cross_transformer.py, MIT License (see LICENSE in
this directory, Copyright (c) 2025 Jiquan Wang).

Only change: the absolute import ``from models.criss_cross_transformer`` in cbramod.py was
made package-relative so it can be imported from this project.
"""
from .cbramod import CBraMod  # noqa: F401
