"""
scripts package

This file marks the "scripts" directory as a Python package so that modules
inside (features.py, preprocess.py, etc.) can be imported with
    from scripts.features import engineer_features
even when running `python scripts/predict.py`.
"""