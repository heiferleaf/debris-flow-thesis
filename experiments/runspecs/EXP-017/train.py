"""Standalone production training is disabled; use the bounded five-fold run.sh.

The Python train_stage API remains internal to pipeline.py. A one-stage CUDA CLI
cannot bind the complete dense/preflight/budget/timer/shutdown contract.
"""
import argparse


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_known_args(argv)
    parser.error('Standalone EXP-017 training is disabled. Use experiments/runspecs/EXP-017/run.sh with the complete preflight, dense reference, budget and platform timer evidence.')


if __name__=='__main__':main()
