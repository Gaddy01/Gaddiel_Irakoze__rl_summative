"""Convenience launcher: python -m training.dqn_training --episodes 20"""
import sys
from training.experiment import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--algorithm", "dqn"]
    main()
