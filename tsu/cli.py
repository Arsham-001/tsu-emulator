"""
TSU CLI (early stub)
Later: add commands for sampling, benchmarks, visualization, cloud job submission.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(prog="tsu", description="TSU Platform CLI")
    subparsers = parser.add_subparsers(dest="command")

    sample_parser = subparsers.add_parser("sample", help="Run a quick Gaussian sample")
    sample_parser.add_argument("--n", type=int, default=5, help="Number of samples")
    sample_parser.add_argument("--mu", type=float, default=0.0, help="Mean")
    sample_parser.add_argument("--sigma", type=float, default=1.0, help="Std dev")

    subparsers.add_parser("version", help="Show TSU version")

    args = parser.parse_args()

    if args.command == "sample":
        from tsu.core import ThermalSamplingUnit

        tsu = ThermalSamplingUnit()
        samples = tsu.sample_gaussian(mu=args.mu, sigma=args.sigma, n_samples=args.n)
        print("Samples:", samples)
    elif args.command == "version":
        from tsu import __version__

        print("TSU version:", __version__)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
