"""Adapt a complete SpatialTIP checkpoint to one expression/image sample.

Optional boolean heldout masks use True=excluded evaluation entry. Those entries
must already be zero in the supplied expression matrix and are excluded from MLM
targets. A heldout mask is unnecessary for ordinary target adaptation.
"""

import argparse
from pathlib import Path

from train import add_training_arguments, run_training, _existing_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expression', required=True, help='Prepared H5AD with gene names, tissue IDs and spatial coordinates')
    parser.add_argument('--patches', required=True, help='HDF5 RGB patches indexed by barcode')
    parser.add_argument('--heldout', help='Optional scipy sparse boolean NPZ; True excludes an externally held-out entry')
    parser.add_argument('--sample', help='Sample label for the run manifest; defaults to the expression filename stem')
    add_training_arguments(parser, finetune=True)
    args = parser.parse_args()
    expression = _existing_file(args.expression, 'Expression')
    patches = _existing_file(args.patches, 'Patches')
    heldout = str(_existing_file(args.heldout, 'Heldout mask')) if args.heldout else None
    samples = [{'sample': args.sample or Path(expression).stem, 'tissue': 'from expression.obs[tissue]',
                'expression': str(expression), 'patches': str(patches), 'heldout': heldout}]
    run_training(args, samples)


if __name__ == '__main__':
    main()
