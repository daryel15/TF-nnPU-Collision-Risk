import argparse
from .artifacts import verify_manifest, validate_data_and_splits
from .evaluation import evaluate_checkpoint
parser=argparse.ArgumentParser(description='Verify the release or reproduce the supplied checkpoint.')
parser.add_argument('command',choices=['verify','evaluate'])
parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
parser.add_argument('--output',default=None,help='New output directory; existing directories are not overwritten.')
args=parser.parse_args()
if args.command=='verify':
    print('Verified hashes:',verify_manifest()); print(validate_data_and_splits())
else: evaluate_checkpoint(args.device,args.output)
