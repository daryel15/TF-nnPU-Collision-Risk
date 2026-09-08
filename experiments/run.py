import argparse
import importlib
parser=argparse.ArgumentParser(description='Display reference results or opt into training.')
parser.add_argument('experiment',choices=['gat','cmpa','sulstm','cost_sensitive','sensitivity','stage2'])
parser.add_argument('--train',action='store_true')
parser.add_argument('--smoke',action='store_true',help='One epoch on reduced samples; not a scientific result.')
parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
parser.add_argument('--output',default=None)
args=parser.parse_args()
if args.smoke and not args.train: parser.error('--smoke requires --train')
module=importlib.import_module('experiments.'+args.experiment)
module.run(train=args.train,smoke=args.smoke,output=args.output,device_name=args.device)
