import argparse
from .preprocess import Preprocessor
from .process import Processor

from .dataset import CustomDataset
from .trainer import Trainer
from .utils import *
from .array import Array

def get_parser():
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest='command', required=True)

    p1 = subparsers.add_parser("get-hlarchical-table", help="get the table in the format defined by hlarchical")
    p1.add_argument('--input', type=str, default='data/1958BC_Euro.bgl.phased', help='input file')
    p1.add_argument('--output', type=str, default='data/1958BC_Euro_digit4.txt', help='output file')
    p1.add_argument('--digit', type=int, default=4, help='digit level for HLA alleles')
    p1.add_argument('--from_tool', type=str, default='snp2hla', help='the tool that generated the input file')

    p2 = subparsers.add_parser("run-snp2hla", help="run SNP2HLA on array data")
    p2.add_argument('--input', type=str, default='1958BC', help='input file prefix')
    p2.add_argument('--ref', type=str, default='HM_CEU_REF', help='reference panel prefix, can be HM_CEU_REF or Pan-Asian_REF currently')
    p2.add_argument('--output', type=str, default='1958BC_European_SNP2HLA', help='output file prefix')

    p3 = subparsers.add_parser("run-hibag", help="run HIBAG on array data, HIBAG package should be installed in R environment")
    p3.add_argument('--input', type=str, default='1958BC', help='input file prefix')
    p3.add_argument('--ref', type=str, default='European', help='reference panel prefix, can be European, Asian, African, or Hispanic currently')
    p3.add_argument('--output', type=str, default='1958BC_European_HIBAG', help='output file prefix')
    p3.add_argument('--Renv', type=str, default='R4.5', help='conda R environment name where HIBAG is installed')

    p4 = subparsers.add_parser("run-deephla", help="run CNN-based DEEP*HLA, to be implemented")
    p4.add_argument('--mode', type=str, default='train', help='mode: train or impute')
    p4.add_argument('--input', type=str, default='1958BC_Pan-Asian_REF', help='input file prefix')
    p4.add_argument('--ref', type=str, default='Pan-Asian_REF', help='reference panel prefix, can be HM_CEU_REF or Pan-Asian_REF currently')
    p4.add_argument('--subset', type=str, default=None, help='subset the input to the HLA regions according to the reference genome, e.g., chr6:28510120-33480577 on GRCh37')
    p4.add_argument('--model_json', type=str, default='Pan-Asian_REF.model.json', help='the config file of the model')
    p4.add_argument('--model_dir', type=str, default='model', help='the output directory of the trained model')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.command == 'get-hlarchical-table':
        ar = Array()
        ar.get_hlarchical_table(in_file=args.input, out_file=args.output, digit=args.digit, from_tool=args.from_tool)
    elif args.command == 'run-snp2hla':
        ar = Array()
        ar.run_snp2hla(in_file=args.input, ref_file=args.ref, out_file=args.output)
    elif args.command == 'run-hibag':
        ar = Array()
        ar.run_hibag(in_file=args.input, ref=args.ref, out_file=args.output, Renv=args.Renv)
    elif args.command == 'run-deephla':
        ar = Array()
        ar.run_deephla(mode=args.mode, in_file=args.input, ref_file=args.ref, subset=args.subset,
                       model_json=args.model_json, model_dir=args.model_dir)

if __name__ == '__main__':
    main()
