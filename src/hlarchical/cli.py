import argparse
from .preprocess import Preprocessor
from .process import Processor
from .dataset import CustomDataset
from .trainer import Trainer
from .array import Array
from .summary import Summary
from .utils import *

def get_parser():
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest='command', required=True)

    p0 = subparsers.add_parser("easy-predict", help="a one-line command to predict HLA alleles using the trained model with default parameters")
    p0.add_argument('--vcf', type=str, default='input.vcf.gz', help='input vcf file of the sample data to be predicted, must be on GRCh37 (CHROM has no chr prefix)')
    p0.add_argument('--output', type=str, default='predicted.txt', help='the output file for the predicted HLA alleles')

    p1 = subparsers.add_parser("phase-sample-on-ref", help="phase the sample data on the reference panel")
    p1.add_argument('--vcf', type=str, default='input.vcf.gz', help='input vcf file of the sample data to be predicted')
    p1.add_argument('--ref', type=str, default='1000G_REF_phased.vcf.gz', help='the phased reference panel, generated using the make_reference function, the default one is on GRCh37')
    p1.add_argument('--genome_build', type=str, default='GRCh37', help='the genome build of the sample array data, must be the same as the reference panel')

    p2 = subparsers.add_parser("get-sample-features", help="get the features of the sample data for prediction")
    p2.add_argument('--vcf', type=str, default='input_phased.vcf.gz', help='the phased sample data')
    p2.add_argument('--features', type=str, default='features_list.txt', help='the list of features used in the model training, generated using the make_features function')
    p2.add_argument('--with_ancestry', type=str, default='False', help='using ancestry as features to do the prediction')
    p2.add_argument('--ancestry_file', type=str, default=None, help='ancestry file with samples in the first column and ancestry info in the second column')
    p2.add_argument('--output', type=str, default='to_predict.txt', help='the output file for the features of the sample data, in the same format as the feature file used in the model training')

    p3 = subparsers.add_parser("predict", help="predict HLA alleles using the trained model")
    p3.add_argument('--input', type=str, default='to_predict_without_ancestry.txt', help='the input file for prediction, generated using the get-sample-features command')
    p3.add_argument('--output', type=str, default='predicted.txt', help='the output file for the predicted HLA alleles')
    p3.add_argument('--model_name', type=str, default='mlp', help='the name of the model to be used for prediction')
    p3.add_argument('--epoch', type=int, default=200, help='the epoch of the trained model to be used for prediction')
    p3.add_argument('--config_file', type=str, default='config.yaml', help='the config file used in the model training')
    p3.add_argument('--maps_file', type=str, default='maps.txt', help='the maps file used in the model training, generated using the make_maps function')
    p3.add_argument('--masks_file', type=str, default='masks.txt', help='the masks file used in the model training, generated using the make_masks function')
    p3.add_argument('--with_ancestry', type=str, default='False', help='using ancestry as features to do the prediction')

    p4 = subparsers.add_parser("get-hlarchical-table", help="get the table in the format defined by hlarchical")
    p4.add_argument('--input', type=str, default='data/1958BC_Euro.bgl.phased', help='input file')
    p4.add_argument('--output', type=str, default='data/1958BC_Euro_digit4.txt', help='output file')
    p4.add_argument('--digit', type=int, default=4, help='digit level for HLA alleles')
    p4.add_argument('--from_tool', type=str, default='snp2hla', help='the tool that generated the input file')

    p11 = subparsers.add_parser("run-snp2hla", help="run SNP2HLA on array data")
    p11.add_argument('--input', type=str, default='1958BC', help='input file prefix')
    p11.add_argument('--ref', type=str, default='HM_CEU_REF', help='reference panel prefix, can be HM_CEU_REF or Pan-Asian_REF currently')
    p11.add_argument('--output', type=str, default='1958BC_European_SNP2HLA', help='output file prefix')

    p12 = subparsers.add_parser("run-hibag", help="run HIBAG on array data, HIBAG package should be installed in R environment")
    p12.add_argument('--input', type=str, default='1958BC', help='input file prefix')
    p12.add_argument('--ref', type=str, default='European', help='reference panel prefix, can be European, Asian, African, or Hispanic currently')
    p12.add_argument('--output', type=str, default='1958BC_European_HIBAG', help='output file prefix')
    p12.add_argument('--Renv', type=str, default='R4.5', help='conda R environment name where HIBAG is installed')

    p13 = subparsers.add_parser("run-deephla", help="run CNN-based DEEP*HLA on array data")
    p13.add_argument('--mode', type=str, default='train', help='mode: train or impute')
    p13.add_argument('--input', type=str, default='OMNI_Pan-Asian_REF', help='input file prefix')
    p13.add_argument('--output', type=str, default='OMNI_Pan-Asian', help='output file prefix')
    p13.add_argument('--ref', type=str, default='Pan-Asian_REF', help='reference panel prefix, can be HM_CEU_REF or Pan-Asian_REF currently')
    p13.add_argument('--model_json', type=str, default='Pan-Asian_REF.model.json', help='the config file of the model')
    p13.add_argument('--hla_json', type=str, default='Pan-Asian_REF.hla.json', help='the config file of the HLA genes')
    p13.add_argument('--model_dir', type=str, default='model_Pan-Asian', help='the output directory of the trained model')
    p13.add_argument('--subset', type=str, default=None, help='subset the input to the HLA regions according to the reference genome, e.g., chr6:28510120-33480577 on GRCh37')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.command == 'easy-predict':
        hla = Preprocessor()
        sample_vcf = args.vcf
        ref_vcf = '1000G_REF_phased.vcf.gz'
        genome_build = 'GRCh37'
        hla.phase_sample_on_reference(sample_vcf=sample_vcf, ref_vcf=ref_vcf, genome_build=genome_build)

        with_ancestry = False
        ancestry_file = None
        features_file = 'features_list.txt'
        out_file = 'to_predict.txt'
        hla = Processor(with_ancestry=with_ancestry, ancestry_file=ancestry_file)
        hla.get_sample_features(sample_vcf=sample_vcf, features_file=features_file, out_file=out_file)

        in_file = 'to_predict.txt'
        out_file = args.output
        model_name = 'mlp'
        config_file = 'config.yaml'
        epoch = 200
        maps_file = 'maps.txt'
        masks_file = 'masks.txt'
        hla = Trainer(config_file=config_file, model_name=model_name, maps_file=maps_file, masks_file=masks_file, with_ancestry=with_ancestry)
        hla.predict(pred_file=in_file, out_file=out_file, epoch=epoch)

    elif args.command == 'phase-sample-on-ref':
        hla = Preprocessor()
        sample_vcf = args.vcf
        ref_vcf = args.ref
        genome_build = args.genome_build
        hla.phase_sample_on_reference(sample_vcf=sample_vcf, ref_vcf=ref_vcf, genome_build=genome_build)
    elif args.command == 'get-sample-features':
        sample_vcf = args.vcf
        features_file = args.features
        ancestry_file = args.ancestry_file
        with_ancestry = True if args.with_ancestry.lower() in ['true', 'yes'] else False
        ancestry_file = args.ancestry_file
        hla = Processor(with_ancestry=with_ancestry, ancestry_file=ancestry_file)
        out_file = args.output
        hla.get_sample_features(sample_vcf=sample_vcf, features_file=features_file, out_file=out_file)
    elif args.command == 'predict':
        in_file = args.input
        out_file = args.output
        model_name = args.model_name
        config_file = args.config_file
        epoch = args.epoch
        maps_file = args.maps_file
        masks_file = args.masks_file
        with_ancestry = True if args.with_ancestry.lower() in ['true', 'yes'] else False
        hla = Trainer(config_file=config_file, model_name=model_name, maps_file=maps_file, masks_file=masks_file, with_ancestry=with_ancestry)
        hla.predict(pred_file=in_file, out_file=out_file, epoch=epoch)
    elif args.command == 'run-snp2hla':
        hla = Array()
        hla.run_snp2hla(in_file=args.input, ref_file=args.ref, out_file=args.output)
    elif args.command == 'run-hibag':
        hla = Array()
        hla.run_hibag(in_file=args.input, ref=args.ref, out_file=args.output, Renv=args.Renv)
    elif args.command == 'run-deephla':
        hla = Array()
        hla.run_deephla(mode=args.mode, in_file=args.input, out_file=args.output, ref_file=args.ref,
                        hla_json=args.hla_json, model_json=args.model_json, model_dir=args.model_dir, subset=args.subset)
    elif args.command == 'get-hlarchical-table':
        hla = Summary()
        hla.get_hlarchical_table(in_file=args.input, out_file=args.output, digit=args.digit, from_tool=args.from_tool)

if __name__ == '__main__':
    main()
