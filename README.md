<p align="left">
<img src="assets/logo.png" alt="logo" width="250"/>
</p>

## Overview

**Hlarchical** is implementing deep learning models for imputing human leukocyte antigen (HLA) alleles from genotyping data. HLA genes are highly polymorphic and play a critical role in immune response, disease susceptibility, drug hypersensitivity, and transplant compatibility.

While high-resolution HLA typing using sequencing technologies is accurate, it remains expensive and unavailable for many large-scale studies. Existing methods for imputing HLA types from genotyping data include Hidden Markov Model–based SNP2HLA, machine learning–based HIBAG, convolutional neural network–based DEEP*HLA, and Transformer-based HLARIMNT, among others.

Inspired by previous studies, hlarchical aims to improve HLA imputation performance by exploring and exploiting the following features:

(1) hierarchical modeling of HLA alleles that reflects the natural structure of HLA nomenclature (e.g., 2-digit -> 4-digit resolution)

(2) mixture-of-experts (MoE) architectures that enable allele-specific experts to focus on relevant subsets of SNP features while sharing information across related alleles

(3) multi-task learning to jointly optimize predictions across multiple HLA resolutions and loci

(4) configurable model backbones (e.g., MLP, CNN, GPT) and hyperparameters to enable systematic evaluation and optimization

## Installation

- using conda

```
git clone git@github.com:HaniceSun/hlarchical.git
cd hlarchical
conda env create -f environment.yml
conda activate hlarchical
```

# Quick Start

```python

# Predict HLA alleles for a new array dataset using the trained model

hla phase-sample-on-ref --vcf 1000G_array_sanger.vcf.gz
hla get-sample-features --vcf 1000G_array_sanger_phased_on_1000G_REF_phased.vcf.gz
hla predict --input to_predict.txt --model_name mlp --epoch 200

# Train a model on the 1000 Genomes dataset

from hlarchical.array import Array
from hlarchical.preprocess import Preprocessor
from hlarchical.process import Processor
from hlarchical.dataset import CustomDataset
from hlarchical.trainer import Trainer

hla = Array()
hla.get_hlarchical_table(in_dir='HLA-HD', out_file='1000G_WGS_HLA-HD.txt', digit=4, from_tool='hla-hd')

hla = Preprocessor()
hla.hlarchical_table_to_vcf(in_file='1000G_WGS_HLA-HD.txt')
hla.subset_samples_vcf(vcf_file='1000G_array_broad.vcf.gz', sample_list='1000G_array_broad_samples_overlappedWGS.txt', out_file='1000G_array_broad_overlappedWGS.vcf.gz')
hla.subset_samples_vcf(vcf_file='1000G_WGS_HLA-HD.vcf.gz', sample_list='1000G_array_broad_samples_overlappedWGS.txt', out_file='1000G_WGS_HLA-HD_overlappedBroad.vcf.gz')
hla.subset_variants_vcf(vcf_file='1000G_array_broad_overlappedWGS.vcf.gz', genome_build='GRCh37')
hla.fixref_vcf(vcf_file='1000G_array_broad_overlappedWGS_variantSubset.vcf.gz')
hla.make_reference(ref_variant_vcf='1000G_array_broad_overlappedWGS_variantSubset_fixref.vcf.gz',
ref_hla_vcf='1000G_WGS_HLA-HD_overlappedBroad.vcf.gz', out_file='1000G_REF_phased.vcf.gz', burnin=10, iterations=15)

hla = Processor(ref_phased='1000G_REF_phased.vcf.gz', label_include=['HLA'], feature_exclude=['HLA'], expert_by='ld')
hla.make_features()
hla.make_maps()
hla.make_labels()
hla.make_masks()
hla = CustomDataset(features_file='features.txt', labels_file='labels.txt', maps_file='maps.txt', out_file='hla')
hla.split_save_dataset(ratio=[0.8, 0.2])

train_file='hla_dataset_train.pt'
val_file='hla_dataset_test.pt'
test_file='hla_dataset_test.pt'
config_file='config.yaml'
model_name='mlp'
hla = Trainer(config_file=config_file, model_name=model_name, train_file=train_file, val_file=val_file, test_file=test_file)
hla.count_parameters()
hla.run(end_epoch=300)
```

## Author and License

**Author:** Han Sun

**Email:** hansun@stanford.edu

**License:** [MIT License](LICENSE)
