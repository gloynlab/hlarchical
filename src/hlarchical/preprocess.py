import os
import pandas as pd
import subprocess
import gzip

class Preprocessor:
    def __init__(self):
        self.HLA = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        self.hla_chrom = ['6', 'chr6']

    def hlarchical_table_to_vcf(self, in_file='1000G_WGS_HLA-HD.txt', genome_build='GRCh37', hla_pos_file='HLA_gene_position.txt'):
        hla_pos_file = hla_pos_file.split('.txt')[0] + f'_{genome_build}.txt'
        if os.path.exists(hla_pos_file) == False:
            self.get_hla_position(out_file=hla_pos_file, genome_build=genome_build)
        df_pos = pd.read_table(hla_pos_file, header=None, sep='\t', dtype=str)
        pos_dict = {}
        for n in range(df_pos.shape[0]):
            gene = df_pos.iloc[n, 0]
            chrom = df_pos.iloc[n, 1]
            pos = df_pos.iloc[n, 2]
            pos_dict[gene] = (chrom, pos)

        df = pd.read_table(in_file, header=0, sep='\t', dtype=str)

        D = {}
        A = []
        S = []
        for k, g in df.groupby('SampleID'):
            if k not in S:
                S.append(k)
            D.setdefault(k, {})
            for gene in self.HLA:
                for a in ['Allele1', 'Allele2']:
                    allele2d = '.'
                    allele4d = '.'
                    allele = g.loc[g['HLA'] == gene, a].values[0]
                    D[k].setdefault(a, [])
                    if allele not in ['.', './.', '0', 'NA']:
                        alleles = allele.split(':')
                        allele2d = f'{alleles[0]}:{alleles[1]}'
                        allele4d = f'{alleles[0]}:{alleles[1]}:{alleles[2]}'
                        D[k][a].append(allele2d)
                        D[k][a].append(allele4d)
                        A.append(allele2d)
                        A.append(allele4d)

        L = []
        for allele in sorted(set(A)):
            gene = allele.split(':')[0]
            if gene in pos_dict:
                chrom, pos = pos_dict[gene]
                ref = 'A'
                alt = 'P'
                row = [chrom, pos, allele, ref, alt, '.', 'PASS', '.', 'GT']
                for sample in S:
                    if allele in D[sample]['Allele1'] and allele in D[sample]['Allele2']:
                        gt = '1/1'
                    elif allele in D[sample]['Allele1'] or allele in D[sample]['Allele2']:
                        gt = '0/1'
                    else:
                        gt = '0/0'
                    row.append(gt)
                L.append(row)

        bfile = in_file.split('.txt')[0]
        df = pd.DataFrame(L)
        df.columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'] + S
        out_file = f'{bfile}.vcf'
        with open(out_file, 'w') as outfile:
            outfile.write('##fileformat=VCFv4.2\n')
            outfile.write('##source=VCFPhaser\n')
            outfile.write('##contig=<ID=6>\n')
            outfile.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        df.to_csv(out_file, sep='\t', index=False, mode='a')

        samples_sorted = f'{bfile}_samples_sorted.txt'
        cmd = f'bcftools query -l {out_file} | sort > {samples_sorted}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        cmd = f'bcftools view -S {samples_sorted} {out_file} -Oz -o {bfile}_sampleSorted.vcf.gz; rm {out_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        cmd = f'bcftools sort {bfile}_sampleSorted.vcf.gz -Oz -o {bfile}.vcf.gz; rm {bfile}_sampleSorted.vcf.gz; bcftools index {bfile}.vcf.gz'
        print(cmd)
        subprocess.run(cmd, shell=True)

    def subset_samples_vcf(self, vcf_file, sample_list, out_file, n_threads=4):
        cmd = f'bcftools view --threads {n_threads} -S {sample_list} {vcf_file} -Oz -o {out_file}; tabix -p vcf {out_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)

    def subset_variants_vcf(self, vcf_file, genome_build='GRCh37', flank=1e6, n_threads=4):
        hla_pos_file=f'HLA_gene_position_{genome_build}.txt' 
        if os.path.exists(hla_pos_file) == False:
            self.get_hla_position(out_file=hla_pos_file, genome_build=genome_build)
        out_file = vcf_file.split('.vcf.gz')[0] + '_variantSubset.vcf.gz'
        df = pd.read_table(hla_pos_file, header=None, sep='\t')
        chrom = df.iloc[0, 1].astype(str)
        start = df.iloc[:, 2].min() - int(flank)
        end = df.iloc[:, 2].max() + int(flank)
        cmd = f'bcftools view --threads {n_threads} -r {chrom}:{start}-{end} {vcf_file} -Oz -o {out_file}; tabix -p vcf {out_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)

    def fixref_vcf(self, vcf_file, genome_build='GRCh37'):
        out_file = vcf_file.split('.vcf')[0] + '_fixref.vcf.gz'
        fasta_file = self.get_genome_reference(genome_build=genome_build)
        # flip/swap using fixref
        cmd = f'bcftools +fixref {vcf_file} -Oz -o {out_file} -- -d -f {fasta_file} -m flip'
        print(cmd)
        subprocess.run(cmd, shell=True)
        # stats after fixref
        cmd = f'bcftools +fixref {out_file} -- -f {fasta_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)

    def make_reference(self, ref_variant_vcf='1000G_array_broad_overlappedWGS_variantSubset_fixref.vcf.gz', ref_hla_vcf='1000G_WGS_HLA-HD_overlappedBroad.vcf.gz', out_file='1000G_REF_phased.vcf.gz', marker_file=None, burnin=10, iterations=15):
        cmd = f'bcftools query -l {ref_variant_vcf}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        ref_variant_samples = result.stdout.strip().split('\n')
        print(f'Number of samples in variant VCF: {len(ref_variant_samples)}')
        cmd = f'bcftools query -l {ref_hla_vcf}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        ref_hla_samples = result.stdout.strip().split('\n')
        print(f'Number of samples in HLA VCF: {len(ref_hla_samples)}')
        if ref_variant_samples != ref_hla_samples:
            print('Error: Sample names in variant and HLA VCF files do not match.')
            return

        concated_file = out_file.split('_phased')[0] + '_concated.vcf.gz'
        bfile = concated_file.split('.vcf.gz')[0]
        pos_uniq_file = bfile + '_posUniq.vcf.gz'
        cmd = f'bcftools concat {" ".join([ref_variant_vcf, ref_hla_vcf])} -Oz -o {concated_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        self.vcf_pos_unique(concated_file)	
        cmd = f'bcftools sort {pos_uniq_file} -Oz -o {concated_file}; bcftools index {concated_file}; rm {pos_uniq_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        cmd = f'beagle gt={concated_file} out={out_file.split(".vcf")[0]} burnin={burnin} iterations={iterations}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        os.remove(concated_file)
        os.remove(f'{concated_file}.csi')

    def phase_sample_on_reference(self, sample_vcf='GDA.vcf.gz', ref_vcf='1000G_REF_phased.vcf.gz', genome_build='GRCh37', subset_variants=True, fix_ref=True, flank=1e6):
        out_file = sample_vcf.split('.vcf')[0] + '_phased_on_' + ref_vcf
        if subset_variants:
            print('subsetting sample variants to HLA region...')
            self.subset_variants_vcf(sample_vcf, genome_build=genome_build, flank=flank)
            sample_vcf = sample_vcf.split('.vcf')[0] + '_variantSubset.vcf.gz'
        if fix_ref:
            print('refix on sample VCF...')
            self.fixref_vcf(sample_vcf, genome_build=genome_build)
            sample_vcf = sample_vcf.split('.vcf')[0] + '_fixref.vcf.gz'
        cmd = f'beagle gt={sample_vcf} ref={ref_vcf} out={out_file.split(".vcf")[0]}'
        print(cmd)
        subprocess.run(cmd, shell=True)

    def get_genome_reference(self, genome_build='GRCh37'):
        if genome_build in ['GRCh38', 'hg38']:
            fasta_url = 'ftp://ftp.ensembl.org/pub/release-101/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz'
        elif genome_build in ['GRCh37', 'hg19']:
            fasta_url = 'ftp://ftp.ensembl.org/pub/release-75/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa.gz'
        elif genome_build in ['GRCh36', 'hg18', 'NCBI36']:
            fasta_url = 'ftp://ftp.ensembl.org/pub/release-54/fasta/homo_sapiens/dna/Homo_sapiens.NCBI36.54.dna.toplevel.fa.gz'
        fasta_file = fasta_url.split('/')[-1].split('.gz')[0]
        files = os.listdir('.')
        if fasta_file not in files:
            subprocess.run(f'wget {fasta_url}; gunzip {fasta_file}.gz', shell=True)
        return fasta_file

    def get_hla_position(self, out_file, genome_build='GRCh38'):
        D = {}
        if genome_build in ['GRCh38', 'hg38']:
            gtf_url = 'ftp://ftp.ensembl.org/pub/release-101/gtf/homo_sapiens/Homo_sapiens.GRCh38.101.gtf.gz'

        elif genome_build in ['GRCh37', 'hg19']:
            gtf_url = 'ftp://ftp.ensembl.org/pub/release-75/gtf/homo_sapiens/Homo_sapiens.GRCh37.75.gtf.gz'

        elif genome_build in ['hg18', 'NCBI36', 'GRCh36']:
            #gtf_url = 'ftp://ftp.ensembl.org/pub/release-54/gtf/homo_sapiens/Homo_sapiens.NCBI36.54.gtf.gz'
            gtf_url = None
            # from DEEP-HLA
            D['HLA-A'] = [('6', 30019970)]
            D['HLA-B'] = [('6', 31431272)]
            D['HLA-C'] = [('6', 31346171)]
            D['HLA-DPA1'] = [('6', 33145064)]
            D['HLA-DPB1'] = [('6', 33157346)]
            D['HLA-DQA1'] = [('6', 32716284)]
            D['HLA-DQB1'] = [('6', 32739039)]
            D['HLA-DRB1'] = [('6', 32660042)]

        if gtf_url:
            files = os.listdir('.')
            self.gtf_file = gtf_url.split('/')[-1].split('.gz')[0]
            if self.gtf_file not in files:
                subprocess.run(f'wget {gtf_url}; gunzip {self.gtf_file}.gz', shell=True)
            print(self.gtf_file)

            with open(self.gtf_file) as infile:
                for line in infile:
                    line = line.strip()
                    fields = line.split('\t')
                    if len(fields) > 8:
                        chrom = fields[0]
                        typ = fields[2]
                        start = int(fields[3])
                        end = int(fields[4])
                        strand = fields[6]
                        info = fields[8]
                        if typ  == 'gene':
                            gene = '.'
                            if info.find('gene_name') != -1:
                                for attr in info.split(';'):
                                    attr = attr.strip()
                                    if attr.startswith('gene_name'):
                                        gene = attr.split(' ')[1].replace('"', '')
                            if chrom in self.hla_chrom and gene in self.HLA:
                                D.setdefault(gene, [])
                                D[gene].append((chrom, start, end, strand))
        L = []
        for gene in self.HLA:
            if gene in D:
                item = sorted(D[gene], key=lambda x: x[1])
                L.append([gene, item[0][0], item[0][1]])
        df = pd.DataFrame(L)
        df.to_csv(out_file, sep='\t', header=False, index=False)

    def vcf_pos_unique(self, in_file):
        seen = {}
        out_file = in_file.replace('.vcf', '_posUniq.vcf')
        if in_file.endswith('.vcf.gz'):
            infile = gzip.open(in_file, 'rt')
            outfile = gzip.open(out_file, 'wt')
        else:
            infile = open(in_file, 'r')
            outfile = open(out_file, 'w')

        for line in infile:
            line = line.strip()
            if line.startswith('#'):
                outfile.write(line + '\n')
            else:
                fields = line.split("\t")
                chrom = fields[0]
                pos = int(fields[1])
                key = (chrom, pos)
                count = seen.get(key, 0)
                seen[key] = count + 1
                new_pos = pos + count
                fields[1] = str(new_pos)
                outfile.write('\t'.join(fields) + '\n')
        infile.close()
        outfile.close()

if __name__ == '__main__':
    pass
