import os

class Seq:
    def __init__(self):
        pass

    def subset_cram_to_fastq(self, cram_file=['HPAP001.cram'], fasta_file='Homo_sapiens_assembly38.fasta', bed_file='keep_chroms.bed', out_file='subset_cram_to_fastq.sh', n_threads=4):
        '''
        bed_file should include chr6 plus other chroms other than chr1-chr5, chr7-chr22, chrX, chrY, chrMT, such as chr1_KI270706v1_random and chrUn_KI270302v1 etc.
        '''
        with open(out_file, 'w') as outfile:
            for f in cram_file:
                subset_file = os.path.basename(f).replace('.cram', '_subset.cram')
                fq1 = subset_file.replace('.cram', '.R1.fastq.gz')
                fq2 = subset_file.replace('.cram', '.R2.fastq.gz')
                cmd = f'samtools view -T {fasta_file} -C -L {bed_file} {f} > {subset_file}'
                cmd2 = f'samtools collate -u -O {subset_file} | samtools fastq --reference {fasta_file} -@ {n_threads} -1 {fq1} -2 {fq2} -0 /dev/null -s /dev/null -n'
                outfile.write(cmd + '\n')
                outfile.write(cmd2 + '\n')

    def run_hlahd(self, fq1_file=['HPAP001_subset.R1.fastq.gz'], freq_data='freq_data', dict_file='dictionary', gene_split_file='HLA_gene.split.txt', out_file='run_hlahd.sh', out_dir='HLA-HD', n_threads=4, min_read_length=100, trim_rate=0.95):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(out_file, 'w') as outfile:
            for fq1 in fq1_file:
                fq2 = fq1.replace('.R1.fastq.gz', '.R2.fastq.gz')
                sample_id = os.path.basename(fq1).split('_subset')[0].split('.R1')[0].split('.R2')[0]
                cmd = f'hlahd.sh -t {n_threads} -m {min_read_length} -c {trim_rate} -f {freq_data} {fq1} {fq2} {gene_split_file} {dict_file} {sample_id} {out_dir}'
                outfile.write(cmd + '\n')

    def run_xhla(self, fq1_file=['HPAP001_subset.R1.fastq.gz'], sif_file='xhla.sif', genome_index='hg38', out_file='run_xhla.sh', n_threads=4, out_dir='xHLA'):
        # genome reference must be hg38 rather than GRCh38, because the HLA region has been hard coded with chr in xHLA.
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(out_file, 'w') as outfile:
            for fq1 in fq1_file:
                fq2 = fq1.replace('.R1.fastq.gz', '.R2.fastq.gz')
                sample_id = os.path.basename(fq1).split('_subset')[0]
                bam_file = f'{out_dir}/{sample_id}.bam'
                cmd = f'bwa mem -t {n_threads} {genome_index} {fq1} {fq2} | samtools sort -@ {n_threads} -o {bam_file}; samtools index {bam_file}'
                outfile.write(cmd + '\n')
                cmd = f'singularity exec -B `pwd`:`pwd` --pwd `pwd` {sif_file} run.py --sample_id {sample_id} --input_bam_path {bam_file} --output_path {out_dir}/{sample_id}'
                outfile.write(cmd + '\n')

    def run_optitype(self, fq1_file=['HPAP001_subset.R1.fastq.gz'], sif_file='OptiType.sif', out_file='run_optitype.sh', n_threads=4, out_dir='OptiType', seq_type='dna'):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(out_file, 'w') as outfile:
            for fq1 in fq1_file:
                fq2 = fq1.replace('.R1.fastq.gz', '.R2.fastq.gz')
                sample_id = os.path.basename(fq1).split('_subset')[0]
                cmd = f'singularity exec -B `pwd`:`pwd` --pwd `pwd` {sif_file} OptiTypePipeline.py --{seq_type} -i {fq1} {fq2} -o {out_dir} -p {sample_id} -v'
                outfile.write(cmd + '\n')
