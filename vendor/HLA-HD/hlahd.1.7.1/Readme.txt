Installation and manual of HLA-HD (HLA typing from High-quality Dictionary) version 1.1.0
2017/10/02 Shuji Kawaguchi

1. Requirement

g++ compiler
bowtie2
wget (for updating dictionary)

HLA-HD requires bowtie2 to map NGS reads.
Please install bowtie2 on your computer and set path to your environment variables.
For example, if you are using bash, add to your .bashrc the following command.

export PATH=$PATH:/path_to_bowtie2 


2. Installation and updating of dictionary
   
Move to hla_estimation directory (same directory of this Readme.txt) and type the following command.

> sh install.sh

For the installation, the g++ compiler by the GNU Compiler Collection must be installed on your computer.

3. Updating the HLA dictionary (after v.1.1.0)
HLA dictionary can be updated to current release version by typing:

> sh update.dictionary.sh


4. Preperation 

4.1 Add path to HLA-HD 

You should add the current directory to your PATH 
export PATH=$PATH:/path_to_HLA-HD_install_directory/bin

4.2 Change the limitation of open files (If need)

Check open files on your computer by

> ulimit -Sa

If open files are less than 1024, please type 

> ulimit -n 1024

before the running of HLA-HD or change /etc/security/limits.conf according to your system environment.


5. Running
If you have fastq.gz file, unzip gz in advance. 

You can run the program by typing the following commands.

> hlahd.sh -t [thread_num] -m [minimum length of reads] -f [path_to freq_data directory] fastq_1 fastq_2 gene_split_filt path_to_dictionary IDNAME[any name] output_directory 

output_directory must be created before the run.

#Example

> hlahd.sh -t 2 -m 100 -f freq_data/ data/sample_1.fastq data/sample_2.fastq HLA_gene.split.txt dictionary/ sampleID estimation 


6. Output

HLA alleles called by HLA-HD will be recorded to current_directory/output_directory/sampleID/result/sampleID_final.result.txt.

File example

Gene1	Allele1	Allele2
Gene2	Allele1	-
Gene3	Allele1(Candidate1)	Allele2(Candidate1)	Allele1(Candidate2)	Allele2(Candidate2)
Gene4	Not typed	Not typed
.
.
.

Text format is a tab separated values (TSV).
One allele is represented by a hyphen if the gene was typed as having homozygous alleles (line of Gene2).
Multiple candidates are written by parallel notation (line of Gene3).
If allele pair was not determined, result is written as "Not typed" (line of Gene4).
