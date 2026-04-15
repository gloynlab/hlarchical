#!/bin/bash

g++ ./src/split_shell.cpp -O3 -o ./bin/split_shell
g++ ./src/get_diff_fasta.cpp -O3 -o ./bin/get_diff_fasta
g++ ./src/sam_to_fastq_reverse.cpp -O3 -o ./bin/stfr
g++ ./src/split_PM_reads.cpp -O3 -o ./bin/split_pm_read
g++ ./src/pick_up_allele.cpp -O3 -o ./bin/pick_up_allele
g++ ./src/pm_extract.cpp -O3 -o ./bin/pm_extract
g++ ./src/hla_estimation.cpp -O3 -o ./bin/hla_est
g++ ./src/drop_intron_map.cpp -O3 -o ./bin/drop_intron_map

cd dictionary/
sh bw_build.sh
cd ../
