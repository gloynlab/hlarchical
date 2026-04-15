#!/bin/bash

Version="1.6.1"

PN=1
CR=1.0
TSIZE=100
MTSIZE=50
NSIZE=5
NLEN="150"

CMDNAME=$(basename $0)


echo "HLA-HD version "${Version} 

while getopts ot:c:m:n:f:N: OPT
do
  case $OPT in
    "o" ) FLG_O="TRUE" ;;
    "t" ) FLG_T="TRUE" ; VALUE_T="$OPTARG" ;;
    "c" ) FLG_C="TRUE" ; VALUE_C="$OPTARG" ;;
    "m" ) FLG_M="TRUE" ; VALUE_M="$OPTARG" ;;
    "n" ) FLG_N="TRUE" ; VALUE_N="$OPTARG" ;;
    "f" ) FLG_F="TRUE" ; VALUE_F="$OPTARG" ;;
    "N" ) FLG_NL="TRUE" ; VALUE_NL="$OPTARG" ;;
      * ) echo "Usage: $CMDNAME [-t thread] [-c rate_of_cutting] [-m minmum_tag_size] [-n number of mismatch] [-f freq_dir] [-N size of ambiguous charactor] fastqfile1 fastqfile2 HLA_gene.split.txt Dictionary_Path sampl_id(arbitary) output_directory" 1>&2
          exit 1 ;;
  esac
done

FREQ=""
if [ "$FLG_T" = "TRUE" ]; then
  PN=$VALUE_T  
fi
if [ "$FLG_C" = "TRUE" ]; then
  CR=$VALUE_C  
fi
if [ "$FLG_M" = "TRUE" ]; then
  TSIZE=$VALUE_M  
fi
if [ "$FLG_N" = "TRUE" ]; then
  NSIZE=$VALUE_N  
fi
if [ "$FLG_NL" = "TRUE" ]; then
    NLEN=$VALUE_NL
fi

MTSIZE=`expr ${TSIZE} / 2`

shift `expr $OPTIND - 1`

if [ $# -ne 6 ]; then
    echo "Usage: $CMDNAME [-t thread] [-c rate_of_cutting] [-m minmum_tag_size] [-n number of mismatch] [-f freq_dir] [-N size of ambiguous charactor] fastqfile1 fastqfile2 HLA_gene.split.txt HLA_Div_fasta sampl_id(arbitary) output_directory " 1>&2
 exit 1;
fi

PN=`expr ${PN}`

F1=$1
F2=$2
GE=$3
HD=$4
ID=$5
OD=$6

if [ -e ${OD}/ ]; then
:
else
  echo "can't find directory ${OD}" 1>&2
  exit 1;
fi
if [ -e ${OD}/${ID}/ ]; then
:
else
mkdir ${OD}/${ID}
fi
if [ -e ${OD}/${ID}/mapfile ]; then
:
else
mkdir ${OD}/${ID}/mapfile
fi

if [ "$FLG_O" = "TRUE" ]; then
#BW="bowtie2 --mm --np 0 --n-ceil L,0,0.5 -p "$PN
BW="bowtie2 --mm --np 0 --n-ceil L,0,0.7 -p "$PN
else
#BW="bowtie2 --np 0 --n-ceil L,0,0.5 -p "$PN
BW="bowtie2 --np 0 --n-ceil L,0,0.7 -p "$PN
fi
PE="pm_extract"
NM="--NM "${NSIZE}
ST="stfr"
SP="split_pm_read"

#First search
ASAM1=${OD}/${ID}/mapfile/${ID}_all.R1.sam
ASAM2=${OD}/${ID}/mapfile/${ID}_all.R2.sam
${BW} -x ${HD}/all_exon_intron_N${NLEN}.fasta -U ${F1} -S ${ASAM1}
${BW} -x ${HD}/all_exon_intron_N${NLEN}.fasta -U ${F2} -S ${ASAM2}
R1=${OD}/${ID}/mapfile/${ID}.R1.fastq
R2=${OD}/${ID}/mapfile/${ID}.R2.fastq

F1SIZE=`expr ${#F1}`
GZF=${F1:$F1SIZE-3:3}
if [ $GZF = ".gz" ]; then
TMPF1=${OD}/${ID}/mapfile/tmp_R1.fastq
TMPF2=${OD}/${ID}/mapfile/tmp_R2.fastq
zcat ${F1} > ${TMPF1}
zcat ${F2} > ${TMPF2}
${ST} -L ${TSIZE} ${ASAM1} ${TMPF1} ${OD}/${ID}/mapfile/${ID}
mv ${OD}/${ID}/mapfile/${ID}.fastq ${R1}
${ST} -L ${TSIZE} ${ASAM2} ${TMPF2} ${OD}/${ID}/mapfile/${ID}
mv ${OD}/${ID}/mapfile/${ID}.fastq ${R2}
rm ${TMPF1}
rm ${TMPF2}
else
${ST} -L ${TSIZE} ${ASAM1} ${F1} ${OD}/${ID}/mapfile/${ID}
mv ${OD}/${ID}/mapfile/${ID}.fastq ${R1}
${ST} -L ${TSIZE} ${ASAM2} ${F2} ${OD}/${ID}/mapfile/${ID}
mv ${OD}/${ID}/mapfile/${ID}.fastq ${R2}
fi
rm ${ASAM1}
rm ${ASAM2}


#Search perfect match
PSE1=${OD}/${ID}/mapfile/${ID}.all_exon.R1.pmap.sam
PSE2=${OD}/${ID}/mapfile/${ID}.all_exon.R2.pmap.sam
${BW} --score-min L,-1.0,0 -a -x ${HD}/all_exon_N${NLEN}.fasta -U ${R1} -S ${PSE1}
${BW} --score-min L,-1.0,0 -a -x ${HD}/all_exon_N${NLEN}.fasta -U ${R2} -S ${PSE2}
NPSE1=${OD}/${ID}/mapfile/${ID}.all_exon.R1.pmap.NM.sam
NPSE2=${OD}/${ID}/mapfile/${ID}.all_exon.R2.pmap.NM.sam
${PE} --MP 0 --NM 0 ${PSE1} 1.0 > ${NPSE1}
${PE} --MP 0 --NM 0 ${PSE2} 1.0 > ${NPSE2}
PSI1=${OD}/${ID}/mapfile/${ID}.all_intron.R1.pmap.sam
PSI2=${OD}/${ID}/mapfile/${ID}.all_intron.R2.pmap.sam
${BW} --score-min L,-1.0,0 -a -x ${HD}/all_intron_N${NLEN}.fasta -U ${R1} -S ${PSI1}
${BW} --score-min L,-1.0,0 -a -x ${HD}/all_intron_N${NLEN}.fasta -U ${R2} -S ${PSI2}
NPSI1=${OD}/${ID}/mapfile/${ID}.all_intron.R1.pmap.NM.sam
NPSI2=${OD}/${ID}/mapfile/${ID}.all_intron.R2.pmap.NM.sam
${PE} --MP 0 --NM 0 ${PSI1} 1.0 > ${NPSI1}
${PE} --MP 0 --NM 0 ${PSI2} 1.0 > ${NPSI2}
NPS1=${OD}/${ID}/mapfile/${ID}.all.R1.pmap.NM.sam
NPS2=${OD}/${ID}/mapfile/${ID}.all.R2.pmap.NM.sam
cat ${NPSE1} ${NPSI1} > ${NPS1}
cat ${NPSE2} ${NPSI2} > ${NPS2}
PR1=${OD}/${ID}/mapfile/${ID}.R1.pm
PR2=${OD}/${ID}/mapfile/${ID}.R2.pm
${ST} ${NPS1} ${R1} ${PR1}
${ST} ${NPS2} ${R2} ${PR2}
DR1=${OD}/${ID}/mapfile/${ID}.R1.diff.fastq
DR2=${OD}/${ID}/mapfile/${ID}.R2.diff.fastq
get_diff_fasta ${R1} ${PR1}.fastq > ${DR1}
get_diff_fasta ${R2} ${PR2}.fastq > ${DR2}

#Drop reads
ASE1=${OD}/${ID}/mapfile/${ID}.all_exon.R1.sam
ASE2=${OD}/${ID}/mapfile/${ID}.all_exon.R2.sam
ASI1=${OD}/${ID}/mapfile/${ID}.all_intron.R1.sam
ASI2=${OD}/${ID}/mapfile/${ID}.all_intron.R2.sam
${BW} -x ${HD}/all_exon_N${NLEN}.fasta -U ${DR1} -S ${ASE1}
${BW} -x ${HD}/all_exon_N${NLEN}.fasta -U ${DR2} -S ${ASE2}
${BW} -x ${HD}/all_intron_N${NLEN}.fasta -U ${DR1} -S ${ASI1}
${BW} -x ${HD}/all_intron_N${NLEN}.fasta -U ${DR2} -S ${ASI2}
DNPSE1=${OD}/${ID}/mapfile/${ID}.all_exon.R1.diff.pmap.NM.sam
DNPSE2=${OD}/${ID}/mapfile/${ID}.all_exon.R2.diff.pmap.NM.sam
DNPSI1=${OD}/${ID}/mapfile/${ID}.all_intron.R1.diff.pmap.NM.sam
DNPSI2=${OD}/${ID}/mapfile/${ID}.all_intron.R2.diff.pmap.NM.sam
${PE} --MP 0 ${NM} ${ASE1} ${CR} > ${DNPSE1}
${PE} --MP 0 ${NM} ${ASE2} ${CR} > ${DNPSE2}
${PE} --MP 2 ${NM} ${ASI1} ${CR} > ${DNPSI1}
${PE} --MP 2 ${NM} ${ASI2} ${CR} > ${DNPSI2}
cat ${DNPSE1} ${DNPSI1} > ${NPS1}
cat ${DNPSE2} ${DNPSI2} > ${NPS2}
DR12=${OD}/${ID}/mapfile/${ID}.R1.diff2
DR22=${OD}/${ID}/mapfile/${ID}.R2.diff2
${ST} ${NPS1} ${DR1} ${DR12}
${ST} ${NPS2} ${DR2} ${DR22}
mv ${DR12}.fastq ${DR1}
mv ${DR22}.fastq ${DR2}

#Map leavings
SE1=${OD}/${ID}/mapfile/${ID}.all_exon.R1.sam
SE2=${OD}/${ID}/mapfile/${ID}.all_exon.R2.sam
SI1=${OD}/${ID}/mapfile/${ID}.all_intron.R1.sam
SI2=${OD}/${ID}/mapfile/${ID}.all_intron.R2.sam
${BW} -a -x ${HD}/all_exon_N${NLEN}.fasta -U ${DR1} -S ${SE1}
${BW} -a -x ${HD}/all_exon_N${NLEN}.fasta -U ${DR2} -S ${SE2}
${BW} -a -x ${HD}/all_intron_N${NLEN}.fasta -U ${R1} -S ${SI1}
${BW} -a -x ${HD}/all_intron_N${NLEN}.fasta -U ${R2} -S ${SI2}
grep -v ^@ ${NPSE1} >> ${SE1}
grep -v ^@ ${NPSE2} >> ${SE2}

NSE1=${OD}/${ID}/mapfile/${ID}.all_exon.R1.NM.sam
NSE2=${OD}/${ID}/mapfile/${ID}.all_exon.R2.NM.sam
NSI1=${OD}/${ID}/mapfile/${ID}.all_intron.R1.NM.sam
NSI2=${OD}/${ID}/mapfile/${ID}.all_intron.R2.NM.sam
if [ $PN -ge 4 ]; then
${PE} --MP 0 ${NM} ${SE1} ${CR} > ${NSE1} &
${PE} --MP 0 ${NM} ${SE2} ${CR} > ${NSE2} &
${PE} --MP 2 ${NM} ${SI1} ${CR} > ${NSI1} &
${PE} --MP 2 ${NM} ${SI2} ${CR} > ${NSI2} &
wait
else
${PE} --MP 0 ${NM} ${SE1} ${CR} > ${NSE1}
${PE} --MP 0 ${NM} ${SE2} ${CR} > ${NSE2}
${PE} --MP 2 ${NM} ${SI1} ${CR} > ${NSI1}
${PE} --MP 2 ${NM} ${SI2} ${CR} > ${NSI2}
fi
rm ${SE1}
rm ${SE2}
rm ${SI1}
rm ${SI2}


if [ "$FLG_F" = "TRUE" ]; then
    ${SP} -f ${VALUE_F} -t ${TSIZE} -m ${MTSIZE} ${NSE1} ${NSE2} ${NSI1} ${NSI2} ${GE} ${ID} ${HD} ${NLEN} ${OD}
else
    ${SP} -t ${TSIZE} -m ${MTSIZE} ${NSE1} ${NSE2} ${NSI1} ${NSI2} ${GE} ${ID} ${HD} ${NLEN} ${OD}
fi

if [ $PN -ge 4 ]; then
    PN2=4
else
    PN2=${PN}
fi

split_shell ${OD}/${ID}/estimation.sh ${PN2}
for i in `seq 1 ${PN2}`
do
 j=`expr $i - 1`
 sh ${OD}/${ID}/estimation.sh.${j} &
done
wait
rm ${OD}/${ID}/estimation.sh.*

sh ${OD}/${ID}/pickup.sh
