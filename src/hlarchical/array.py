import os
import gzip
import pandas as pd
import subprocess
import torch
import shutil
import glob
from importlib import resources
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class Array():
    def __init__(self):
        self.HLA = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']

    def run_snp2hla(self, in_file='1958BC', ref_file='HM_CEU_REF', out_file='1958BC_European_SNP2HLA', snp2hla_dir=None, heap_size=2000, window_size=1000):
        if not snp2hla_dir:
            snp2hla_dir = f'{resources.files("hlarchical").parent.parent}/vendor/SNP2HLA/home'

        out_file = os.path.abspath(out_file)

        if os.path.exists(f'{in_file}.bed'):
            in_file = os.path.abspath(in_file)
        else:
            raise FileNotFoundError(f'Input file {in_file}.bed not found')

        if os.path.exists(f'{ref_file}.bed'):
            ref_file = os.path.abspath(ref_file)
        else:
            raise FileNotFoundError(f'Reference file {ref_file}.bed not found')

        # make a temporary working directory, not using the installation directory to avoid potential overwritten issues of multiple runs at the same time
        working_dir = f'{out_file}_working'
        shutil.copytree(snp2hla_dir, working_dir)
        os.chdir(working_dir)
        print(f'running SNP2HLA in {working_dir}', flush=True)

        cmd = f'tcsh SNP2HLA.csh {in_file} {ref_file} {out_file} plink {heap_size} {window_size}'
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
        #shutil.rmtree(working_dir)

    def run_hibag(self, in_file='1958BC', ref='European', out_file='1958BC_European_HIBAG', Renv='R4.5'):
        hibag_script = f'{resources.files("hlarchical").parent.parent}/vendor/HIBAG/hibag.R'
        ref_file = f'{ref}-HLA4-hg19.RData'
        if not os.path.exists(ref_file):
            cmd = f'wget https://hibag.s3.amazonaws.com/download/hlares_param/{ref_file}'
            subprocess.run(cmd, shell=True, check=True)
        cmd = f'conda run -n {Renv} Rscript {hibag_script} {in_file} {ref_file} {out_file}'
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)

    def run_deephla(self, mode='train', in_file='1958BC_Pan-Asian_REF', ref_file='Pan-Asian_REF', subset=[], model_json=None, model_dir='model', deephla_dir=None):
        if not deephla_dir:
            deephla_dir = f'{resources.files("hlarchical").parent.parent}/vendor/DEEP-HLA'

        if mode == 'train':
            if not os.path.exists(f'{in_file}.bgl.phased') or not os.path.exists(f'{ref_file}.bgl.phased'):
                print('Use hlarchical run-snp2hla first to get the phased bgl files')
                return

            if subset:
                region = subset.replace('-', ':').split(':')
                df = pd.read_table(f'{in_file}.bgl.phased', sep=' ', header=None)
                wh = []
                for n in range(df.shape[0]):
                    chrom = str(df.iloc[n, 0])
                    pos = int(df.iloc[n, 1])
                    if chrom == region[0] and pos >= int(region[1]) and pos <= int(region[2]):
                        wh.append(True)
                    else:
                        wh.append(False)
                df = df.loc[wh, ] 
                df.to_csv(f'{in_file}.bgl.phased', header=False, index=False, sep=' ')

            hla_json = f'{ref_file}.hla.json'
            if not os.path.exists(hla_json):
                print('Generating HLA info JSON file...')
                cmd = f'conda run -n DEEP-HLA python {deephla_dir}/make_hlainfo.py --ref {ref_file} --out {ref_file}.hla.json'
                print(cmd)
                subprocess.run(cmd, shell=True, check=True)

            if not model_json:
                model_json = f'{ref_file}.model.json'
            if os.path.exists(model_json):
                model_json = model_json.split('.model.json')[0]
                hla_json = hla_json.split('.hla.json')[0]
                cmd = f'conda run -n DEEP-HLA python {deephla_dir}/train.py --ref {ref_file} --sample {in_file} --model {model_json} --hla {hla_json} --model-dir {model_dir}'
                print(cmd)
                subprocess.run(cmd, shell=True, check=True)
            else:
                raise FileNotFoundError(f'{model_json} not found')

        elif mode == 'impute':
            model_json = model_json.split('.model.json')[0]
            hla_json = ref_file
            cmd = f'conda run -n DEEP-HLA python {deephla_dir}/impute.py --sample {in_file} --model {model_json} --hla  {hla_json} --model-dir {model_dir} --out {in_file}'
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)

    def get_hlarchical_table(self, in_file='', in_dir='HLA-HD', out_file='1000G_WGS_HLA-HD.txt', digit=4, from_tool='hla-hd'):
        header = ['SampleID', 'HLA', 'Allele1', 'Allele2']
        if in_file.endswith('.phased'):
            if from_tool == 'snp2hla':
                sep = ' '
                skiprows = 1
                col = 1
                in_header = 0
            elif from_tool == 'deep-hla':
                sep = '\t'
                skiprows = 0
                col = 0
                in_header = None
                samples = pd.read_table(in_file.replace('.deephla.phased', '.fam'), sep=' ', header=None).iloc[:, 1].tolist()

            df = pd.read_table(in_file, sep=sep, skiprows=skiprows, header=in_header)
            df = df.loc[df.iloc[:, col].str.startswith('HLA'), ]

            wh = [len(x.split('_')[2]) == digit for x in df.iloc[:, col]]
            df = df.loc[wh, ]

            L = []
            for n in range(col + 1, df.shape[1], 2):
                if from_tool == 'snp2hla':
                    sample_id = df.columns[n]
                elif from_tool == 'deep-hla':
                    sample_id = samples[int((n-col)/2)]
                allele1 = {}
                allele2 = {}
                for m in range(df.shape[0]):
                    allele = df.iloc[m, col]
                    if digit == 4:
                        allele = f'{allele[0:-2]}:{allele[-2:]}'
                    fields = allele.split('_')
                    k = '-'.join(fields[0:2])
                    if df.iloc[m, n] == 'P':
                        allele1.setdefault(k, [])
                        allele1[k].append(':'.join([k] + fields[2:]))
                    if df.iloc[m, n + 1] == 'P':
                        allele2.setdefault(k, [])
                        allele2[k].append(':'.join([k] + fields[2:]))

                for hla in self.HLA:
                    L.append([sample_id, hla, ','.join(allele1.get(hla, '.')), ','.join(allele2.get(hla, '.'))])

            df = pd.DataFrame(L)
            df.columns = header
            df.to_csv(out_file, header=True, index=False, sep='\t')
            print('Formatted output saved to', out_file)

        elif from_tool == 'hibag':
            df = pd.read_table(in_file, header=0, sep='\t')
            df_out = pd.DataFrame()
            df_out['SampleID'] = df['sample.id']
            df_out['HLA'] = df['HLA']
            if digit == 2:
                df_out['Allele1'] = df['HLA'] + ':' + df['allele1'].str.split(':').str[0]
                df_out['Allele2'] = df['HLA'] + ':' + df['allele2'].str.split(':').str[0]
            elif digit == 4:
                df_out['Allele1'] = df['HLA'] + ':' + df['allele1']
                df_out['Allele2'] = df['HLA'] + ':' + df['allele2']
            df_out.sort_values(by=['SampleID', 'HLA'], inplace=True)
            df_out.to_csv(out_file, sep='\t', index=False, header=True)
            print('Formatted output saved to', out_file)

        elif from_tool == 'hla-hd':
            D = {}
            fs = glob.glob(f'{in_dir}/**/result/*.est.txt', recursive=True)
            for f in sorted(fs):
                sample = f.split('/')[-3]
                hla = 'HLA-' + f.split('/')[-1].split('_')[-1].split('.est.txt')[0]
                D.setdefault(sample, {})
                D[sample].setdefault(hla, ['.', '.'])
                df = pd.read_table(f, comment='#', header=None)
                if df.shape[0] > 0 and df.shape[1] > 1:
                    item1 = df.iloc[0, 0].split(',')
                    item2 = df.iloc[0, 1].split(',')
                    item1x = set([x.replace('*', ':').split(':')[0:int(digit/2) + 1] for x in item1])
                    item2x = set([x.replace('*', ':').split(':')[0:int(digit/2) + 1] for x in item2])
                    if len(item1x) > 1 or len(item2x) > 1:
                        print(f'Warning: multiple alleles found for sample {sample} and HLA {hla} in file {f}. Only the first allele will be used. Allele1: {item1}, Allele2: {item2}', flush=True)

                    allele1 = item1[0]
                    allele2 = item2[0]
                    allele1 = allele1.replace('*', ':')
                    allele2 = allele2.replace('*', ':')
        
                    a1 = allele1.split(':')
                    a2 = allele2.split(':')
                    allele1 = ':'.join(a1[0:int(digit/2) + 1])
                    allele2 = ':'.join(a2[0:int(digit/2) + 1])

                    if allele2 == '-':
                        allele2 = allele1

                    if len(allele1.split(':')) < int(digit/2) + 1:
                        allele1 = '.'
                    if len(allele2.split(':')) < int(digit/2) + 1:
                        allele2 = '.'
                    D[sample][hla] = [allele1, allele2]
        
            L = []
            for sample in sorted(D):
                for hla in self.HLA:
                    allele1, allele2 = ['.', '.']
                    if hla in D[sample]:
                        allele1, allele2 = D[sample][hla]
                    L.append([sample, hla, allele1, allele2])
            df = pd.DataFrame(L)
            df.columns = header
            df.to_csv(out_file, header=True, index=False, sep='\t')
            print('Formatted output saved to', out_file)

        elif from_tool == 'xhla':
            D = {}
            fs = glob.glob(f'{in_dir}/**/*-hla.json', recursive=True)
            for f in fs:
                sample = f.split('/')[-1].split('-hla.json')[0].split('report-')[-1]
                df = pd.read_json(f)
                if df.shape[0]:
                    L = df.loc['alleles', 'hla']
                    for n in range(0, len(L), 2):
                        hla = 'HLA-' + L[n].split('*')[0]
                        D.setdefault(sample, {})
                        D[sample][hla] = ['.', '.']
                        allele1 = 'HLA-' + L[n]
                        allele2 = 'HLA-' + L[n + 1]
                        allele1 = allele1.replace('*', ':')
                        allele2 = allele2.replace('*', ':')
        
                        a1 = allele1.split(':')
                        a2 = allele2.split(':')
        
                        allele1 = ':'.join(a1[0:int(digit/2) + 1])
                        allele2 = ':'.join(a2[0:int(digit/2) + 1])
                        if len(allele1.split(':')) < int(digit/2) + 1:
                            allele1 = '.'
                        if len(allele2.split(':')) < int(digit/2) + 1:
                            allele2 = '.'
                        D[sample][hla] = [allele1, allele2]
        
            L = []
            header = ['SampleID', 'HLA', 'Allele1', 'Allele2']
            for sample in sorted(D):
                for hla in self.HLA:
                    allele1, allele2 = ['.', '.']
                    if hla in D[sample]:
                        allele1, allele2 = D[sample][hla]
                    L.append([sample, hla, allele1, allele2])

            df = pd.DataFrame(L)
            df.columns = header
            df.to_csv(out_file, header=True, index=False, sep='\t')
            print('Formatted output saved to', out_file)

        elif from_tool == 'opti-type':
            D = {}
            fs = glob.glob(f'{in_dir}/**/*_result.tsv', recursive=True)
            for f in sorted(fs):
                sample = f.split('/')[-1].split('_result')[0]
                df = pd.read_table(f, header=0)
                if df.shape[0]:
                    for n in range(1, df.shape[1] - 2, 2):
                        hla = 'HLA-' + df.columns[n][0:-1]
                        D.setdefault(sample, {})
                        D[sample][hla] = ['.', '.']
                        allele1 = 'HLA-' + df.iloc[0, n]
                        allele2 = 'HLA-' + df.iloc[0, n + 1]
                        allele1 = allele1.replace('*', ':')
                        allele2 = allele2.replace('*', ':')
        
                        a1 = allele1.split(':')
                        a2 = allele2.split(':')
        
                        allele1 = ':'.join(a1[0:int(digit/2) + 1])
                        allele2 = ':'.join(a2[0:int(digit/2) + 1])
                        if len(allele1.split(':')) < int(digit/2) + 1:
                            allele1 = '.'
                        if len(allele2.split(':')) < int(digit/2) + 1:
                            allele2 = '.'
                        D[sample][hla] = [allele1, allele2]
        
            L = []
            for sample in sorted(D):
                for hla in self.HLA:
                    allele1, allele2 = ['.', '.']
                    if hla in D[sample]:
                        allele1, allele2 = D[sample][hla]
                    L.append([sample, hla, allele1, allele2])
            df = pd.DataFrame(L)
            df.columns = header
            df.to_csv(out_file, header=True, index=False, sep='\t')
            print('Formatted output saved to', out_file)

        elif from_tool == 'hla-typing':
            # internal use only
            df = pd.read_excel(in_file, dtype=str, skiprows=2)
            header = ['SampleID', 'Race', 'Gender', 'Disease', 'HLA', 'Allele1', 'Allele2']
        
            HLAidx = {}
            HLAidx['HLA-A'] = list(df.columns).index('IMGT/A')
            HLAidx['HLA-B'] = list(df.columns).index('IMGT/B')
            HLAidx['HLA-C'] = list(df.columns).index('IMGT/C')
            HLAidx['HLA-DPA1'] = list(df.columns).index('IMGT/DPA1')
            HLAidx['HLA-DPB1'] = list(df.columns).index('IMGT/DPB1')
            HLAidx['HLA-DQA1'] = list(df.columns).index('IMGT/DQA1')
            HLAidx['HLA-DQB1'] = list(df.columns).index('IMGT/DQB1')
            HLAidx['HLA-DRB1'] = list(df.columns).index('IMGT/DRB1')
        
            L = []
            for n in range(0, df.shape[0], 2):
                sample_id = df.iloc[n, 0]
                race = df.iloc[n, 1]
                gender = df.iloc[n, 2].lower().strip()
                disease = df.iloc[n, 3].lower().strip()
        
                for k in HLAidx.keys():
                    hla = k
                    allele1 = df.iloc[n, HLAidx[k]]
                    allele2 = df.iloc[n + 1, HLAidx[k]]
                    if str(allele1) == 'nan':
                        allele1 = '.'
                    if str(allele2) == 'nan':
                        allele2 = '.'
                    if allele1 not in ['.', 'X']:
                        allele1 = f'{hla}:{allele1}'
                    if allele2 not in ['.', 'X']:
                        allele2 = f'{hla}:{allele2}'
                    L.append([sample_id, race, gender, disease, hla, allele1, allele2])

            df = pd.DataFrame(L)
            df.columns = header
            df.to_csv(out_file, header=True, index=False, sep='\t')

    def merge_hlarchical_tables(self, ancestry_file='GAP_OMNI_GDA.txt', out_file='HLA_OMNI_GDA.txt', digits=[2, 4],
                      tools=['SNP2HLA', 'HIBAG', 'hlarchicalMLP', 'hlarchicalCNN'], Ancestry=['European', 'Asian', 'African', 'Hispanic', 'MA'], Array=['GDA', 'OMNI'], ensemble=['HIBAG', 'SNP2HLA']):
        D = {}
        A = {}
        for digit in digits:
            D.setdefault(digit, {})
            for tool in tools:
                D[digit].setdefault(tool, {})
                for ancestry in Ancestry:
                    D[digit][tool].setdefault(ancestry, {})
                    for array in Array:
                        in_file = f'{array}_{ancestry}_{tool}_digit{digit}.txt'
                        if os.path.exists(in_file):
                            df = pd.read_csv(in_file, sep='\t', header=0)

                            # assuming sample_id contains FID, if all SampleID starts with '\d-'
                            sample_id_with_fid = df['SampleID'].str.contains(r'^\d+-', na=False).all()
                            if sample_id_with_fid:
                                print(f'FID is being excluded from SampleID in {in_file}', flush=True)

                            for i, row in df.iterrows():
                                sample_id = row['SampleID']
                                if sample_id_with_fid:
                                    sample_id = '-'.join(sample_id.split('-')[1:])
                                if in_file.find('hlarchical') != -1:
                                    sample_id = '_'.join(sample_id.split('_')[1:])
                                A[sample_id] = array
                                hla = row['HLA']
                                allele1 = row['Allele1']
                                allele2 = row['Allele2']
                                k = (sample_id, hla)
                                D[digit][tool][ancestry][k] = (allele1, allele2)
    
        df = pd.read_table(ancestry_file, header=0, sep='\t')
        Ls = []
        cols = ['SampleID', 'Superpopulation', 'Population', 'Array', 'HLA']

        for n in range(df.shape[0]):
            sample_id = df['SampleID'].iloc[n]
            sample_name = df['SampleName'].iloc[n]
            superpopulation = df['Superpopulation'].iloc[n]
            population = df['Population'].iloc[n]
            array = A.get(sample_name, '.')
            if superpopulation in ['EAS', 'SAS']:
                ancestry = 'Asian'
            elif superpopulation in ['EUR']:
                ancestry = 'European'
            elif superpopulation in ['AFR']:
                ancestry = 'African'
            elif superpopulation in ['AMR']:
                ancestry = 'Hispanic'
            else:
                ancestry = 'European'
    
            for hla in self.HLA:
                L = [sample_id, superpopulation, population, array, hla]
                for digit in digits:
                    for tool in tools:
                        if n == 0 and hla == self.HLA[0]:
                            cols += [f'Allele1_{tool}_digit{digit}', f'Allele2_{tool}_digit{digit}']
    
                        if tool == 'SNP2HLA':
                            if ancestry in ['Asian', 'European']:
                                pass
                            else:
                                ancestry = 'European'
                        if tool.find('hlarchical') != -1:
                            ancestry = 'MA'
    
                        k = (sample_name, hla)
                        allele1, allele2 = ['.', '.']
                        if ancestry in D[digit][tool]:
                            if k in D[digit][tool][ancestry]:
                                allele1, allele2 = D[digit][tool][ancestry][k]
                        L += [allele1, allele2]
                Ls.append(L)
        df = pd.DataFrame(Ls)
        df.columns = cols

        if ensemble:
            idx = df.columns.tolist().index('HLA')
            for digit in digits:
                for alelle in ['Allele1', 'Allele2']:
                    idx += 1
                    D = {}
                    for tool in tools: 
                        col = f'{alelle}_{tool}_digit{digit}'
                        D[col] = df[col].tolist()

                    L = []
                    for n in range(df.shape[0]):
                        value = '.'
                        for tool in ensemble:
                            col = f'{alelle}_{tool}_digit{digit}'
                            if D[col][n] not in ['.', 'X']:
                                value = D[col][n]
                                break
                        L.append(value)
                    df.insert(loc=idx, column=f'{alelle}_ensemble_digit{digit}', value=L)
        df.to_csv(out_file, sep='\t', index=False)

    def hla_typing_genotyping_scoring(self, in_file):
        out_file = in_file.replace('.txt', '_score.txt')
        out_file_overall = in_file.replace('.txt', '_score_overall.txt')
        out_file_ancestry = in_file.replace('.txt', '_score_ancestry.txt')
        df = pd.read_table(in_file, header=0, sep='\t')
        Ls = []
        cols = df.columns.tolist()
        for n in range(df.shape[0]):
            L = df.iloc[n, :].tolist()
            for m in range(11, df.shape[1], 2):
                digit = int(df.columns[m][-1])
                typ_a1 = df['Allele1_typing'].iloc[n]
                typ_a2 = df['Allele2_typing'].iloc[n]
                if typ_a1 not in ['.', 'X']:
                    typ_a1 = ':'.join(typ_a1.split(':')[0:int(digit/2)+1])
                    typ_a2 = ':'.join(typ_a2.split(':')[0:int(digit/2)+1])
                geno_a1 = df.iloc[n, m]
                geno_a2 = df.iloc[n, m + 1]
                t, g = self._cal_score(typ_a1, typ_a2, geno_a1, geno_a2)
                L.append(f'{g}/{t}')
                if n == 0:
                    cols.append(df.columns[m].replace('Allele1', 'score'))
            Ls.append(L)
        df = pd.DataFrame(Ls) 
        df.columns = cols
        df.to_csv(out_file, sep='\t', index=False, header=True)

        score_idx = []
        for n in range(df.shape[1]):
            if df.columns[n].startswith('score_'):
                score_idx.append(n)

        # overall score for each HLA	
        L = []
        for hla in df['HLA'].unique():
            df2 = df.loc[df['HLA'] == hla]
            for idx in score_idx:
                score_sum = df2.iloc[:, idx].apply(lambda x: int(x.split('/')[0])).sum()
                total_sum = df2.iloc[:, idx].apply(lambda x: int(x.split('/')[1])).sum()
                L.append([hla, df.columns[idx].replace('score_', ''), f'{score_sum/total_sum:4f}', score_sum, total_sum])
        df_overall = pd.DataFrame(L)
        df_overall.columns = ['HLA', 'method', 'score', 'genotyping', 'typing']
        df_overall.to_csv(out_file_overall, sep='\t', index=False, header=True)

        # per ancestry score for each HLA
        L = []
        for hla in df['HLA'].unique():
            df2 = df.loc[df['HLA'] == hla]
            for ancestry in df2['Superpopulation'].unique():
                df3 = df2.loc[df2['Superpopulation'] == ancestry]
                for idx in score_idx:
                    score_sum = df3.iloc[:, idx].apply(lambda x: int(x.split('/')[0])).sum()
                    total_sum = df3.iloc[:, idx].apply(lambda x: int(x.split('/')[1])).sum()
                    L.append([hla, ancestry, df.columns[idx].replace('score_', ''), f'{score_sum/total_sum:4f}', score_sum, total_sum])
        df_ancestry = pd.DataFrame(L)
        df_ancestry.columns = ['HLA', 'ancestry', 'method', 'score', 'genotyping', 'typing']
        df_ancestry.to_csv(out_file_ancestry, sep='\t', index=False, header=True)

    def bar_plot_score(self, in_file, digits=[2, 4], methods=['SNP2HLA', 'HIBAG', 'hlarchicalMLP', 'hlarchicalCNN']):
        df = pd.read_table(in_file, header=0, sep='\t')
        for digit in digits:
            for method in methods:
                df2 = df.loc[df['method'] == f'{method}_digit{digit}']
                df3 = df2.loc[~df2['HLA'].isin(['HLA-DPA1'])]
                plt.figure()
                if 'ancestry' in df2.columns:
                    ax = sns.barplot(x='HLA', y='score', hue='ancestry', data=df2)
                    plt.legend(bbox_to_anchor=(0.5, 0.995), loc='upper center', ncols=5)
                else:
                    ax = sns.barplot(x='HLA', y='score', data=df2)

                score_avg = df2['genotyping'].sum() / df2['typing'].sum()
                score_avg2 = df3['genotyping'].sum() / df3['typing'].sum()
                txt = f'Average accuracy: {score_avg:.4f}'
                txt2 = f'Average accuracy excluding HLA-DPA1: {score_avg2:.4f}'
                print([in_file, digit, method, txt], flush=True)
                print([in_file, digit, method, txt2], flush=True)
                ax.text(0.98, 0.07, txt, ha='right', va='bottom', transform=ax.transAxes, fontsize=10)
                ax.text(0.98, 0.02, txt2, ha='right', va='bottom', transform=ax.transAxes, fontsize=10, weight='bold')

                ax.set_ylim(0, 1.2)
                ax.set_title(f'Accuracy at {digit}-digit resolution using {method}')
                ax.set_ylabel('Score')
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=90)
                plt.tight_layout()
                out_file = in_file.replace('.txt', f'_{method}_digit{digit}_barplot.pdf')
                out_file2 = in_file.replace('.txt', f'_{method}_digit{digit}_barplot.png')
                plt.savefig(out_file)
                plt.savefig(out_file2)
                plt.close()
                print('Bar plot saved to', out_file)

    def _cal_score(self, ta1, ta2, ga1, ga2):
        t = 0
        g = 0
        for x in [ta1, ta2]:
            if x not in ['.', 'X'] and x.find('--') == -1:
                t += 1
    
        for x in [ga1, ga2]:
            if x not in ['.', 'X']:
                if x in [ta1, ta2]:
                    g += 1
        g = min(t, g)
        if ga1 == ga2 and ta1 != ta2:
            g = min(g, 1)
        return t, g

if __name__ == "__main__":
    pass
