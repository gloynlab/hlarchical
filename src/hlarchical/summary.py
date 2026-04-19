from .utils import *

class Summary():
    def __init__(self):
        self.HLA = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']

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
                samples = pd.read_table(in_file.replace('.deephla.phased', '_SNP2HLA.fam'), sep=' ', header=None).iloc[:, 1].tolist()

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
                    item1x = set([':'.join(x.replace('*', ':').split(':')[0:int(digit/2) + 1]) for x in item1])
                    item2x = set([':'.join(x.replace('*', ':').split(':')[0:int(digit/2) + 1]) for x in item2])
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
        else:
            raise ValueError(f'Unsupported tool: {from_tool}. Supported tools are: snp2hla, deep-hla, hibag, hla-hd, xhla, opti-type, hla-typing.')

    def merge_hlarchical_tables(self, out_file='HLA_OMNI_GDA_GAP.txt', digits=[2, 4], tools=['SNP2HLA', 'HIBAG', 'hlarchicalMLPwithoutAncestry', 'hlarchicalMLPwithAncestry'],
                                Ancestry=['European', 'Asian', 'African', 'Hispanic', 'MA'], Array=['GDA', 'OMNI'], ancestry_file='GAP_OMNI_GDA.txt'):
        D = {}
        SA = {}
        SA['SampleID'] = {}
        SA['SampleName'] = {}
        SA['Superpopulation'] = {}
        SA['Population'] = {}
        SA['Array'] = {}

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

                            # assuming sample_id contains FID, if all SampleID starts with '\d+-' or '\d+_'
                            sample_id_with_fid_dash = df['SampleID'].str.contains(r'^\d+-', na=False).all()
                            sample_id_with_fid_underscore = df['SampleID'].str.contains(r'^\d+_', na=False).all()
                            if sample_id_with_fid_dash or sample_id_with_fid_underscore:
                                print(f'FID is being excluded from SampleID in {in_file}', flush=True)

                            for i, row in df.iterrows():
                                sample_id = row['SampleID']
                                if sample_id_with_fid_dash:
                                    sample_id = '-'.join(sample_id.split('-')[1:])
                                elif sample_id_with_fid_underscore:
                                    sample_id = '_'.join(sample_id.split('_')[1:])
                                SA['SampleID'][sample_id] = sample_id
                                SA['Array'][sample_id] = array
                                hla = row['HLA']
                                allele1 = row['Allele1']
                                allele2 = row['Allele2']
                                k = (sample_id, hla)
                                D[digit][tool][ancestry][k] = (allele1, allele2)
    
        Ls = []
        if ancestry_file is not None and os.path.exists(ancestry_file):
            df = pd.read_table(ancestry_file, header=0, sep='\t')
            SA['SampleName'] = dict(zip(df['SampleID'], df['SampleName']))
            SA['Superpopulation'] = dict(zip(df['SampleID'], df['Superpopulation']))
            SA['Population'] = dict(zip(df['SampleID'], df['Population']))
        else:
            print('Ancestry not used', flush=True)

        cols = ['SampleID', 'Superpopulation', 'Population', 'Array', 'HLA']
        sample_ids = sorted(SA['SampleID'])
        for sample_id in sample_ids:
            sample_name = SA['SampleName'].get(sample_id, sample_id)
            superpopulation = SA['Superpopulation'].get(sample_id, '.')
            population = SA['Population'].get(sample_id, '.')
            array = SA['Array'].get(sample_id, '.')

            for hla in self.HLA:
                L = [sample_id, superpopulation, population, array, hla]
                for digit in digits:
                    for tool in tools:
                        if sample_id == sample_ids[0] and hla == self.HLA[0]:
                            cols += [f'Allele1_{tool}_digit{digit}', f'Allele2_{tool}_digit{digit}']
    
                        if tool in ['SNP2HLA', 'DEEPHLA']:
                            if superpopulation in ['EUR']:
                                ancestry = 'European'
                            elif superpopulation in ['EAS', 'SAS']:
                                ancestry = 'Asian'
                            else:
                                ancestry = 'European'
                        elif tool in ['HIBAG']:
                            if superpopulation in ['EUR']:
                                ancestry = 'European'
                            elif superpopulation in ['EAS', 'SAS']:
                                ancestry = 'Asian'
                            elif superpopulation in ['AFR']:
                                ancestry = 'African'
                            elif superpopulation in ['AMR']:
                                ancestry = 'Hispanic'
                            else:
                                ancestry = 'European'
                        elif tool.find('hlarchical') != -1:
                            ancestry = 'MA'
                        else:
                            raise ValueError(f'Unsupported tool: {tool}.')

                        k = (sample_name, hla)
                        allele1, allele2 = ['.', '.']
                        if ancestry in D[digit][tool]:
                            if k in D[digit][tool][ancestry]:
                                allele1, allele2 = D[digit][tool][ancestry][k]
                        L += [allele1, allele2]
                Ls.append(L)
        df = pd.DataFrame(Ls)
        df.columns = cols
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

    def bar_plot_score(self, in_file, digits=[2, 4], methods=['SNP2HLA', 'HIBAG', 'hlarchicalMLPwithAncestry', 'hlarchicalMLPwithoutAncestry', 'hlarchicalMLPm12'], cmap='colorblind'):
        df = pd.read_table(in_file, header=0, sep='\t')
        for digit in digits:
            for method in methods:
                df2 = df.loc[df['method'] == f'{method}_digit{digit}']
                df3 = df2.loc[~df2['HLA'].isin(['HLA-DPA1'])]
                plt.figure()
                if 'ancestry' in df2.columns:
                    hue_order = sorted(df2['ancestry'].unique()) if 'ancestry' in df2.columns else None
                    ax = sns.barplot(x='HLA', y='score', hue='ancestry', data=df2, palette=cmap, hue_order=hue_order)
                    plt.legend(bbox_to_anchor=(0.5, 0.995), loc='upper center', ncols=5)
                else:
                    ax = sns.barplot(x='HLA', y='score', data=df2)

                score_avg = df2['genotyping'].sum() / df2['typing'].sum()
                score_avg2 = df3['genotyping'].sum() / df3['typing'].sum()
                txt = f'Average accuracy: {score_avg:.4f}'
                txt2 = f'Average accuracy excluding HLA-DPA1: {score_avg2:.4f}'
                print([in_file, digit, method, txt], flush=True)
                print([in_file, digit, method, txt2], flush=True)

                ax.set_ylim(0, 1.2)
                ax.set_title(f'Average accuracy:{score_avg:.4f} ({method} {digit}digit)')
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
