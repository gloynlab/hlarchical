from .utils import *

class Summary():
    def __init__(self):
        self.HLA = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']

    def get_hlarchical_table(self, in_file='', in_dir='HLA-HD', out_file='1000G_WGS_HLA-HD.txt', digit=4, from_tool='hla-hd', fix_sample_name_by_fam=None):
        D = {}
        if fix_sample_name_by_fam is not None and os.path.exists(fix_sample_name_by_fam):
            D = self._fix_sample_name_by_fam(fix_sample_name_by_fam)

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
                    L.append([D.get(sample_id, sample_id), hla, ','.join(allele1.get(hla, '.')), ','.join(allele2.get(hla, '.'))])

            df = pd.DataFrame(L)
            df.columns = header
            df.to_csv(out_file, header=True, index=False, sep='\t')
            print('Formatted output saved to', out_file)

        elif from_tool == 'hibag':
            df = pd.read_table(in_file, header=0, sep='\t')
            df_out = pd.DataFrame()
            df_out['SampleID'] = [D.get(k, k) for k in df['sample.id']]
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

        elif from_tool == 'michigan-server':
            D = {}
            with gzip.open(in_file, 'rt') as f:
                for line in f:
                    line = line.strip()
                    if line[0] == '#':
                        if line.find('#CHROM') == 0:
                            header = line.split('\t')
                    else:
                        fields = line.split('\t')
                        if fields[2].find('HLA') == 0:
                            hla = fields[2].replace('_', '-').replace('*', ':')
                            gene = hla.split(':')[0]
                            if (len(hla.split(':')) - 1) * 2 == digit:
                                r2 = float(fields[7].split('R2=')[-1].split(';')[0])
                                for n in range(9, len(fields)):
                                    sample = header[n]
                                    D.setdefault(sample, {})
                                    D[sample].setdefault(gene, {})
                                    fds = fields[n].split(':')
                                    gt = fds[0]
                                    ds = float(fds[-1])
                                    if gt.find('1') != -1:
                                        D[sample][gene].setdefault(gt, [])
                                        D[sample][gene][gt].append([hla, r2, ds])

            for sample in D:
                for gene in D[sample]:
                    for gt in D[sample][gene]:
                        D[sample][gene][gt] = sorted(D[sample][gene][gt], key=lambda x:x[1], reverse=True)

            L = []
            for sample in D:
                for gene in self.HLA:
                    allele1 = '.'
                    allele2 = '.'
                    for gt in ['1|1', '0|1', '1|0', '1/1', '0/1', '1/0']:
                        if gt in D[sample][gene]:
                            if gt in ['1|1', '1/1']:
                                allele1 = D[sample][gene][gt][0][0]
                                allele2 = D[sample][gene][gt][0][0]
                            elif gt in ['0|1', '0/1']:
                                allele1 = D[sample][gene][gt][0][0]
                            elif gt in ['1|0', '1/0']:
                                allele2 = D[sample][gene][gt][0][0]
                    L.append([sample, gene, allele1, allele2])
            df = pd.DataFrame(L, columns = ['SampleID', 'HLA', 'Allele1', 'Allele2'])
            df.to_csv(out_file, header=True, index=False, sep='\t')

        elif from_tool == 'hla-typing-stanford':
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
        elif from_tool == 'hla-typing-iidp':
            df = pd.read_table(in_file, header=0, low_memory=False)
            L = []
            for n in range(df.shape[0]):
                rrid = df['RRID'].iloc[n].split(':')[-1]
                gender = df['gender'].iloc[n]
                race = df['race'].iloc[n]
                ethnicity = df['ethnicity'].iloc[n]
                age = df['age'].iloc[n]
                height = df['height'].iloc[n]
                weight = df['weight'].iloc[n]
                bmi = df['bmi'].iloc[n]

                for i, k in enumerate([gender, race, ethnicity]):
                    try:
                        k = int(k)
                        if i == 0:
                            if k == 1:
                                gender = 'male'
                            elif k == 2:
                                gender = 'female'
                    except:
                        k = '.'
                for k in [age, height, weight, bmi]:
                    try:
                        k = float(k)
                    except:
                        k = '.'

                hla_a_a1 = self._fix_hla_allele_iidp(df['hla_a'].iloc[n], self.HLA[0])
                hla_a_a2 = self._fix_hla_allele_iidp(df['hla_a_1'].iloc[n], self.HLA[0])
                L.append([rrid, gender, race, ethnicity, age, height, weight, bmi, self.HLA[0], hla_a_a1, hla_a_a2])

                hla_b_a1 = self._fix_hla_allele_iidp(df['hla_b'].iloc[n], self.HLA[1])
                hla_b_a2 = self._fix_hla_allele_iidp(df['hla_b_1'].iloc[n], self.HLA[1])
                L.append([rrid, gender, race, ethnicity, age, height, weight, bmi, self.HLA[1], hla_b_a1, hla_b_a2])

                hla_c_a1 = self._fix_hla_allele_iidp(df['hla_c'].iloc[n], self.HLA[2])
                hla_c_a2 = self._fix_hla_allele_iidp(df['hla_c_1'].iloc[n], self.HLA[2])
                L.append([rrid, gender, race, ethnicity, age, height, weight, bmi, self.HLA[2], hla_c_a1, hla_c_a2])

                hla_dpa1_a1 = '.'
                hla_dpa1_a2 = '.'
                L.append([rrid, gender, race, ethnicity, age, height, weight, bmi, self.HLA[3], '.', '.'])

                hla_dpb1_a1 = '.'
                hla_dpb1_a2 = '.'
                L.append([rrid, gender, race, ethnicity, age, height, weight, bmi, self.HLA[4], '.', '.'])

                hla_dqa1_a1 = '.'
                hla_dqa1_a2 = '.'
                L.append([rrid, gender, race, ethnicity, age, height, weight, bmi, self.HLA[5], '.', '.'])

                hla_dqb1_a1 = self._fix_hla_allele_iidp(df['hla_dq'].iloc[n], self.HLA[6])
                hla_dqb1_a2 = self._fix_hla_allele_iidp(df['hla_dq_1'].iloc[n], self.HLA[6])
                L.append([rrid, gender, race, ethnicity, age, height, weight, bmi, self.HLA[6], hla_dqb1_a1, hla_dqb1_a2])

                hla_drb1_a1 = self._fix_hla_allele_iidp(df['hla_dr'].iloc[n], self.HLA[7])
                hla_drb1_a2 = self._fix_hla_allele_iidp(df['hla_dr_1'].iloc[n], self.HLA[7])
                L.append([rrid, gender, race, ethnicity, age, height, weight, bmi, self.HLA[7], hla_drb1_a1, hla_drb1_a2])

            df = pd.DataFrame(L)
            df.columns = ['RRID', 'Gender', 'Race', 'Ethnicity', 'Age', 'Height', 'Weight', 'BMI', 'HLA', 'Allele1', 'Allele2']
            df.to_csv(out_file, header=True, index=False, sep='\t')
        else:
            raise ValueError(f'Unsupported tool: {from_tool}.')

    def _fix_sample_name_by_fam(self, fam_file):
        D1 = {}
        D2 = {}
        df = pd.read_table(fam_file, header=None, sep=' ')
        for n in range(df.shape[0]):
            fid = df.iloc[n, 0]
            iid = df.iloc[n, 1]
            v = f'{fid}_{iid}'
            k1 = iid
            k2 = f'{fid}-{iid}'
            D1.setdefault(k1, [])
            D1.setdefault(k2, [])
            D1[k1].append(v)
            D1[k2].append(v)
        for k in D1:
            if len(D1[k]) == 1:
                D2[k] = D1[k][0]
            else:
                D2[k] = D1[k][0]
                for n in range(1, len(D1[k])):
                    k2 = f'{k}.{n+1}'
                    D2[k2] = D1[k][n]
        return D2

    def _fix_hla_allele_iidp(self, allele, hla):
        allele = str(allele)
        if allele.find(':') != -1:
            allele = allele.split(':')[0]
        if allele.startswith('A') or allele.startswith('B') or allele.startswith('C'):
            allele = allele[1:]
        try:
            a = int(allele)
            if a == 0:
                allele = '.'
            elif len(allele) == 1:
                allele = f'0{allele}'
            elif len(allele) == 4:
                allele = allele[0:2]
        except:
            allele = '.'
        if allele not in ['.']:
            allele = f'{hla}:{allele}'
        return allele

    def merge_hlarchical_tables(self, out_file='HLA_OMNI-GDA_withGAP.txt', digits=[2, 4], tools=['SNP2HLA', 'HIBAG', 'hlarchical'],
                                Ancestry=['European', 'Asian', 'African', 'Hispanic', 'MA'], ancestry_file='GAP_OMNI-GDA.txt'):
        D = {}
        for digit in digits:
            D.setdefault(digit, {})
            for tool in tools:
                D[digit].setdefault(tool, {})
                for ancestry in Ancestry:
                    D[digit][tool].setdefault(ancestry, {})
                    in_file = f'HLA_OMNI-GDA_{ancestry}_{tool}_digit{digit}.txt'
                    if os.path.exists(in_file):
                        df = pd.read_csv(in_file, sep='\t', header=0)
                        for n in range(df.shape[0]):
                            sample_id = df['SampleID'].iloc[n]
                            hla = df['HLA'].iloc[n]
                            allele1 = df['Allele1'].iloc[n]
                            allele2 = df['Allele2'].iloc[n]
                            k = (sample_id, hla)
                            D[digit][tool][ancestry][k] = (allele1, allele2)

        SA = {}
        if ancestry_file is not None and os.path.exists(ancestry_file):
            df = pd.read_table(ancestry_file, header=0, sep='\t')
            for n in range(df.shape[0]):
                sample_id = df['SampleID'].iloc[n]
                sample_name = df['SampleName'].iloc[n]
                array = df['Array'].iloc[n]
                source = df['Source'].iloc[n]
                sex = df['Sex'].iloc[n]
                batch = df['Batch'].iloc[n]
                project = df['Project'].iloc[n]
                rrid = df['RRID'].iloc[n]
                qc = df['QC'].iloc[n]
                extra = df['Extra'].iloc[n]
                superpopulation = df['Superpopulation'].iloc[n]
                population = df['Population'].iloc[n]
                SA[sample_id] = [sample_name, array, source, sex, batch, project, rrid, qc, extra, superpopulation, population]
        else:
            print('Ancestry not used', flush=True)

        Ls = []
        cols = ['SampleID', 'SampleName', 'Array', 'Source', 'Sex', 'Batch', 'Project', 'RRID', 'QC', 'Extra', 'Superpopulation', 'Population', 'HLA']
        sa = sorted(SA.items(), key=lambda x: [x[1][0], x[0]])
        sample_ids = [item[0] for item in sa]
        for sample_id in sample_ids:
            v = SA.get(sample_id, ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'])
            sample_name = v[0]
            array = v[1]
            source = v[2]
            sex = v[3]
            batch = v[4]
            project = v[5]
            rrid = v[6]
            qc = v[7]
            extra = v[8]
            superpopulation = v[9]
            population = v[10]

            for hla in self.HLA:
                L = [sample_id, sample_name, array, source, sex, batch, project, rrid, qc, extra, superpopulation, population, hla]
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
                        elif tool.lower().find('michigan') != -1:
                            ancestry = 'MA'
                        elif tool.find('hlarchical') != -1:
                            ancestry = 'MA'
                        else:
                            raise ValueError(f'Unsupported tool: {tool}.')

                        k = (sample_id, hla)
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
                ax.set_title(f'Average accuracy:{score_avg:.4f}\n({method} {digit}digit)')
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
