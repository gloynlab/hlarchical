from .utils import *

class Processor:
    def __init__(self, ref_phased='1000G_REF_phased.vcf.gz', with_ancestry=False, ancestry_file='ancestry.txt', label_include=['HLA'], feature_exclude=['HLA'], expert_by='ld'):
        self.label_include = label_include
        self.feature_exclude = feature_exclude
        self.expert_by = expert_by

        self.with_ancestry = with_ancestry
        if with_ancestry:
            self.data_dir = data_dir + '/with_ancestry'
        else:
            self.data_dir = data_dir + '/without_ancestry'
        if with_ancestry:
            self.ancestry_file = ancestry_file
            if not os.path.exists(ancestry_file):
                self.ancestry_file = self.data_dir + '/' + ancestry_file
                if not os.path.exists(self.ancestry_file):
                    print(f'ancestry file {ancestry_file} not found')
                else:
                    print(f'using ancestry file {self.ancestry_file}')
            else:
                print(f'using ancestry file {ancestry_file}')

        self.ref_phased = self.read_vcf(ref_phased)

        # Extract HLA for targets
        wh = []
        for n in range(self.ref_phased.shape[0]):
            ID = str(self.ref_phased['ID'].iloc[n])
            flag = False
            for flt in label_include:
                if ID.find(flt) != -1:
                    flag = True
                    break
            wh.append(flag)
        self.ref_phased_target = self.ref_phased[wh].copy()

        # Extract non-HLA for features
        wh = []
        for n in range(self.ref_phased.shape[0]):
            ID = str(self.ref_phased['ID'].iloc[n])
            flag = True
            for flt in feature_exclude:
                if ID.find(flt) != -1:
                    flag = False
                    break
            wh.append(flag)
        self.ref_phased_feature = self.ref_phased[wh].copy()

        self.ld_blocks = {}
        self.ld_blocks['HLA-A'] = ['HLA-A']
        self.ld_blocks['HLA-B'] = ['HLA-B', 'HLA-C']
        self.ld_blocks['HLA-C'] = ['HLA-B', 'HLA-C']
        self.ld_blocks['HLA-DPA1'] = ['HLA-DPA1', 'HLA-DPB1']
        self.ld_blocks['HLA-DPB1'] = ['HLA-DPA1', 'HLA-DPB1']
        self.ld_blocks['HLA-DQA1'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        self.ld_blocks['HLA-DQB1'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        self.ld_blocks['HLA-DRB1'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        self.expert_groups = {}
        self.expert_groups['E0'] = ['HLA-A']
        self.expert_groups['E1'] = ['HLA-B', 'HLA-C']
        self.expert_groups['E2'] = ['HLA-DPA1', 'HLA-DPB1']
        self.expert_groups['E3'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']

    def make_features(self, out_file='features.txt'):
        df = self.ref_phased_feature.iloc[:, 9:].T
        df.columns = self.ref_phased_feature['ID'] + '_' + self.ref_phased_feature['POS'].astype(str)
        df.reset_index(inplace=True, names=['sample'])

        if self.with_ancestry:
            df_anc = pd.read_table(self.ancestry_file, header=0, sep='\t', low_memory=False)
            df_anc_encoded = pd.get_dummies(df_anc.iloc[:, 1], prefix='ancestry').astype(int).astype(str)
            df_anc_encoded = df_anc_encoded + '|' + df_anc_encoded
            df_anc_encoded['sample'] = df_anc.iloc[:, 0]
            df_merged = pd.merge(df, df_anc_encoded, on='sample')
            if df_merged['sample'].tolist() != df['sample'].tolist():
                raise ValueError('Mismatch in sample names between features and ancestry data.')
            df = df_merged

        df.to_csv(out_file, sep='\t', index=False, header=True)
        out_file_list = out_file.split('.txt')[0] + '_list.txt'
        df.columns[1:].to_series().to_csv(out_file_list, index=False, header=False)

    def make_maps(self, out_file='maps.txt'):
        D = {}
        for n in range(self.ref_phased_target.shape[0]):
            ID = self.ref_phased_target['ID'].iloc[n]
            fields = ID.split(':')
            head = ':'.join(fields[0:-1])
            D.setdefault(head, [])
            if ID not in D[head]:
                D[head].append(ID)

        H = {}
        for k in sorted(D):
            H[k] = sorted(D[k])

        maps = []
        for head in H:
            for allele in H[head]:
                digit = len(allele.split(':')[1:]) * 2
                maps.append([digit, allele, H[head].index(allele) + 1, head])
        maps = pd.DataFrame(maps, columns=['digit', 'allele', 'label', 'head'])
        maps.sort_values(by=['digit', 'head', 'label'], inplace=True)

        heads = []
        for digit in sorted(maps['digit'].unique()):
            df_sub = maps[maps['digit'] == digit]
            for n in range(df_sub.shape[0]):
                head = df_sub['head'].iloc[n]
                if head not in heads:
                    heads.append(head)
        maps['head_idx'] = [heads.index(x) for x in maps['head']]

        parent = []
        parent_value = []
        expert = []
        for n in range(maps.shape[0]):
            digit = maps['digit'].iloc[n]
            head = maps['head'].iloc[n]
            if digit == 2:
                p = '.'
                p_val = -1
                e = maps['head'].iloc[n]
            else:
                p = ':'.join(head.split(':')[0:-1])
                p_val = H[p].index(head) + 1
                e = head.split(':')[0]
            parent.append(p)
            parent_value.append(p_val)
            expert.append(e)
        maps['parent'] = parent
        maps['parent_val'] = parent_value

        if self.expert_by == 'gene':
            maps['expert'] = expert
        elif self.expert_by == 'ld':
            E = []
            for x in expert:
                expert_id = '.'
                for k in self.expert_groups:
                    if x in self.expert_groups[k]:
                        expert_id = k
                        break
                if expert_id == '.':
                    expert_id = self.expert_groups.keys()[0]
                    print(f'{x} not found in LD groups, assigned to {expert_id}')
                E.append(expert_id)
            maps['expert'] = E 

        maps.to_csv(out_file, sep='\t', index=False, header=True)
        print('processed label maps:')
        print(maps)
        print(f'maps data saved to {out_file}')

    def make_labels(self, out_file='labels.txt', maps_file='maps.txt'):
        maps = pd.read_table(maps_file, header=0, sep='\t', low_memory=False)
        heads = maps['head'].unique().tolist()
        n_heads = len(heads)
        samples = self.ref_phased_target.columns[9:]
        D = dict(zip(maps['allele'], maps['label']))
        df = pd.DataFrame('0', index=samples, columns=heads)
        df1 = pd.DataFrame('0', index=samples, columns=heads)
        df2 = pd.DataFrame('0', index=samples, columns=heads)
        for n in range(self.ref_phased_target.shape[0]):
            ID = self.ref_phased_target['ID'].iloc[n]
            fields = ID.split(':')
            head = ':'.join(fields[0:-1])
            head_idx = heads.index(head)
            for m in range(len(samples)):
                allele = self.ref_phased_target.iloc[n, 9 + m].split('|')
                if len(allele) == 2:
                    if allele[0] == '1':
                        df1.iloc[m, head_idx] = D[ID]
                    if allele[1] == '1':
                        df2.iloc[m, head_idx] = D[ID]
        for n in range(df.shape[0]):
            for m in range(df.shape[1]):
                df.iloc[n, m] = str(df1.iloc[n, m]) + '|' + str(df2.iloc[n, m])
        df.index.name = 'sample'
        df.to_csv(out_file, sep='\t', index=True, header=True)

    def make_masks(self, out_file='masks.txt', features_file='features.txt', flank=500000):
        H = {}
        for n in range(self.ref_phased_target.shape[0]):
            gene = self.ref_phased_target['ID'].iloc[n].split(':')[0]
            pos = self.ref_phased_target['POS'].iloc[n]
            H.setdefault(gene, [])
            H[gene].append(pos)

        start_end_dict = {}
        for gene in H:
            if self.expert_by == 'gene':
                positions = H[gene]
                start_end_dict[gene] = (min(positions), max(positions))
            elif self.expert_by == 'ld':
                positions = []
                if gene in self.ld_blocks:
                    for g in self.ld_blocks[gene]:
                        if g in H:
                            positions += H[g]
                start_end_dict[gene] = (min(positions), max(positions))

        features = pd.read_table(features_file, header=0, sep='\t', low_memory=False)
        L = []
        E = []
        for gene in start_end_dict:
            if self.expert_by == 'gene':
                expert = gene
            elif self.expert_by == 'ld':
                expert = '.'
                for k in self.expert_groups:
                    if gene in self.expert_groups[k]:
                        expert = k
                        break
                if expert == '.':
                    expert = self.expert_groups.keys()[0]
                    print(f'{gene} not found in LD groups, assigned to {expert}')

            if expert not in E:
                E.append(expert)
                pos_min, pos_max = start_end_dict[gene]
                m = []
                for n in range(1, features.shape[1]):
                    fields = features.columns[n].split('_')
                    name = '_'.join(fields[0:-1])
                    if name.find('ancestry') != -1:
                        m.append(1)
                    else:
                        pos = int(fields[-1])
                        if pos >= pos_min - flank and pos <= pos_max + flank:
                            m.append(1)
                        else:
                            m.append(0)
                L.append([expert] + m)
        df = pd.DataFrame(L)
        df.columns = ['expert'] + features.columns[1:].tolist()
        df.to_csv(out_file, sep='\t', index=False, header=True)
        print(f'processed masks data: {out_file}')

    def get_sample_features(self, sample_vcf, features_file='features_list.txt', out_file='to_predict.txt'):
        if not os.path.exists(features_file):
            features_list = self.data_dir + '/' + features_file
            if os.path.exists(features_list):
                features_file = features_list
            else:
                raise FileNotFoundError(f'Features file {features_file} or {features_list} not found.')

        df = self.read_vcf(sample_vcf)
        samples = df.columns[9:].tolist()
        df_features = pd.read_table(features_file, header=None, sep='\t', low_memory=False)
        if df_features.shape[1] > 1:
            raise ValueError(f'Expected features file to have only one column, but found {df_features.shape[1]} columns.')
        features = [x for x in df_features.iloc[:, 0] if not x.startswith('ancestry_')]
        ancestry = [x.split('ancestry_')[-1] for x in df_features.iloc[:, 0] if x.startswith('ancestry_')]
        if len(ancestry):
            print(f'Ancestry features identified: {ancestry}')

        D = {}
        for n in range(df.shape[0]):
            k = str(df['ID'].iloc[n]) + '_' + str(df['POS'].iloc[n])
            D[k] = [x.split(':')[0] for x in df.iloc[n, 9:]]
        L = []
        for f in features:
            if f in D:
                L.append(D[f])
            else:
                L.append(['0|0'] * len(samples))
        df = pd.DataFrame(L).T
        df.index = samples
        df.columns = features
        df.reset_index(inplace=True, names=['sample'])

        if self.with_ancestry:
            df_anc = pd.read_table(self.ancestry_file, header=0, sep='\t', low_memory=False)
            # ancestry encoding must match the one used in the training data
            if not set(df_anc.iloc[:, 1]).issubset(set(ancestry)):
                raise ValueError('Ancestry categories in the ancestry file do not match those in the features list.')
            df_anc_encoded = pd.get_dummies(pd.Categorical(df_anc.iloc[:, 1], categories=ancestry), prefix='ancestry').astype(int).astype(str)
            df_anc_encoded = df_anc_encoded + '|' + df_anc_encoded
            df_anc_encoded['sample'] = df_anc.iloc[:, 0]
            df_merged = pd.merge(df, df_anc_encoded, on='sample')
            if df_merged['sample'].tolist() != df['sample'].tolist():
                raise ValueError('Mismatch in sample names between features and ancestry data.')
            df = df_merged
        else:
            print(f'Proceeding without ancestry features.')

        # double check that the feature columns in the sample data match the features list
        if df.columns[1:].to_list() != df_features.iloc[:, 0].to_list():
            raise ValueError('Mismatch in feature names between sample data and features list.')

        df.to_csv(out_file, sep='\t', index=False, header=True)
        print(f'sample features saved to {out_file}')

    def read_vcf(self, in_file):
        if in_file.endswith('vcf.gz'):
            with gzip.open(in_file, 'rt') as f:
                lines = [l for l in f if l.startswith('##')]
                n_header = len(lines)
        elif in_file.endswith('vcf'):
            with open(in_file, 'r') as f:
                lines = [l for l in f if l.startswith('##')]
                n_header = len(lines)
        else:
            raise ValueError('Input file must VCF format end with .vcf or .vcf.gz')

        df = pd.read_table(in_file, sep='\t', skiprows=n_header)
        df.rename(columns={'#CHROM': 'CHROM'}, inplace=True)
        return df

if __name__ == '__main__':
    pass
