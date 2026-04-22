import os
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from forestplot import forestplot
from .utils import *

class AssociationDiseaseHLA():
    def __init__(self):
        self.HLA = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']

    def preprocess(self, in_file, digit=2):
        df = pd.read_table(in_file, header=0, sep='\t')
        for hla in self.HLA:
            df2 = df[df['HLA'] == hla].copy()
            df2 = self.allele_binary_encoding(df2, digit=digit)
            out_file = in_file.replace('.txt', f'_{hla}_digit{digit}.txt')
            df2.to_csv(out_file, index=False, sep='\t')

    def allele_binary_encoding(self, df, digit=2):
        alleles = []
        A1 = []
        A2 = []
        for n in range(df.shape[0]):
            allele1 = df.iloc[n]['Allele1']
            allele2 = df.iloc[n]['Allele2']
            a1 = '.'
            a2 = '.'
            if allele1 not in ['.', 'X']:
                a1 = ':'.join(allele1.split(':')[0:int(digit/2) + 1])
                if a1.find('--') == -1:
                    if a1 not in alleles:
                        alleles.append(a1)
                else:
                    a1 = '.'
            if allele2 not in ['.', 'X']:
                a2 = ':'.join(allele2.split(':')[0:int(digit/2) + 1])
                if a2.find('--') == -1:
                    if a2 not in alleles:
                        alleles.append(a2)
                else:
                    a2 = '.'
            A1.append(a1)
            A2.append(a2)
        df['A1'] = A1
        df['A2'] = A2
        for allele in sorted(alleles):
            L = []
            for n in range(df.shape[0]):
                nA = 0
                if df.iloc[n]['A1'] == allele:
                    nA += 1
                if df.iloc[n]['A2'] == allele:
                    nA += 1
                L.append(nA)
            df[allele] = L
        return df

    def association_test(self, in_file, formula='Condition ~ HLA + Ancestry', out_dir='stats'):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        df0 = pd.read_table(in_file, header=0, sep='\t')
        idx = df0.columns.get_loc('A2')
        for n in range(idx + 1, df0.shape[1]):
            hla = df0.columns[n]
            out_file = out_dir + '/' + in_file.replace('.txt', f'_{hla}_stats.txt')
            out_file_covs = out_dir + '/' + in_file.replace('.txt', f'_{hla}_covs.txt')
            df = pd.DataFrame()
            df['HLA'] = df0[hla].astype(int)
            df['Condition'] = [0 if x == 'control' else 1 for x in df0['Condition']]
            df['Ancestry'] = df0['Ancestry'].astype('category')

            if formula.find('Ancestry') != -1:
                self._set_cat_reference(df, 'Ancestry', 'EUR') 

            model = smf.logit(formula, data=df)
            try:
                result = model.fit()
                summary = result.summary()
                df_covs = pd.DataFrame(result.cov_params())
                with open(out_file, 'w') as f:
                    f.write(hla + '\n')
                    f.write(summary.as_text())
                df_covs.to_csv(out_file_covs, sep='\t')
            except Exception as e:
                print(e)

    def sort_by_pvalue(self, in_dir, out_file, on='HLA'):
        fs = [f for f in os.listdir(in_dir) if f.endswith('_stats.txt')]
        L = []
        for f in fs:
            with open(in_dir + '/' + f, 'r') as infile:
                hla = infile.readline().strip()
                for line in infile:
                    if line.startswith(on):
                        values = line.strip().split()
                        L.append([hla] + values[1:])

        df = pd.DataFrame(L, columns=['HLA', 'coef', 'std_err', 'z', 'pvalue', 'ci_low', 'ci_high'])
        df.sort_values(by='pvalue', inplace=True)
        df.to_csv(out_file, index=False, sep='\t')

    def _set_cat_reference(self, df, column, reference):
        df[column] = df[column].cat.reorder_categories(
                [reference] + [x for x in df[column].cat.categories if x != reference], ordered=True)

    def llr_test(self, ll_full, ll_reduced, df_full, df_reduced):
        llr_stat = 2 * (ll_full - ll_reduced)
        df_diff = df_full - df_reduced
        p_value = chi2.sf(llr_stat, df_diff)
        print(f"LLR Statistic: {llr_stat}, p-value: {p_value}")
        return llr_stat, p_value

    def forest_plot(self, in_file, p_threshold = 0.05, top_n=20):
        df = pd.read_table(in_file, header=0, sep='\t')
        df = df[df['pvalue'] < p_threshold].copy()
        df = df.head(top_n)
        df.sort_values(by=['HLA', 'pvalue'], inplace=True)

        formatted_pvalue = []
        for p in df['pvalue']:
            if p < 0.001:
                formatted_pvalue.append(f"{p:.4f}***")
            elif p < 0.01:
                formatted_pvalue.append(f"{p:.4f}**")
            elif p < 0.05:
                formatted_pvalue.append(f"{p:.4f}*")
            else:
                formatted_pvalue.append(f"{p:.4f}")
        df['formatted_pvalue'] = formatted_pvalue

        forestplot(df, estimate='coef', ll='ci_low', hl='ci_high', se='std_err', xlabel='beta', varlabel='HLA',
                              rightannote=["formatted_pvalue"], right_annoteheaders=["pvalue"],
                              ci_report=False, **{'markercolor':'C1', 'linecolor':'C0', 'variable_header':'HLA'})
        out_file = in_file.replace('_sorted.txt', '.pdf')
        plt.savefig(out_file, bbox_inches="tight")
