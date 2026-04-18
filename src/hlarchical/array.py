from .utils import *

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

    def run_deephla(self, mode='train', in_file='OMNI_Pan-Asian_REF', ref_file='Pan-Asian_REF', model_json='Pan-Asian_REF.model.json', hla_json='Pan-Asian_REF.hla.json', model_dir='model_Pan-Asian_REF', deephla_dir=None, out_file='OMNI_Pan-Asian', subset=[]):
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

            if not os.path.exists(model_json): 
                raise FileNotFoundError(f'{model_json} not found')

            if not os.path.exists(hla_json):
                print('Generating HLA info JSON file...')
                cmd = f'conda run -n DEEP-HLA python {deephla_dir}/make_hlainfo.py --ref {ref_file} --out {hla_json}'
                print(cmd)
                subprocess.run(cmd, shell=True)

            if os.path.exists(model_json) and os.path.exists(hla_json):
                model_json = model_json.split('.model.json')[0]
                hla_json = hla_json.split('.hla.json')[0]
                cmd = f'conda run -n DEEP-HLA python {deephla_dir}/train.py --ref {ref_file} --sample {in_file} --model {model_json} --hla {hla_json} --model-dir {model_dir}'
                print(cmd)
                subprocess.run(cmd, shell=True)
            else:
                raise FileNotFoundError(f'{model_json} not found')

        elif mode == 'impute':
            model_json = model_json.split('.model.json')[0]
            hla_json = hla_json.split('.hla.json')[0]
            cmd = f'conda run -n DEEP-HLA python {deephla_dir}/impute.py --sample {in_file} --model {model_json} --hla  {hla_json} --model-dir {model_dir} --out {out_file}'
            print(cmd)
            subprocess.run(cmd, shell=True)
