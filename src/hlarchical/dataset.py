import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from .utils import *

class CustomDataset(Dataset):
    def __init__(self, features_file='features.txt', labels_file='labels.txt', maps_file='maps.txt', out_file='hla'):
        self.out_file = out_file
        df_features = pd.read_table(features_file, header=0, sep='\t')
        df_labels = pd.read_table(labels_file, header=0, sep='\t')
        if df_features.iloc[:, 0].tolist() != df_labels.iloc[:, 0].tolist():
            raise ValueError('The sample column of features and labels must be the same')

        mat = []
        for n in range(1, df_features.shape[1]):
            mat.append(df_features.iloc[:, n].str.split('|', expand=True).astype(int).values)
        mat = np.array(mat).transpose(1, 2, 0)
        self.X = torch.tensor(mat, dtype=torch.float32)

        mat = []
        for n in range(1, df_labels.shape[1]):
            mat.append(df_labels.iloc[:, n].str.split('|', expand=True).astype(int).values)
        mat = np.array(mat).transpose(1, 2, 0)
        self.y = torch.tensor(mat, dtype=torch.long)
        print(f'X shape: {self.X.shape}')
        print(f'y shape: {self.y.shape}')
        self.random_seed = 42

    def __len__(self):
        return(len(self.y))

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return(X, y)

    def split_save_dataset(self, ratio=[0.8, 0.2], batch_size=32, shuffle=True, num_workers=0, n_cv=0):
        self.train_file = self.out_file + '_dataset_train.pt'
        self.val_file = self.out_file + '_dataset_val.pt'
        self.test_file = self.out_file + '_dataset_test.pt'

        if len(ratio) == 2:
            self.ds_train, self.ds_test = torch.utils.data.random_split(self, ratio)
            if n_cv == 0:
                self.dl_train = DataLoader(self.ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
                self.dl_test = DataLoader(self.ds_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
                torch.save(self.dl_train, self.train_file)
                torch.save(self.dl_test, self.test_file)
                print(f'datasets saved to {self.train_file} and {self.test_file}')
            else:
                kf = KFold(n_splits=n_cv, shuffle=True, random_state=self.random_seed)
                for fold, (train_index, val_index) in enumerate(kf.split(self.ds_train)):
                    ds_train_cv = torch.utils.data.Subset(self.ds_train, train_index)
                    ds_val_cv = torch.utils.data.Subset(self.ds_train, val_index)

                    dl_train_cv = DataLoader(ds_train_cv, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
                    dl_val_cv = DataLoader(ds_val_cv, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

                    train_file_cv = self.out_file + f'_dataset_train_fold{fold+1}.pt'
                    val_file_cv = self.out_file + f'_dataset_val_fold{fold+1}.pt'

                    torch.save(dl_train_cv, train_file_cv)
                    torch.save(dl_val_cv, val_file_cv)
                    print(f'datasets saved to {train_file_cv} and {val_file_cv}')

        elif len(ratio) == 3:
            self.ds_train, self.ds_val, self.ds_test = torch.utils.data.random_split(self, ratio)

            self.dl_train = DataLoader(self.ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            self.dl_val = DataLoader(self.ds_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            self.dl_test = DataLoader(self.ds_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            torch.save(self.dl_train, self.train_file)
            torch.save(self.dl_val, self.val_file)
            torch.save(self.dl_test, self.test_file)
            print(f'datasets saved to {self.train_file}, {self.val_file}, and {self.test_file}')

if __name__ == '__main__':
    ds = CustomDataset()
    ds.split_save_dataset(ratio=[0.8, 0.2])

