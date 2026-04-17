import torch
import torch.nn as nn
from .utils import *

class MLPBackbone(nn.Module):
    def __init__(
        self,
        input_channels=2,
        input_length=1000,
        hidden_dims=(128, 64),
        dropout=0.3,
    ):
        super().__init__()

        self.input_dim = input_channels * input_length

        layers = []
        prev_dim = self.input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.net(x)


class CNNBackbone(nn.Module):
    def __init__(
        self,
        input_channels=2,
        input_length=1000,
        hidden_dims=(64, 128, 256),
        kernel_sizes=(7, 5, 3),
        strides=(1, 1, 1),
        dropout=0.1,
        use_batchnorm=True,
        global_pool="avg",  # "avg" or "max"
    ):
        super().__init__()

        assert len(hidden_dims) == len(kernel_sizes) == len(strides)

        layers = []
        prev_c = input_channels

        for out_c, k, s in zip(hidden_dims, kernel_sizes, strides):
            layers.append(
                nn.Conv1d(
                    prev_c,
                    out_c,
                    kernel_size=k,
                    stride=s,
                    padding=k // 2
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_c = out_c

        self.conv = nn.Sequential(*layers)

        if global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif global_pool == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError("global_pool must be 'avg' or 'max'")

        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        """
        x: (batch, channels, length)
        """
        x = self.conv(x)
        x = self.pool(x)      # (B, C, 1)
        x = x.squeeze(-1)     # (B, C)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, N, W, D):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(N)
        self.bn2 = nn.BatchNorm1d(N)
        self.conv1 = nn.Conv1d(N, N, W, dilation=D, padding='same')
        self.conv2 = nn.Conv1d(N, N, W, dilation=D, padding='same')
    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return(out)

class SpliceAIBackbone(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv1d(cfg.in_channels, cfg.out_channels, 1)
        self.conv2 = nn.Conv1d(cfg.out_channels, cfg.out_channels, 1)
        self.resblocks = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(len(cfg.NWD)):
            n,w,d = cfg.NWD[i]
            self.resblocks.append(ResidualBlock(n, w, d))
            if (i+1)%cfg.n_blocks == 0:
                self.convs.append(nn.Conv1d(cfg.out_channels, cfg.out_channels, 1))

        if cfg.global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif cfg.global_pool == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError("global_pool must be 'avg' or 'max'")
        self.output_dim = cfg.out_channels

    def forward(self, x):
        out = self.conv1(x)
        skip = self.conv2(out)
        for i in range(len(self.cfg.NWD)):
            n,w,d = self.cfg.NWD[i]
            out = self.resblocks[i](out)
            j = 0
            if (i+1)%self.cfg.n_blocks == 0:
                cv = self.convs[j](out)
                skip = cv + skip
                j += 1
        out = self.pool(skip)
        out = out.squeeze(-1)
        return out

class HierarchicalHLA(nn.Module):
    def __init__(self, cfg, maps_file='maps.txt', masks_file='masks.txt'):
        super().__init__()
        self.moe = False
        self.masks = {}

        if masks_file is not None:
            df = pd.read_table(masks_file, header=0, sep='\t')
            for n in range(df.shape[0]):
                m = df.iloc[n, 1:].values.astype(bool)
                self.masks[df.iloc[n, 0]] = m
                cfg.input_length = len(m)
        else:
            if not hasattr(cfg, 'input_length'):
                raise ValueError('input_length needed when masks file not provided')

        if hasattr(cfg, 'moe') and cfg.moe:
            self.moe = True
            print(f'using Mixture of Experts')

        self.maps = {}
        self.expert_to_head = {}

        df = pd.read_table(maps_file, header=0, sep='\t')
        for n in range(df.shape[0]):
            head = df['head'].iloc[n]
            expert = df['expert'].iloc[n]
            label = df['label'].iloc[n]
            self.maps[head] = [label, expert]
            self.expert_to_head[expert] = head

        if cfg.backbone == 'mlp':
            backbone_class = MLPBackbone
        elif cfg.backbone == 'cnn':
            backbone_class = CNNBackbone
        elif cfg.backbone == 'spliceai':
            backbone_class = SpliceAIBackbone

        if not self.moe:
            if cfg.backbone == 'mlp':
                self.backbone = backbone_class(
                    input_channels=cfg.input_channels,
                    input_length=cfg.input_length,
                    hidden_dims=cfg.hidden_dims,
                    dropout=cfg.dropout,
                )
            elif cfg.backbone == 'cnn':
                self.backbone = backbone_class(
                    input_channels=cfg.input_channels,
                    input_length=cfg.input_length,
                    hidden_dims=cfg.hidden_dims,
                    kernel_sizes=cfg.kernel_sizes,
                    strides=cfg.strides,
                    dropout=cfg.dropout,
                    use_batchnorm=cfg.use_batchnorm,
                    global_pool=cfg.global_pool,
                )
            elif cfg.backbone == 'spliceai':
                self.backbone = backbone_class(cfg)
            self.heads = nn.ModuleDict({head:nn.Linear(self.backbone.output_dim, (self.maps[head][0] + 1) * 2) for head in self.maps})
        else:
            self.experts = nn.ModuleDict()
            for e in self.expert_to_head:
                mask = self.masks[e]
                if cfg.backbone == 'mlp':
                    expert = backbone_class(
                        input_channels=cfg.input_channels,
                        input_length=sum(mask),
                        hidden_dims=cfg.hidden_dims,
                        dropout=cfg.dropout,
                    )
                elif cfg.backbone == 'cnn':
                    expert = backbone_class(
                        input_channels=cfg.input_channels,
                        input_length=sum(mask),
                        hidden_dims=cfg.hidden_dims,
                        kernel_sizes=cfg.kernel_sizes,
                        strides=cfg.strides,
                        dropout=cfg.dropout,
                        use_batchnorm=cfg.use_batchnorm,
                        global_pool=cfg.global_pool,
                    )
                elif cfg.backbone == 'spliceai':
                    expert = backbone_class(cfg)
                self.experts[e] = expert
            self.heads = nn.ModuleDict({head:nn.Linear(self.experts[self.maps[head][-1]].output_dim, (self.maps[head][0] + 1) * 2) for head in self.maps})

    def forward(self, x):
        if not self.moe:
            x = self.backbone(x)
            outputs = {}
            for head in self.heads:
                h = self.heads[head](x)
                h = h.view(h.size(0), -1, 2)
                outputs[head] = h
            return outputs
        else:
            experts = {}
            for e in self.experts:
                mask = self.masks[e]
                x_m = x[:, :, mask]
                x_out = self.experts[e](x_m)
                experts[e] = x_out

            outputs = {}
            for head in self.heads:
                e = self.maps[head][-1]
                x_e = experts[e]
                h = self.heads[head](x_e)
                h = h.view(h.size(0), -1, 2)
                outputs[head] = h
            return outputs

if __name__ == "__main__":
    class Config:
        pass

    cfg = Config()
    cfg.input_channels = 2
    cfg.input_length = 1000
    cfg.hidden_dims = (128, 64)
    cfg.dropout = 0.3
    cfg.moe = True
    cfg.backbone_class = MLPBackbone

    model = HierarchicalHLA(cfg)
    data = torch.randn(4, 2, 1000)
    outputs = model(data)
    #print(outputs['HLA-A'].shape)
    #print(outputs['HLA-B'].shape)
