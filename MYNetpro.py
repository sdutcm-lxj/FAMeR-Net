from asyncore import poll3
import timm
import torch.nn as nn
import torch
import numpy as np
from DAT_org import Spatial_Embeddings, Encoder, PatchExpand2D
from tools import CrossAttention, SelfAttention
import torchvision.models as models
# from .TF_configs import get_model_config


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)

        if self.scale_factor[0] > 1:
            x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Down_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_block, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.Maxpool(x)
        x = self.conv(x)
        return x


class DRA_C(nn.Module):
    """ Channel-wise DRA Module """
    def __init__(self, skip_dim, decoder_dim, img_size, config):
        super().__init__()
        # 修改点：patch_size 改为 img_size // 16（原为 //14），适配256输入下Transformer输出16x16
        self.patch_size = img_size // 16
        self.ft_size = img_size
        self.patch_embeddings = nn.Conv2d(in_channels=decoder_dim,
                                          out_channels=decoder_dim,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)

        self.conv = nn.Sequential(
            nn.Conv2d(decoder_dim, skip_dim, kernel_size=(1, 1), bias=True),
            nn.BatchNorm2d(skip_dim),
            nn.ReLU(inplace=True))
        self.query = nn.Linear(decoder_dim, skip_dim, bias=False)
        self.key = nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.value = nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.out = nn.Linear(skip_dim, skip_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.psi = nn.InstanceNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.reconstruct = Reconstruct(skip_dim, skip_dim, kernel_size=1,
                                       scale_factor=(self.patch_size, self.patch_size))

    def forward(self, decoder, trans):
        decoder_mask = self.conv(decoder)
        decoder_L = self.patch_embeddings(decoder).flatten(2).transpose(-1, -2)

        query = self.query(decoder_L).transpose(-1, -2)
        key = self.key(trans)
        value = self.value(trans).transpose(-1, -2)
        ch_similarity_matrix = torch.matmul(query, key)

        ch_similarity_matrix = self.softmax(self.psi(ch_similarity_matrix.unsqueeze(1)).squeeze(1))

        out = torch.matmul(ch_similarity_matrix, value).transpose(-1, -2)

        out = self.out(out)
        out = self.reconstruct(out)
        out = out * decoder_mask

        return out


class DRA_S(nn.Module):
    """ Spatial-wise DRA Module """
    def __init__(self, skip_dim, decoder_dim, img_size, config, double=False):
        super().__init__()
        # 修改点：patch_size 改为 img_size // 16
        self.patch_size = img_size // 16
        self.ft_size = img_size
        self.patch_embeddings = nn.Conv2d(in_channels=decoder_dim,
                                          out_channels=decoder_dim,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)

        self.conv = nn.Sequential(
            nn.Conv2d(decoder_dim, skip_dim, kernel_size=(1, 1), bias=True),
            nn.BatchNorm2d(skip_dim),
            nn.ReLU(inplace=True))
        self.query = nn.Linear(decoder_dim, skip_dim, bias=False)
        self.key = nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.value = nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.out = nn.Linear(skip_dim, skip_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.psi = nn.InstanceNorm2d(1)
        self.reconstruct = Reconstruct(skip_dim, skip_dim, kernel_size=1,
                                       scale_factor=(self.patch_size, self.patch_size))

    def forward(self, decoder, trans_o):
        decoder_mask = self.conv(decoder)
        decoder_L = self.patch_embeddings(decoder).flatten(2).transpose(-1, -2)
        trans = trans_o
        query = self.query(decoder_L)
        key = self.key(trans).transpose(-1, -2)
        value = self.value(trans)
        sp_similarity_matrix = torch.matmul(query, key)
        sp_similarity_matrix = self.softmax(self.psi(sp_similarity_matrix.unsqueeze(0)).squeeze(0))
        out = torch.matmul(sp_similarity_matrix, value)
        out = self.out(out)
        out = self.reconstruct(out)
        out = out * decoder_mask
        return out


class Up_BlockV2(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, img_size, config, double=False):
        super().__init__()
        self.scale_factor = (img_size // 16, img_size // 16)  # 用于上采样因子，但实际未使用
        self.up = PatchExpand2D(in_ch // 2)
        self.pam = DRA_C(skip_ch, in_ch // 2, img_size, config)  # channel-wise DRA
        # 若需要使用 spatial-wise DRA，可替换为 DRA_S
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, decoder, o_i):
        if o_i is not None:
            decoder = decoder.permute(0, 2, 3, 1)
            d_i = self.up(decoder).permute(0, 3, 1, 2)
            o_hat_i = self.pam(d_i, o_i)
            x = torch.cat((o_hat_i, d_i), dim=1)
            x = self.conv(x)
        else:
            x = self.up(decoder.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class Up_Block(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, img_size, config):
        super().__init__()
        self.scale_factor = (img_size // 14, img_size // 14)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True))
        self.pam = DRA_S(skip_ch, in_ch // 2, img_size, config)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, decoder, o_i):
        if o_i is not None:
            d_i = self.up(decoder)
            o_hat_i = self.pam(d_i, o_i)
            x = torch.cat((o_i, d_i), dim=1)
            x = self.conv(x)
        else:
            x = self.up(decoder)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.mul(x, y)
        return y


class CABlock(nn.Module):
    def __init__(self, channel_c=512, wh=196):
        super(CABlock, self).__init__()
        self.query = nn.Linear(196, 196, bias=False)
        self.key = nn.Linear(512, 512, bias=False)
        self.value = nn.Linear(196, 196, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.psi = nn.InstanceNorm2d(1)
        self.out = nn.Linear(196, 196, bias=False)
        self.conv1d = nn.Linear(192, 512, bias=False)

    def forward(self, x, y):
        b, c, w, h = y.size()
        i = self.conv1d(x.transpose(-1, -2)).transpose(-1, -2)
        j = torch.flatten(y, start_dim=2)
        query = self.query(i)
        key = self.key(j.transpose(-1, -2))
        value = self.value(i)
        sp_similarity_matrix = torch.matmul(query, key)
        sp_similarity_matrix = self.softmax(self.psi(sp_similarity_matrix.unsqueeze(0)).squeeze(0))
        out = torch.matmul(sp_similarity_matrix, value)
        out = self.out(out)
        out = out.view(b, c, h, w)
        return out


class SABlock(nn.Module):
    def __init__(self, channel_c=1024, wh=196):
        super(SABlock, self).__init__()
        self.fc1 = nn.Linear(196, 392)
        self.fc2 = nn.Linear(392, 196)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(1024, 1024, bias=False)
        self.key = nn.Linear(196, 196, bias=False)
        self.value = nn.Linear(1024, 1024, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.psi = nn.InstanceNorm2d(1)
        self.out = nn.Linear(1024, 512, bias=False)
        self.conv1d = nn.Linear(192, 512, bias=False)

    def forward(self, x, y):
        b, c, w, h = y.size()
        i = self.conv1d(x.transpose(-1, -2)).transpose(-1, -2)
        j = torch.flatten(y, start_dim=2)
        i = torch.cat((i, j), dim=1)
        i = self.act_fn(self.fc1(i))
        i = self.fc2(self.dropout(i))
        i = i.transpose(-1, -2)
        query = self.query(i)
        key = self.key(i.transpose(-1, -2))
        value = self.value(i)
        sp_similarity_matrix = torch.matmul(query, key)
        sp_similarity_matrix = self.softmax(self.psi(sp_similarity_matrix.unsqueeze(0)).squeeze(0))
        out = torch.matmul(sp_similarity_matrix, value)
        out = self.out(out).transpose(-1, -2)
        out = out.view(b, c, h, w)
        return out


class MYNet(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=256, supervision=False):  # 修改默认img_size为256
        super().__init__()
        self.n_classes = n_classes
        resnet = models.resnet34(pretrained=False)
        filters_resnet = [64, 64, 128, 256, 512]
        filters_decoder = config.decoder_channels

        # =====================================================
        # Encoder
        # =====================================================
        self.Conv1 = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.Conv2 = resnet.layer1
        self.Conv3 = resnet.layer2
        self.Conv4 = resnet.layer3
        self.Conv5 = resnet.layer4

        transformer = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=False, img_size=256)
        self.transformers = nn.ModuleList([transformer.blocks[i] for i in range(12)])
        self.patch_embed = transformer.patch_embed
        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.zip = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        # =====================================================
        # DAT Module
        # =====================================================
        self.se = SEBlock(channel=1024)
        # 修改Spatial_Embeddings的img_size参数，适配256输入下ResNet各阶段输出尺寸：e1=128, e2=128, e3=64, e4=32
        self.emb1 = Spatial_Embeddings(config, 8, img_size=128, in_channels=64)
        self.emb2 = Spatial_Embeddings(config, 8, img_size=128, in_channels=64)
        self.emb3 = Spatial_Embeddings(config, 4, img_size=64, in_channels=128)
        self.emb4 = Spatial_Embeddings(config, 2, img_size=32, in_channels=256)

        self.ca1 = CrossAttention(config.transformer["num_heads"], config.transformer["embedding_channels"], 192,
                                  config.transformer["embedding_channels"])
        self.ca2 = CrossAttention(config.transformer["num_heads"], config.transformer["embedding_channels"], 192,
                                  config.transformer["embedding_channels"])
        self.ca3 = CrossAttention(config.transformer["num_heads"], config.transformer["embedding_channels"], 192,
                                  config.transformer["embedding_channels"])
        self.ca4 = CrossAttention(config.transformer["num_heads"], config.transformer["embedding_channels"], 192,
                                  config.transformer["embedding_channels"])

        self.encoder = Encoder(config, channel_num=filters_resnet[0:5])

        # =====================================================
        # DRA & Decoder
        # =====================================================
        # 修改Up_BlockV2的img_size参数：上采样后的目标尺寸分别为32,64,128,256
        self.Up5 = Up_BlockV2(filters_resnet[4], filters_resnet[3], filters_decoder[3], 32, config)
        self.Up4 = Up_BlockV2(filters_decoder[3], filters_resnet[2], filters_decoder[2], 64, config)
        self.Up3 = Up_BlockV2(filters_decoder[2], filters_resnet[1], filters_decoder[1], 128, config)
        self.Up2 = Up_BlockV2(filters_decoder[1], filters_resnet[0], filters_decoder[0], 256, config, double=True)

        self.pred = nn.Sequential(
            nn.Conv2d(filters_decoder[0], filters_decoder[0] // 2, kernel_size=1),
            nn.BatchNorm2d(filters_decoder[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_decoder[0] // 2, n_classes, kernel_size=1),
        )
        self.last_activation = nn.Sigmoid()

        self.supervision = supervision
        if self.supervision:
            self.predaux = nn.Sequential(
                nn.Conv2d(filters_decoder[0] * 2, filters_decoder[0] // 2, kernel_size=1),
                nn.BatchNorm2d(filters_decoder[0] // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(filters_decoder[0] // 2, n_classes, kernel_size=1),
            )
            self.last_activation_aux = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e1 = self.Conv1(x)                     # 256 -> 128
        e1_maxp = self.firstrelu(self.firstbn(e1))
        e2 = self.Conv2(e1_maxp)               # 128 -> 128
        e3 = self.Conv3(e2)                    # 128 -> 64
        e4 = self.Conv4(e3)                    # 64 -> 32
        e5 = self.Conv5(e4)                    # 32 -> 16

        # Transformer branch
        token1 = self.emb1(e1)                 # e1 size: 128
        token2 = self.emb2(e2)                 # 128
        token3 = self.emb3(e3)                 # 64
        token4 = self.emb4(e4)                 # 32

        emb = self.patch_embed(x)              # 256 -> 16x16 patches (256 tokens)

        for i in range(0, 2):
            emb = self.transformers[i](emb)
        token1 = self.ca1(token1, emb)

        for i in range(2, 4):
            emb = self.transformers[i](emb)
        token2 = self.ca2(token2, emb)

        for i in range(4, 8):
            emb = self.transformers[i](emb)
        token3 = self.ca3(token3, emb)

        for i in range(8, 10):
            emb = self.transformers[i](emb)
        token4 = self.ca4(token3, emb)

        for i in range(10, 12):
            emb = self.transformers[i](emb)

        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 192, 16, 16)   # 16x16
        feature_tf = self.conv_seq_img(feature_tf)

        e5 = self.zip(self.se(torch.cat((e5, feature_tf), dim=1)))  # e5:16x16

        token1, token2, token3, token4 = self.encoder(token1, token2, token3, token4)

        # Decoder with DRA
        d4 = self.Up5(e5, None)                 # 16 -> 32
        d3 = self.Up4(d4, token3)               # 32 -> 64
        d2 = self.Up3(d3, token2)               # 64 -> 128
        d1 = self.Up2(d2, token1)               # 128 -> 256

        if self.n_classes == 1:
            out = self.pred(d1)
        else:
            out = self.pred(d1)

        return out, d4, d3, d2, d1