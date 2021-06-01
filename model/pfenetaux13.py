import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2
import pdb
import model.resnet as models



def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat
  


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        layers = args.layers
        classes = args.classes
        sync_bn = args.sync_bn
        pretrained = True
        assert layers in [50, 101, 152]
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.zoom_factor = args.zoom_factor
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.shot = args.shot
        assert layers in [50, 101, 152]

        self.ppm_scales = args.ppm_scales
        self.EM_k = args.emk
        models.BatchNorm = BatchNorm
        

        print('INFO: Using ResNet {}'.format(layers))
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        reduce_dim = 256
        fea_dim = 1024 + 512       

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )                 

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )  

        self.pyramid_bins = self.ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )


        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []        
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))                      
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            
        self.init_merge = nn.ModuleList(self.init_merge) 
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)                             


        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )                        
     
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
     


    def forward(self, x, s_x=torch.FloatTensor(1,1,3,473,473).cuda(), s_y=torch.FloatTensor(1,1,473,473).cuda(), s_seed=None, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)  
            query_feat_4 = self.layer4(query_feat_3)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        #   Support Feature     
        pri_proto_list = []
        aux_proto_list = []
        final_supp_list = []
        # supp_feat_list = []
        mask_list = []
        supp_feats = []
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                final_supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(final_supp_feat_4)

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_feat_v = Weighted_GAP(supp_feat, mask)
            # supp_feat_list.appned(supp_feat_v)
            # 计算sup上的相似率
            # print(supp_feat.size(),supp_feat_v.size())
            for i in range(self.EM_k):
                probs = F.cosine_similarity(supp_feat,supp_feat_v,dim=1).unsqueeze(1)
                aux_probs = (1-probs) * mask
                aux_feat_v = Weighted_GAP(supp_feat,aux_probs)
                supp_feat_v = Weighted_GAP(supp_feat,probs)
            pri_proto_list.append(supp_feat_v)
            aux_proto_list.append(aux_feat_v)

        # prior mask
        corr_query_mask = self.priormask(final_supp_list,mask_list,query_feat_4,query_feat_3,query_feat)

        if self.shot > 1:
            pri_proto = pri_proto_list[0]
            aux_proto = aux_proto_list[0]
            # channel_att = supp_feat_list[0]
            for i in range(1, len(pri_proto_list)):
                pri_proto += pri_proto_list[i]
                aux_proto += aux_proto_list[i]
                # channel_att = supp_feat_list[i]
            pri_proto /= len(pri_proto_list)
            aux_proto /= len(aux_proto_list)
        else:
            pri_proto = pri_proto_list[0]
            aux_proto = aux_proto_list[0]

        out,out_list = self.decoder(corr_query_mask,[pri_proto,aux_proto],query_feat)

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            # calculate query
            main_loss = self.criterion(out, y.long())
            aux_loss = torch.zeros_like(main_loss).cuda()

            for idx_k in range(len(out_list)):    
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())   
            aux_loss = aux_loss / len(out_list)

            return out.max(1)[1], main_loss, aux_loss
        else:
            return out




    def priormask(self,final_supp_list,supp_mask_list,query_feat_4,query_feat_3,query_feat):
        '''

        Args:
            final_supp_list:
            supp_mask_list:
            query_feat_4:

        Returns:

        '''
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(supp_mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]),
                                       mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear',
                                        align_corners=True)
        return corr_query_mask

    def decoder(self,corr_query_mask: torch.Tensor,prototypes: list, query_feat:torch.Tensor):
        '''

        Args:
            corr_query_mask:
            prototype:
            query_feat:

        Returns:

        '''
        out_list = []
        pyramid_feat_list = []

        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
            proto_feat_bin = torch.cat([proto.expand(-1, -1, bin, bin) for proto in prototypes],dim=1)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([query_feat_bin, proto_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)
        return out,out_list

    def _optimizer(self, args):
        optimizer = torch.optim.SGD(
            [
                {'params': self.down_query.parameters()},
                {'params': self.down_supp.parameters()},
                {'params': self.init_merge.parameters()},
                {'params': self.alpha_conv.parameters()},
                {'params': self.beta_conv.parameters()},
                {'params': self.inner_cls.parameters()},
                {'params': self.res1.parameters()},
                {'params': self.res2.parameters()},        
                {'params': self.cls.parameters()}
            ],
            lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer



