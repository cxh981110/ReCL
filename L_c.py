import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, cls_num_list, temperature=0.1, contrast_mode='all',
                 ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.cls_num_list = cls_num_list

    def forward(self, config, features, centers, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)    #在cifar100中，feater的维度是128*2*128，变换后的features的维度还是128*2*128

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:    #传递了labels没有传递mask
            labels = labels.contiguous().view(-1, 1)    #将labels转换为了128*1的张量
            new_label = torch.cat((labels, labels), dim=0)
            
            labels_centers = torch.arange(len(self.cls_num_list), device=device).view(-1, 1)
            labels = torch.cat([labels.repeat(2, 1), labels_centers], dim=0)
            mask = torch.eq(labels[:2 * batch_size], labels.T).float().to(device)   #创建了一个布尔类型的张量，类别一样的是True，用来找同类的样本，mask维度是128*128
        #     print('labels',mask.shape)
        
        tail_h = config.tail_class_idx[0]
        tail_t = config.tail_class_idx[1]
        # wnt = 1.0 * (sum(self.cls_num_list)/sum(self.cls_num_list[:tail_h]))
        # wt = 0.7 * (sum(self.cls_num_list)/sum(self.cls_num_list[tail_h:]))
        new_label = new_label.squeeze()
        labels = labels.squeeze()
        
        mask_tail = torch.zeros(new_label.size(0), labels.size(0), dtype=torch.bool).to(device)
        
        mask_tail[(new_label.unsqueeze(1) > tail_h) & (new_label.unsqueeze(1) < tail_t) & (labels.unsqueeze(0) > tail_h) & (labels.unsqueeze(0) < tail_t) & (new_label.unsqueeze(1) != labels.unsqueeze(0))] = True
        mask_tail = mask_tail.float()


        contrast_count = features.shape[1]    #contrast_count = 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)       #将features的维度由128*2*1024转换为了256*1024
        contrast_feature = torch.cat([contrast_feature, centers], dim=0)  
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature            #anchor_features的维度也是256*1024，可以理解为将两个不同数据增强的128张图像的特征链接在了一起
            anchor_count = contrast_count                #anchor_count=2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature[:2*batch_size], contrast_feature.T),    #得到的是256*256的张量，其中每一个数都代表两个特征向量的点积，并且经过temperature的缩放，就是对比学习中的相似度得分
            self.temperature)                                    # anchor_dot_contrast就是相似度矩阵
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)     #找出anchor_dot_contrast每一行最大的值(相似度得分)， logits_max的维度是256*1
        logits = anchor_dot_contrast - logits_max.detach()        #anchor_dot_contrast中的元素减去logits_max相对应的行元素。94行进行详细说明

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,                 #dim=1，按照列来的
            torch.arange(batch_size * 2).view(-1, 1).to(device),   #index   维度是256*1
            0
        )                      #这一块代码的作用是将mask对角线上的元素置为0，其余的置为1，维度是256*256
        mask = mask * logits_mask          #此时的mask是256*256,并且是对角线为0的张量，这一步的目的是为了将自己与自己的相似度置为0
        # mask_tail = mask_tail * logits_mask
        mask_neg = torch.ones_like(mask) - mask
        # mask_neg = mask_neg * logits_mask
        mask_no_tail = mask_neg - mask_tail
        mask_tail = 5*mask_tail
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask   
        #不同的logit改成e的logit次方，自己与自己的相似度除外
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # compute mean of log-likelihood over negative
        mean_log_prob_neg = (mask_neg * log_prob).sum(1) / mask_neg.sum(1)
        # mean_log_prob_tail = (mask_tail * log_prob).sum(1) / mask_tail.sum(1)
        mean_log_prob_tail = (mask_tail * log_prob).sum(1) / (mask_tail.sum(1) + (mask_tail.sum(1) == 0).float())
        mean_log_prob_notail = (mask_no_tail * log_prob).sum(1) / mask_no_tail.sum(1)

        # loss
        # loss = - mean_log_prob_pos
        loss = -(mean_log_prob_tail+mean_log_prob_notail)      
        loss = loss.view(anchor_count, batch_size).mean()
        #mean_log_prob_neg是一个负数，我希望这个数能够越来越小
        return loss

        '''为了能匹配到减法,先将logits_max由256*1扩展成256*256,其中每一行的所有元素都是一样的。比如logits_max本来是从1到256的张量。
        现在变成266个1,256个2一直到256个256的张量'''
        # 里边的内容：第一行的值就是一个数，是anchor_dot_contrast除了第一行以外，剩下255行最大点积的和，第二行也是一个数，是anchor_dot_contrast除了第二行以外，剩下255行最大点积的和，同理往下推导

