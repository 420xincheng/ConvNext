# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BackBone(nn.Module):
    def __init__(self, base_encoder, num_classes, dim=128):
        super(BackBone, self).__init__()
        self.net = base_encoder
        self.fc = nn.Linear(self.net.out_dim, num_classes)
        self.head = nn.Sequential(
            nn.Linear(self.net.out_dim, self.net.out_dim),
            nn.ReLU(True),
            nn.Linear(self.net.out_dim, dim)
        )

    def forward(self, x, return_feat=False):
        if self.training:
            x = self.net(x)
            logits = self.fc(x)
            embedding = self.head(x)
            return logits, F.normalize(embedding)
        else:
            x = self.net(x)
            logits = self.fc(x)
            
            if return_feat:
                return logits, x
            return logits


class SimMatch(nn.Module):
    def __init__(self, base_encoder, num_classes=10,  momentum=0.999, dim=128,  K=256, args=None):
        super(SimMatch, self).__init__()
        self.m = momentum
        self.num_classes = num_classes
        self.encoder_q = BackBone(base_encoder, num_classes, dim=128)
        self.ema = copy.deepcopy(self.encoder_q)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.ema.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        self.register_buffer("bank", torch.randn(dim, K))
        self.bank = nn.functional.normalize(self.bank, dim=0)
        self.register_buffer("labels", torch.zeros(K, dtype=torch.long))

        if args.DA:
            self.DA_len = 32
            self.register_buffer("DA_queue", torch.zeros(self.DA_len, num_classes, dtype=torch.float))
            self.register_buffer("DA_ptr", torch.zeros(1, dtype=torch.long))
        

    def momentum_update_ema(self):
        for param_train, param_eval in zip(self.encoder_q.parameters(), self.ema.parameters()):
            param_eval.copy_(param_eval * self.m + param_train.detach() * (1-self.m))
        for buffer_train, buffer_eval in zip(self.encoder_q.buffers(), self.ema.buffers()):
            buffer_eval.copy_(buffer_train)


    @torch.no_grad()
    def _update_bank(self, k, labels, index, args):
        if torch.distributed.get_world_size() > 1:
            k      = concat_all_gather(k)
            labels = concat_all_gather(labels)
            index  = concat_all_gather(index)
        self.bank[:, index] =  F.normalize(self.bank[:, index] * args.bank_m +  k.t() * (1-args.bank_m))
        self.labels[index] = labels

    @torch.no_grad()
    def distribution_alignment(self, probs):
        probs_bt_mean = probs.mean(0)
        ptr = int(self.DA_ptr)
        if torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(probs_bt_mean)
            self.DA_queue[ptr] = probs_bt_mean / torch.distributed.get_world_size()
        else:
            self.DA_queue[ptr] = probs_bt_mean
        self.DA_ptr[0] = (ptr + 1) % self.DA_len
        probs = probs / self.DA_queue.mean(0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()
    

    def forward(self, im_x, im_u_w=None, im_u_s=None, labels=None, index=None, start_unlabel=False, args=None):
        self.momentum_update_ema()
        bx = im_x.shape[0]
        bu = im_u_w.shape[0]
        bank = self.bank.clone().detach()

        logits, embedding = self.encoder_q(torch.cat([im_x, im_u_w, im_u_s]))
        logits_x, logits_u_w, logits_u_s = logits[:bx], logits[bx:bx+bu], logits[bx+bu:]
        embedding_x, embedding_u_w, embedding_u_s = embedding[:bx], embedding[bx:bx+bu], embedding[bx+bu:]
        
        prob_u_w = F.softmax(logits_u_w, dim=-1)
        if args.DA:
            prob_u_w = self.distribution_alignment(prob_u_w)
        
        if start_unlabel:
            with torch.no_grad():
                teacher_logits = embedding_u_w @ bank
                teacher_prob_orig = F.softmax(teacher_logits / args.tt, dim=1)
                
                factor = prob_u_w.gather(1, self.labels.expand([bu, -1]))
                teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                if args.c_smooth < 1:
                    bs = teacher_prob_orig.size(0)
                    aggregated_prob = torch.zeros([bs, self.num_classes], device=teacher_prob_orig.device)
                    aggregated_prob = aggregated_prob.scatter_add(1, self.labels.expand([bs,-1]) , teacher_prob_orig)
                    prob_u_w = prob_u_w * args.c_smooth + aggregated_prob * (1-args.c_smooth)
            student_logits = embedding_u_s @ bank
            student_prob = F.softmax(student_logits / args.st, dim=1)
            loss_in = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1)
        else:
            loss_in = torch.tensor(0, dtype=torch.float).cuda()

        self._update_bank(embedding_x, labels, index, args)
        return logits_x, prob_u_w, logits_u_s, loss_in

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output
