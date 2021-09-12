import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.init import xavier_normal_
from .odeblock import ODEBlock
from .MGCN import *
from .MGCNLayer import *

class TANGO(nn.Module):
    def __init__(self, num_e, num_rel, params, device, logger):
        super().__init__()

        self.num_e = num_e
        self.num_rel = num_rel
        self.p = params
        self.core = self.p.gde_core
        self.core_layer = self.p.core_layer
        self.score_func = self.p.score_func
        self.solver = self.p.solver
        self.rtol = self.p.rtol
        self.atol = self.p.atol
        self.device = self.p.device
        self.initsize = self.p.initsize
        self.adjoint_flag = self.p.adjoint_flag
        self.drop = self.p.dropout
        self.hidsize = self.p.hidsize
        self.embsize = self.p.embsize

        self.device = device
        self.logger = logger
        if self.p.activation.lower() == 'tanh':
            self.act = torch.tanh
        elif self.p.activation.lower() == 'relu':
            self.act = F.relu
        elif self.p.activation.lower() == 'leakyrelu':
            self.act = F.leaky_relu

        # define loss
        self.loss = torch.nn.CrossEntropyLoss()

        # define entity and relation embeddings
        self.emb_e = self.get_param((self.num_e, self.initsize))
        self.emb_r = self.get_param((self.num_rel * 2, self.initsize))

        # define graph ode core
        self.gde_func = self.construct_gde_func()

        # define ode block
        self.odeblock = self.construct_GDEBlock(self.gde_func)

        # define jump modules
        if self.p.jump:
            self.jump, self.jump_weight = self.Jump()
            self.gde_func.jump = self.jump
            self.gde_func.jump_weight = self.jump_weight

        # score function TuckER
        if self.score_func.lower() == "tucker":
            self.W_tk, self.input_dropout, self.hidden_dropout1, self.hidden_dropout2, self.bn0, self.bn1 = self.TuckER()

    def get_param(self, shape):
        # a function to initialize embedding
        param = Parameter(torch.empty(shape, requires_grad=True, device=self.device))
        torch.nn.init.xavier_normal_(param.data)
        return param

    def add_base(self):
        model = MGCNLayerWrapper(None, None, self.num_e, self.num_rel, self.act, drop1=self.drop, drop2=self.drop,
                                       sub=None, rel=None, params=self.p)
        model.to(self.device)
        return model

    def construct_gde_func(self):
        gdefunc = self.add_base()
        return gdefunc

    def construct_GDEBlock(self, gdefunc):
        gde = ODEBlock(odefunc=gdefunc, method=self.solver, atol=self.atol, rtol=self.rtol, adjoint=self.adjoint_flag).to(self.device)
        return gde

    def TuckER(self):
        W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.hidsize, self.hidsize, self.hidsize)),
                                    dtype=torch.float, device=self.device, requires_grad=True))
        input_dropout = torch.nn.Dropout(self.drop)
        hidden_dropout1 = torch.nn.Dropout(self.drop)
        hidden_dropout2 = torch.nn.Dropout(self.drop)

        bn0 = torch.nn.BatchNorm1d(self.hidsize)
        bn1 = torch.nn.BatchNorm1d(self.hidsize)

        input_dropout.to(self.device)
        hidden_dropout1.to(self.device)
        hidden_dropout2.to(self.device)
        bn0.to(self.device)
        bn1.to(self.device)

        return W, input_dropout, hidden_dropout1, hidden_dropout2, bn0, bn1

    def Jump(self):
        if self.p.rel_jump:
            jump = MGCNConvLayer(self.hidsize, self.hidsize, act=self.act, params=self.p, isjump=True, diag=True)
        else:
            jump = GCNConvLayer(self.hidsize, self.hidsize, act=self.act, params=self.p)

        jump.to(self.device)
        jump_weight = torch.FloatTensor([self.p.jump_init]).to(self.device)
        return jump, jump_weight

    def loss_comp(self, sub, rel, emb, label, core, obj=None):
        score = self.score_comp(sub, rel, emb, core)
        return self.loss(score, obj)


    def score_comp(self, sub, rel, emb, core):
        sub_emb, rel_emb, all_emb = self.find_related(sub, rel, emb)
        if self.score_func.lower() == 'distmult':
            obj_emb = torch.cat([torch.index_select(self.emb_e, 0, sub), sub_emb], dim=1) * rel_emb.repeat(1,2)
            s = torch.mm(obj_emb, torch.cat([self.emb_e, all_emb], dim=1).transpose(1,0))

        if self.score_func.lower() == 'tucker':
            x = self.bn0(sub_emb)
            x = self.input_dropout(x)
            x = x.view(-1, 1, sub_emb.size(1))

            W_mat = torch.mm(rel_emb, self.W_tk.view(rel_emb.size(1), -1))
            W_mat = W_mat.view(-1, sub_emb.size(1), sub_emb.size(1))
            W_mat = self.hidden_dropout1(W_mat)

            x = torch.bmm(x, W_mat)
            x = x.view(-1, sub_emb.size(1))
            x = self.bn1(x)
            x = self.hidden_dropout2(x)
            s = torch.mm(x, all_emb.transpose(1, 0))

        return s

    def find_related(self, sub, rel, emb):
        x = emb[:self.num_e,:]
        r = emb[self.num_e:,:]
        assert x.shape[0] == self.num_e
        assert r.shape[0] == self.num_rel * 2
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        return sub_emb, rel_emb, x

    def push_data(self, *args):
        out_args = []
        for arg in args:
            arg = [_arg.to(self.device) for _arg in arg]
            out_args.append(arg)
        return out_args

    def forward(self, sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list,
                edge_type_list, edge_id_jump, edge_w_jump, rel_jump):
        # self.test_flag = 0

        # push data onto gpu
        if self.p.jump:
            [sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump] = \
                        self.push_data(sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump)
        else:
            [sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list] = \
                        self.push_data(sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list)
        # for RE decoder
        # if self.score_func.lower() == 're' or self.score_func.lower():
        #     [obj_tar] = self.push_data(obj_tar)


        emb = torch.cat([self.emb_e, self.emb_r], dim=0)

        for i in range(len(times)):
            self.odeblock.odefunc.set_graph(edge_index_list[i], edge_type_list[i])
            if i != (len(times) - 1):
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight, False)

                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=times[i+1], cheby_grid=self.p.cheby_grid)
            else:
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                           jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                           False)
                    
                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=tar_times[0], cheby_grid=self.p.cheby_grid)

                loss = self.loss_comp(sub_tar[0], rel_tar[0], emb, lab_tar[0], self.odeblock.odefunc,
                                              obj=obj_tar[0])

        return loss

    def forward_eval(self, times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump):
        # push data onto gpu
        if self.p.jump:
            [times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump] = \
                self.push_data(times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump)
        else:
            [times, tar_times, edge_index_list, edge_type_list, edge_index_list] = \
                self.push_data(times, tar_times, edge_index_list, edge_type_list, edge_index_list)

        emb = torch.cat([self.emb_e, self.emb_r], dim=0)

        for i in range(len(times)):
            self.odeblock.odefunc.set_graph(edge_index_list[i], edge_type_list[i])
            if i != (len(times) - 1):
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                           jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                           False)
                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=times[i + 1], cheby_grid=self.p.cheby_grid)
            else:
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                           jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                           False)
                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=tar_times[0], cheby_grid=self.p.cheby_grid)

        return emb