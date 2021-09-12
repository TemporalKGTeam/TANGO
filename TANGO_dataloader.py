import numpy as np
import torch
import torch.utils

class TANGOtrainDataset(torch.utils.data.Dataset):
    def __init__(self,
                 params,
                 triples: list, # triples['train']
                 adjs: list, # {'edge_index': tensor, 'edge_type': tensor}
                 adjlist: list, # [adjmtx,...,adjmtx], adjmtx is torch sparse tensor
                 so2r: list,
                 num_e: int,
                 input_steps: int,
                 target_steps: int,
                 delta_steps: int = 0,
                 time_stamps: list = None,
                 num_samp=None,
                 neg_samp=None):

        assert isinstance(triples, list)
        self.p = params
        self.num_e = num_e
        self.num_samp = num_samp
        self.input_steps = input_steps
        self.target_steps = target_steps
        self.delta_steps = delta_steps
        self.triples = triples
        self.adjs = adjs
        self.so2r = so2r
        self.neg_samp = neg_samp

        self.len = len(self.triples) - self.input_steps - self.target_steps - self.delta_steps + 1

        assert len(triples) == len(time_stamps), "length of time stamps do not match with trajectories"
        self.time_stamps = time_stamps
        self.adjlist = adjlist

    def __getitem__(self, idx):
        # target timestamps
        target_time_stamps = []
        for t_idx in range(idx + self.input_steps + self.delta_steps,
                           idx + self.input_steps + self.delta_steps + self.target_steps):
            target_time_stamps.append(self.time_stamps[t_idx])

        # graph info: (sub, rel, obj)
        triple_input = []
        for i_idx in range(idx, idx + self.input_steps):
            triple_input.append(torch.tensor([list(trp['triple']) for trp in self.triples[i_idx]]))

        # sub
        subject_input = [torch.stack([_trp[i,:] for i in range(_trp.shape[0])], dim=0)[:,0]  for _trp in triple_input]

        # rel
        relation_input = [torch.stack([_trp[i,:] for i in range(_trp.shape[0])], dim=0)[:,1]  for _trp in triple_input]

        # obj
        object_input = [torch.stack([_trp[i,:] for i in range(_trp.shape[0])], dim=0)[:,2]  for _trp in triple_input]


        # graph info: label corresponding to (sub, rel, obj)
        label_input = []
        for i_idx in range(idx, idx + self.input_steps):
            label_input.append(torch.stack([self.get_label(trp['label']) for trp in self.triples[i_idx]], dim=0))

        # pred graph info: (sub, rel, obj)
        triple_tar = []
        for t_idx in range(idx + self.input_steps + self.delta_steps,
                           idx + self.input_steps + self.delta_steps + self.target_steps):
            triple_tar.append(torch.tensor([list(trp['triple']) for trp in self.triples[t_idx]]))

        # sub
        subject_tar = [torch.stack([_trp[i, :] for i in range(_trp.shape[0])], dim=0)[:, 0] for _trp in triple_tar]

        # rel
        relation_tar = [torch.stack([_trp[i, :] for i in range(_trp.shape[0])], dim=0)[:, 1] for _trp in triple_tar]

        # obj
        object_tar = [torch.stack([_trp[i, :] for i in range(_trp.shape[0])], dim=0)[:, 2] for _trp in triple_tar]


        # pred graph info: label corresponding to (sub, rel, obj)
        label_tar = []
        for t_idx in range(idx + self.input_steps + self.delta_steps,
                           idx + self.input_steps + self.delta_steps + self.target_steps):
            label_tar.append(torch.stack([self.get_label(trp['label']) for trp in self.triples[t_idx]], dim=0))

        # input timestamps
        input_time_stamps = []
        for i_idx in range(idx, idx + self.input_steps):
            input_time_stamps.append(self.time_stamps[i_idx])


        # edge information
        edge_index_list = []
        edge_type_list = []
        for i_idx in range(idx, idx + self.input_steps):
            edge_index_list.append(self.adjs[i_idx]['edge_index'])
            edge_type_list.append(self.adjs[i_idx]['edge_type'])

        # adjacency tensor ('mtx' means matrix, we preserve this name)
        adj_mtx_list = []
        if self.input_steps != 1:
            for i_idx in range(idx, idx + self.input_steps):
                if i_idx == (idx + self.input_steps - 1):
                    adj_mtx_list.append(adj_mtx_list[-1])
                else:
                    adj_mtx_list.append(self.adjlist[i_idx + 1] - self.adjlist[i_idx])
        # so2r
        so2r_list = []
        for i_idx in range(idx, idx + self.input_steps):
            so2r_list.append(self.so2r[i_idx])

        edge_id_jump, edge_w_jump, rel_jump = [], [], []
        if self.p.jump:
            # jump relation
            if self.p.rel_jump:
                for i, a in enumerate(adj_mtx_list):
                    if i != len(adj_mtx_list) - 1:
                        jumped = torch.nonzero(a._values())
                        edge_id_jump.append(torch.cat([a._indices()[:, jumped][0], a._indices()[:, jumped][2]], dim=1).t())
                        edge_w_jump.append(a._values()[jumped])
                        rel_jump.append(a._indices()[:, jumped][1].squeeze(1))
                    else:
                        edge_id_jump.append(edge_id_jump[-1])
                        edge_w_jump.append(edge_w_jump[-1])
                        rel_jump.append(rel_jump[-1])
            else:
                for a in adj_mtx_list:
                    jumped = torch.nonzero(a._values()).squeeze(1)
                    edge_id_jump.append(a._indices()[:, jumped])
                    edge_w_jump.append(a._values()[jumped].unsqueeze(-1))

        return (subject_input, relation_input, object_input, label_input, subject_tar, relation_tar, object_tar,
                label_tar, target_time_stamps, input_time_stamps, edge_index_list, edge_type_list, adj_mtx_list,
                edge_w_jump, edge_id_jump, rel_jump)

    def __len__(self):
        return self.len

    def get_label(self, label):
        y = np.zeros([self.num_e], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

class TANGOtestDataset(torch.utils.data.Dataset):
    def __init__(self,
                 params,
                 triples: list, # triples['train']
                 adjs: list, # {'edge_index': tensor, 'edge_type': tensor}
                 adjlist: list, # [adjmtx,...,adjmtx], adjmtx is torch sparse tensor
                 so2r: list,
                 num_e: int,
                 input_steps: int,
                 target_steps: int,
                 delta_steps: int = 0,
                 time_stamps: list = None,
                 t_indep_trp: dict = None,
                 num_samp=None,
                 induct_tar=None):

        assert isinstance(triples, list)
        self.p = params
        self.num_e = num_e
        self.num_samp = num_samp
        self.input_steps = input_steps
        self.target_steps = target_steps
        self.delta_steps = delta_steps
        self.triples = triples
        self.adjs = adjs
        self.so2r = so2r
        self.t_indep_trp = t_indep_trp
        self.induct_tar = induct_tar

        self.len = len(self.triples) - self.input_steps - self.target_steps - self.delta_steps + 1

        assert len(triples) == len(time_stamps), "length of time stamps do not match with trajectories"
        self.time_stamps = time_stamps
        self.adjlist = adjlist


    def __getitem__(self, idx):
        # target timestamps
        target_time_stamps = []
        for t_idx in range(idx + self.input_steps + self.delta_steps,
                           idx + self.input_steps + self.delta_steps + self.target_steps):
            target_time_stamps.append(self.time_stamps[t_idx])

        # graph info: (sub, rel, obj)
        triple_input = []
        for i_idx in range(idx, idx + self.input_steps):
            triple_input.append(torch.tensor([list(trp['triple']) for trp in self.triples[i_idx]]))

        # sub
        subject_input = [torch.stack([_trp[i,:] for i in range(_trp.shape[0])], dim=0)[:,0]  for _trp in triple_input]

        # rel
        relation_input = [torch.stack([_trp[i,:] for i in range(_trp.shape[0])], dim=0)[:,1]  for _trp in triple_input]

        # obj
        object_input = [torch.stack([_trp[i,:] for i in range(_trp.shape[0])], dim=0)[:,2]  for _trp in triple_input]

        # graph info: label corresponding to (sub, rel, obj)
        label_input = []
        for i_idx in range(idx, idx + self.input_steps):
            label_input.append(torch.stack([self.get_label(trp['label']) for trp in self.triples[i_idx]], dim=0))

        if self.induct_tar == None:
            # pred graph info: (sub, rel, obj)
            triple_tar = []
            for t_idx in range(idx + self.input_steps + self.delta_steps,
                               idx + self.input_steps + self.delta_steps + self.target_steps):
                triple_tar.append(torch.tensor([list(trp['triple']) for trp in self.triples[t_idx]]))

            # sub
            subject_tar = [torch.stack([_trp[i, :] for i in range(_trp.shape[0])], dim=0)[:, 0] for _trp in triple_tar]

            # rel
            relation_tar = [torch.stack([_trp[i, :] for i in range(_trp.shape[0])], dim=0)[:, 1] for _trp in triple_tar]

            # obj
            object_tar = [torch.stack([_trp[i, :] for i in range(_trp.shape[0])], dim=0)[:, 2] for _trp in triple_tar]


            # pred graph info: label corresponding to (sub, rel, obj)
            label_tar = []
            for t_idx in range(idx + self.input_steps + self.delta_steps,
                               idx + self.input_steps + self.delta_steps + self.target_steps):
                label_tar.append(torch.stack([self.get_label(trp['label']) for trp in self.triples[t_idx]], dim=0))
        else:
            # pred graph info: (sub, rel, obj)
            triple_tar = []
            for t_idx in range(idx + self.input_steps + self.delta_steps,
                               idx + self.input_steps + self.delta_steps + self.target_steps):
                    triple_tar.append(torch.tensor([list(trp['triple']) for trp in self.induct_tar[t_idx]]))
            if len(self.induct_tar[t_idx]) == 0:
                subject_tar, relation_tar, object_tar, label_tar = [], [], [], []
            else:
                # sub
                subject_tar = [torch.stack([_trp[i, :] for i in range(_trp.shape[0])], dim=0)[:, 0] for _trp in triple_tar]

                # rel
                relation_tar = [torch.stack([_trp[i, :] for i in range(_trp.shape[0])], dim=0)[:, 1] for _trp in triple_tar]

                # obj
                object_tar = [torch.stack([_trp[i, :] for i in range(_trp.shape[0])], dim=0)[:, 2] for _trp in triple_tar]

                # pred graph info: label corresponding to (sub, rel, obj)
                label_tar = []
                for t_idx in range(idx + self.input_steps + self.delta_steps,
                                   idx + self.input_steps + self.delta_steps + self.target_steps):
                    label_tar.append(torch.stack([self.get_label(trp['label']) for trp in self.induct_tar[t_idx]], dim=0))


        # input timestamps
        input_time_stamps = []
        for i_idx in range(idx, idx + self.input_steps):
            input_time_stamps.append(self.time_stamps[i_idx])

        # edge information
        edge_index_list = []
        edge_type_list = []
        for i_idx in range(idx, idx + self.input_steps):
            edge_index_list.append(self.adjs[i_idx]['edge_index'])
            edge_type_list.append(self.adjs[i_idx]['edge_type'])

        # time independent label
        indep_lab = []
        for t_idx in range(idx + self.input_steps + self.delta_steps,
                           idx + self.input_steps + self.delta_steps + self.target_steps):
            indep_lab.append(torch.stack([self.get_label(self.t_indep_trp[(trp['triple'][0], trp['triple'][1])]) for trp in self.triples[t_idx]], dim=0))


        # adjacency tensor
        adj_mtx_list = []
        if self.input_steps != 1:
            for i_idx in range(idx, idx + self.input_steps):
                if i_idx == (idx + self.input_steps - 1):
                    adj_mtx_list.append(adj_mtx_list[-1])
                else:
                    adj_mtx_list.append(self.adjlist[i_idx + 1] - self.adjlist[i_idx])

        # so2r
        so2r_list = []
        for i_idx in range(idx, idx + self.input_steps):
            so2r_list.append(self.so2r[i_idx])

        edge_id_jump, edge_w_jump, rel_jump = [], [], []
        if self.p.jump:
            # jump relation
            if self.p.rel_jump:
                for i, a in enumerate(adj_mtx_list):
                    if i != len(adj_mtx_list) - 1:
                        jumped = torch.nonzero(a._values())
                        edge_id_jump.append(torch.cat([a._indices()[:, jumped][0], a._indices()[:, jumped][2]], dim=1).t())
                        edge_w_jump.append(a._values()[jumped])
                        rel_jump.append(a._indices()[:, jumped][1].squeeze(1))
                        #print(rel_jump[-1].shape)
                    else:
                        edge_id_jump.append(edge_id_jump[-1])
                        edge_w_jump.append(edge_w_jump[-1])
                        rel_jump.append(rel_jump[-1])
            else:
                for a in adj_mtx_list:
                    jumped = torch.nonzero(a._values()).squeeze(1)
                    edge_id_jump.append(a._indices()[:, jumped])
                    edge_w_jump.append(a._values()[jumped].unsqueeze(-1))

        return (subject_input, relation_input, object_input, label_input, subject_tar, relation_tar, object_tar,
                label_tar, target_time_stamps, input_time_stamps, edge_index_list, edge_type_list, indep_lab,
                adj_mtx_list, edge_w_jump, edge_id_jump, rel_jump)

    def __len__(self):
        return self.len

    def get_label(self, label):
        y = np.zeros([self.num_e], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

class TANGOtrainDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(TANGOtrainDataLoader, self).__init__(*args, **kwargs)


    def collate_fn(self, batch):
        for item in batch:
            sub_in = item[0]
            rel_in = item[1]
            obj_in = item[2]
            lab_in = item[3]
            sub_tar = item[4]
            rel_tar = item[5]
            obj_tar = item[6]
            lab_tar = item[7]
            tar_ts = item[8]
            in_ts = item[9]
            edg_id = item[10]
            edg_typ = item[11]
            adj_mtx = item[12]
            edg_jump_w = item[13]
            edg_jump_id = item[14]
            rel_jump = item[15]

        return (sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, tar_ts, in_ts, edg_id, edg_typ,
                adj_mtx, edg_jump_w, edg_jump_id, rel_jump)


class TANGOtestDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(TANGOtestDataLoader, self).__init__(*args, **kwargs)


    def collate_fn(self, batch):
        for item in batch:
            sub_in = item[0]
            rel_in = item[1]
            obj_in = item[2]
            lab_in = item[3]
            sub_tar = item[4]
            rel_tar = item[5]
            obj_tar = item[6]
            lab_tar = item[7]
            tar_ts = item[8]
            in_ts = item[9]
            edg_id = item[10]
            edg_typ = item[11]
            indep_lab = item[12]
            adj_mtx = item[13]
            edg_jump_w = item[14]
            edg_jump_id = item[15]
            rel_jump = item[16]

        return (sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, tar_ts, in_ts, edg_id, edg_typ,
                indep_lab, adj_mtx, edg_jump_w, edg_jump_id, rel_jump)