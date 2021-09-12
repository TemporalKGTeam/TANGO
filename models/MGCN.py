import torch
from torch.nn.parameter import Parameter
from .MGCNLayer import MGCNConvLayer

class MGCNLayerWrapper(torch.nn.Module):
	def __init__(self, edge_index, edge_type, num_e, num_rel, act, drop1, drop2, sub, rel, params=None):
		super().__init__()
		self.nfe = 0
		self.p = params
		self.edge_index = edge_index
		self.edge_type = edge_type
		self.p.hidsize = self.p.embsize if self.p.core_layer == 1 else self.p.hidsize
		self.num_e = num_e
		self.num_rel = num_rel
		self.device = self.p.device
		self.act = act
		self.drop_l1 = torch.nn.Dropout(drop1)
		self.drop_l2 = torch.nn.Dropout(drop2)
		self.sub = sub
		self.rel = rel

		# transition layer
		self.jump = None
		self.jump_weight = None

		# residual layer
		if self.p.res:
			self.res = torch.nn.Parameter(torch.FloatTensor([0.1]))

		# define MGCN Layer
		self.conv1 = MGCNConvLayer(self.p.initsize, self.p.hidsize, act=self.act, params=self.p)
		self.conv2 = MGCNConvLayer(self.p.hidsize, self.p.embsize, act=self.act, params=self.p) if self.p.core_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(num_e)))

	def set_graph(self, edge_index, edge_type):
		self.edge_index = edge_index
		self.edge_type = edge_type

	def set_jumpfunc(self, edge_id_jump, edge_w_jump, jumpfunc, jumpw=None, skip=False, rel_jump=None):
		self.edge_id_jump = edge_id_jump
		self.edge_w_jump = edge_w_jump
		self.jump = jumpfunc
		self.jump_weight = jumpw
		self.skip = skip
		self.rel_jump = rel_jump

	def forward(self, t, emb):
		self.nfe += 1
		jump_emb = emb.clone()
		if self.p.res:
			emb = emb + self.res * self.conv1(emb, self.edge_index, self.edge_type, self.num_e)
			emb = self.drop_l1(emb)
			emb = (emb + self.res * self.conv2(emb, self.edge_index, self.edge_type, self.num_e)) if self.p.core_layer == 2 else emb
			emb = self.drop_l2(emb) if self.p.core_layer == 2 else emb
		else:
			emb	= self.conv1(emb, self.edge_index, self.edge_type, self.num_e)
			emb	= self.drop_l1(emb)
			emb	= self.conv2(emb, self.edge_index, self.edge_type, self.num_e) 	if self.p.core_layer == 2 else emb
			emb	= self.drop_l2(emb) 							if self.p.core_layer == 2 else emb

		if self.p.jump:
			if not self.skip:
				if self.p.rel_jump:
					jump_res = self.jump.forward(jump_emb, self.edge_id_jump, self.rel_jump, self.num_e,
													  dN=self.edge_w_jump)
				else:
					jump_res = self.jump(jump_emb, self.edge_id_jump, dN=self.edge_w_jump)
				emb = emb + self.jump_weight * jump_res
				emb = self.drop_l2(emb)

		return emb