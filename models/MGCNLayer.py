import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from torch_scatter import scatter_add
from .message_passing import MessagePassing

class MGCNConvLayer(MessagePassing):
	def __init__(self, in_channels, out_channels, act=lambda x: x, params=None, isjump=False, diag=False):
		super(self.__class__, self).__init__()

		self.p = params
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.act = act
		self.device = None
		self.diag = diag

		if self.diag:
			self.w = get_param((1, out_channels))
			self.w_rel = get_param((1, out_channels))
		else:
			self.w = get_param((in_channels, out_channels))
			self.w_rel = get_param((in_channels, out_channels))  # for custom rgcn layer

		self.drop = torch.nn.Dropout(self.p.dropout)
		self.bn = torch.nn.BatchNorm1d(out_channels)

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, edge_index, edge_type, num_e, dN=None):
		if self.device is None:
			self.device = self.p.device

		ent_emb = x[:num_e,:]
		rel_embed = x[num_e:,:]
		self.norm = self.compute_norm(edge_index, num_e)
		res = self.propagate('add', edge_index, edge_type=edge_type, rel_embed=rel_embed, x=ent_emb, edge_norm=self.norm, dN=dN)
		out = self.drop(res)

		if self.p.bias: out = out + self.bias
		out = self.bn(out)

		# if self.diag:
		# 	return torch.cat([self.act(out), rel_embed * self.w_rel], dim=0)
		# else:
		# 	return torch.cat([self.act(out), torch.matmul(rel_embed, self.w_rel)], dim=0)
		return torch.cat([self.act(out), rel_embed], dim=0)

	def forward_jump(self, x, edge_index, rel_jump, num_e, dN):
		if self.device is None:
			self.device = self.p.device

		ent_emb = x[:num_e, :]
		rel_embed = x[num_e:, :]

		self.norm = self.compute_norm(edge_index, num_e)
		res = self.propagate_jump('add', edge_index, rel_jump=rel_jump, rel_embed=rel_embed, x=ent_emb,
							 edge_norm=self.norm, dN=dN)
		out = self.drop(res)

		if self.p.bias: out = out + self.bias
		out = self.bn(out)

		# return torch.cat([self.act(out), torch.matmul(rel_embed, self.w_rel)], dim=0)
		return torch.cat([self.act(out), rel_embed], dim=0)

	def message(self, x_j, edge_type, rel_embed, edge_norm, dN):
		weight 	= self.w
		rel_emb = torch.index_select(rel_embed, 0, edge_type)
		if self.p.opn == 'None':
			xj_rel = x_j
		else:
			xj_rel  = self.rel_transform(x_j, rel_emb)
		if dN is not None:
			if self.diag:
				out = xj_rel * weight * dN
			else:
				out = torch.mm(xj_rel, weight) * dN
		else:
			if self.diag:
				out = xj_rel * weight
			else:
				out = torch.mm(xj_rel, weight)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def message_jump(self, x_j, rel_jump, rel_embed, edge_norm, dN):
		weight 	= self.w
		rel_emb = []

		for r_j in rel_jump:
			rel_emb.append(torch.mean(torch.index_select(rel_embed, 0, r_j), dim=0, keepdim=True))
		rel_emb = torch.stack(rel_emb, 0).squeeze(1)
		if self.p.opn == 'None':
			xj_rel = x_j
		else:
			xj_rel  = self.rel_transform(x_j, rel_emb)
		if dN is not None:
			out = torch.mm(xj_rel, weight) * dN
		else:
			out = torch.mm(xj_rel, weight)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

def get_param(shape):
	param = torch.nn.Parameter(torch.Tensor(*shape));
	xavier_normal_(param.data)
	return param