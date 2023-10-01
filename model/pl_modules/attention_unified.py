import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def transform_to_batch(frac_coords, atom_types, lattice, num_atoms):
    max = torch.max(num_atoms)
    
    frac_coords_tensor = torch.empty(0).to(frac_coords.device)
    atom_type_tensor = torch.empty(0).to(atom_types.device)
    lattice_tensor = torch.empty(0).to(lattice.device)

    ones = torch.ones((num_atoms.size(0), max), device = frac_coords.device)
    mask = torch.empty(0).to(atom_types.device)

    sum = 0
    for i in range(len(num_atoms)):
        value = int(num_atoms[i])
        frac_coords_tensor = torch.concat(
            [frac_coords_tensor, F.pad(frac_coords[sum:sum + value , :], (0,0,0,max-num_atoms[i]), mode='constant',value=0).unsqueeze(0)])
        atom_type_tensor = torch.concat(
            [atom_type_tensor, F.pad(atom_types[sum:sum + value , :], (0,0,0,max-num_atoms[i]), mode='constant',value=0).unsqueeze(0)])
        lattice_tensor = torch.concat(
            [lattice_tensor, F.pad(lattice[sum:sum + value, :], (0,0,0,max-num_atoms[i]), mode='constant',value=0).unsqueeze(0)])
        mask_new = F.pad(ones[i,0:value], (0,max-value), mode='constant',value=0).unsqueeze(0)
        mask = torch.concat([mask, mask_new])
        
        sum += value

    mask = mask == 0
    
    return frac_coords_tensor, atom_type_tensor, lattice_tensor, mask

def transform_from_batch(frac_coords, atom_types, lattice, num_atoms):

    frac_coords_tensor = torch.empty(0).to(frac_coords.device)
    atom_type_tensor = torch.empty(0).to(atom_types.device)
    lattice_tensor = torch.empty(0).to(lattice.device)

    for coord, type, l, num_atom in zip(frac_coords, atom_types, lattice, num_atoms):
        frac_coords_tensor = torch.concat(
            [frac_coords_tensor, coord[0:num_atom, :]])
        atom_type_tensor = torch.concat(
            [atom_type_tensor, type[0:num_atom]])
        lattice_tensor = torch.concat(
            [lattice_tensor, l[0:num_atom, :]])
    
    return frac_coords_tensor, atom_type_tensor, lattice_tensor



def transform_to_batch_singular(data, num_atoms):
    max = torch.max(num_atoms)

    batched = torch.empty(0).to(data.device)

    sum = 0
    for i in range(len(num_atoms)):
        value = int(num_atoms[i])
        batched = torch.concat(
            [batched, F.pad(data[sum:sum + value , :], (0,0,0,max-num_atoms[i]), mode='constant',value=0).unsqueeze(0)])
        
        sum += value

    return batched

def transform_from_batch_singular(batched, num_atoms):
    joint_tensor = torch.empty(0).to(batched.device)

    for crystal, num_atom in zip(batched, num_atoms):
        joint_tensor = torch.concat(
            [joint_tensor, crystal[0:num_atom, :]])
    
    return joint_tensor



def get_L_features(L, num_atoms):
    L = L.repeat_interleave(num_atoms,dim=0)
    return L


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x






class AttentionSinCos_Layer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_attention_heads_complex,
        w_in,
        kdim,
        vdim,
        # hidden_dim,
        batch_first = True,
        norm = False,
        residual_complex = False,
        useSwiGLU = False,
        mlp=False,

    ):
        super(AttentionSinCos_Layer, self).__init__()

        self.norm = norm
        self.residual_complex = residual_complex
        self.mlp = mlp
        
        self.activation = nn.SiLU() 
        self.num_attention_heads_complex = num_attention_heads_complex


        self.Wq = nn.Linear(3, 3*num_attention_heads_complex, bias=False) 
        self.Wk = nn.Linear(3, 3*num_attention_heads_complex, bias=False)
        self.Wv = nn.Linear(3, 3*num_attention_heads_complex, bias=False) 
        
        W = torch.empty(0)
        for i in range(self.num_attention_heads_complex):
            W = torch.concat([W, torch.eye(3, 3) * (i+1)], dim = 0)

        self.Wq.weight = nn.Parameter(W, requires_grad=w_in)
        self.Wk.weight = nn.Parameter(W, requires_grad=w_in)
        self.Wv.weight = nn.Parameter(W, requires_grad=w_in)

        self.Wq_out = nn.Linear(input_dim, input_dim) 
        self.Wk_out = nn.Linear(kdim, kdim)
        self.Wv_out = nn.Linear(vdim, vdim)

        self.attention = nn.MultiheadAttention(input_dim, num_heads=num_attention_heads_complex,  
                                    kdim=kdim, vdim=vdim,
                                    batch_first = batch_first)

        if self.mlp:
            self.layer_norm_1 = nn.LayerNorm(input_dim)
            self.layer_norm_2 = nn.LayerNorm(input_dim)

            
            if useSwiGLU:
                l1 = nn.Linear(input_dim, 2 * 4 * input_dim)
                l2 = nn.Linear( 4 * input_dim, input_dim)
                self.mlp = nn.Sequential(
                    l1,
                    SwiGLU(),
                    l2
                )
            else:
                l1 = nn.Linear(input_dim,  4 * input_dim)
                l2 = nn.Linear( 4 * input_dim, input_dim)
                self.mlp = nn.Sequential(
                    l1,
                    nn.SiLU(),
                    l2
                )
    
    
    
    def periodic_w(self, frac, k):
        FT = torch.zeros(frac.size(0), frac.size(1),2*k*3).to(frac.device)
        
        FT[:,:, 0:(2*k*3-1):2] = torch.sin(2 * np.pi * frac )
        FT[:,:, 1:(2*k*3):2] = torch.cos(2 * np.pi * frac)

        return FT

    def forward(self, frac_coords, atoms, L, mask):

        q = self.Wq(frac_coords)
        k = self.Wk(frac_coords)
        v = self.Wv(frac_coords)
        
        q = self.periodic_w(q, k = self.num_attention_heads_complex)
        k = self.periodic_w(k, k = self.num_attention_heads_complex)
        v = self.periodic_w(v, k = self.num_attention_heads_complex)


        q = torch.concat([atoms, q],dim=2)
        k = torch.concat([atoms, k, L],dim=2)
        v = torch.concat([atoms, v, L],dim=2)

        q = self.Wq_out(q)
        k = self.Wk_out(k)
        v = self.Wv_out(v)

            
        result, _ = self.attention(q, k, v, key_padding_mask = mask)

        if self.residual_complex:
            x_1 = q + result
            result = x_1
        if self.norm:
            result = self.layer_norm_1(result)

        if self.mlp:
            result = self.mlp(result)

            if self.residual_complex:
                result = x_1 + result
            if self.norm:
                result = self.layer_norm_2(result)

        return result
    



class AttentionMultihead_Layer(nn.Module):
    def __init__(
        self,
        input_dim,
        kdim,
        vdim,
        num_heads,
        hidden_dim=256,
        useSwiGLU = False,
        batch_first = True,
        a = 0.2,
        norm = True,
        mlp = True,
    ):
        super(AttentionMultihead_Layer, self).__init__()

        self.hidden_dim = hidden_dim
        self.norm = norm
        self.mlp = mlp

        self.Wq = nn.Linear(input_dim, input_dim) 
        self.Wk = nn.Linear(kdim, kdim)
        self.Wv = nn.Linear(vdim, vdim) 
        # nn.init.xavier_uniform_(self.Wk.weight, gain=0.8) #1.414)
        # nn.init.xavier_uniform_(self.Wq.weight, gain=0.8) #1.414)
        # nn.init.xavier_uniform_(self.Wv.weight, gain=0.8) #1.414)

        self.attention = nn.MultiheadAttention(input_dim, kdim=kdim, vdim=vdim, \
                                               num_heads=num_heads,  batch_first = batch_first)

        self.post_layer_norm = nn.LayerNorm(input_dim)

        if useSwiGLU:
            l1 = nn.Linear(input_dim, 2 * hidden_dim)
            l2 = nn.Linear(hidden_dim, input_dim)
            self.mlp = nn.Sequential(
                l1,
                SwiGLU(),
                l2
            )
        else:
            l1 = nn.Linear(input_dim, hidden_dim)
            l2 = nn.Linear(hidden_dim, input_dim)
            self.mlp = nn.Sequential(
                l1,
                nn.SiLU(),
                l2
            )

    def forward(self, q, k, v, mask):
        
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
            
        result, _ = self.attention(q, k, v, key_padding_mask = mask)

        q_1 = q + result
        result = q_1


        if self.mlp:
            if self.norm:
                result = self.post_layer_norm(result)
            result = self.mlp(result)
        result = q_1 + result

        return result




class AttentionSinCos_Separate_jointMLP(nn.Module):
    """Decoder with custom attention."""

    def __init__(
        self,
        skip_coords_network=False,
        
        w_in = True,
        mlp_sin_cos = False,
        norm = True,

        residual_complex = False,
        residual_complex_after = False,

        num_attention_heads_complex = 8,

        separate_embedding_atoms_lattice = False,
        separate_embedding_atoms_dim = 32,
        separate_embedding_lattice_dim = 32,
        
        hidden_dim_final_mlp_joint = 32,
        hidden_dim_final_mlp_lattice = 32,

        useSwiGLU = False,
        atom_type_dim = 128,
        attention_layers = 4,
        num_heads = 4,
        lattice_MLP = True,

        embedding_num_atoms = True,
        output_atom_type_dim = 32,
    ):
        super(AttentionSinCos_Separate_jointMLP, self).__init__()

        self.skip_coords_network = skip_coords_network

        self.num_attention_layers = attention_layers


        self.separate_embedding_atoms_lattice = separate_embedding_atoms_lattice

        self.num_attention_heads_complex = num_attention_heads_complex

        self.num_heads =num_heads


        if self.skip_coords_network:
            self.input_dim_coords = 3
        else:
            self.input_dim_coords = 3 * 2 * self.num_attention_heads_complex
        
        self.output_atom_type_dim = output_atom_type_dim

        if self.separate_embedding_atoms_lattice:

            self.embedding_atom_types = nn.Linear(atom_type_dim, separate_embedding_atoms_dim, bias=False)
            self.embedding_lattice = nn.Linear(6, separate_embedding_lattice_dim, bias=False)


            self.input_dim = separate_embedding_atoms_dim + self.input_dim_coords + separate_embedding_lattice_dim

            self.input_dim_atoms = separate_embedding_atoms_dim
            self.input_dim_lattice = separate_embedding_lattice_dim

        else:
            self.input_dim = atom_type_dim + self.input_dim_coords + 6

            self.input_dim_atoms = atom_type_dim
            self.input_dim_lattice = 6

        
        self.input_dim_nodes = self.input_dim_atoms + self.input_dim_coords

        self.atom_norm = nn.LayerNorm(self.input_dim_atoms)
        self.coord_norm = nn.LayerNorm(self.input_dim_coords)
        self.lattice_norm = nn.LayerNorm(self.input_dim_lattice)
            
        self.embedding_num_atoms = embedding_num_atoms
        if self.embedding_num_atoms:
            self.input_dim += 1
            self.input_dim_lattice += 1

        a = 0.2
        activation = nn.SiLU() 

        self.residual_complex_after = residual_complex_after



        if not self.skip_coords_network:

            self.attention_sincos = AttentionSinCos_Layer( 
                input_dim = self.input_dim_nodes, 
                kdim = self.input_dim_nodes + self.input_dim_lattice, 
                vdim = self.input_dim_nodes + self.input_dim_lattice,
                residual_complex = residual_complex,
                num_attention_heads_complex=num_attention_heads_complex,
                w_in=w_in,
                mlp=mlp_sin_cos,
                useSwiGLU=useSwiGLU)

            

        self.attention_nodes = nn.ModuleList([
            AttentionMultihead_Layer(input_dim = self.input_dim_nodes, 
                num_heads = self.num_heads, # attention_nodes=True,
                kdim = self.input_dim_nodes + self.input_dim_lattice, 
                vdim = self.input_dim_nodes + self.input_dim_lattice,
                useSwiGLU = useSwiGLU,
                hidden_dim= 4*self.input_dim_nodes, #self.hidden_dim_attention_mlp,
                a = a,
                norm = norm) for _ in range(self.num_attention_layers)])

        self.pre_norm_attention_nodes_nodes = nn.ModuleList([
            nn.LayerNorm(self.input_dim_nodes) for _ in range(self.num_attention_layers)])
        self.pre_norm_attention_nodes_lattice = nn.ModuleList([
            nn.LayerNorm(self.input_dim_lattice) for _ in range(self.num_attention_layers)])
                        
        self.attention_lattice = nn.ModuleList([
            AttentionMultihead_Layer(input_dim = self.input_dim_lattice, 
                num_heads = self.num_heads, # attention_nodes=False,
                kdim = self.input_dim_nodes + self.input_dim_lattice, 
                vdim = self.input_dim_nodes + self.input_dim_lattice,
                useSwiGLU = useSwiGLU,
                hidden_dim= 4*self.input_dim_lattice, #self.hidden_dim_attention_mlp,
                a = a,
                norm = norm) for _ in range(self.num_attention_layers)])
        
        self.pre_norm_attention_lattice_nodes = nn.ModuleList([
            nn.LayerNorm(self.input_dim_nodes) for _ in range(self.num_attention_layers)])
        self.pre_norm_attention_lattice_lattice = nn.ModuleList([
            nn.LayerNorm(self.input_dim_lattice) for _ in range(self.num_attention_layers)])


        l1 = nn.Linear(self.input_dim_nodes, hidden_dim_final_mlp_joint)
        l2 = nn.Linear(hidden_dim_final_mlp_joint,hidden_dim_final_mlp_joint)
        l3 = nn.Linear(hidden_dim_final_mlp_joint,hidden_dim_final_mlp_joint)

        l4 = nn.Linear(hidden_dim_final_mlp_joint, output_atom_type_dim + 3)
        # nn.init.xavier_uniform_(l1.weight, gain=0.8)
        # nn.init.xavier_uniform_(l2.weight, gain=0.8)
        self.joint_final_MLP = nn.Sequential(l1,
                                        activation,
                                        nn.LayerNorm(hidden_dim_final_mlp_joint),
                                        l2,
                                        nn.LayerNorm(hidden_dim_final_mlp_joint),
                                        activation,
                                        l3,
                                        nn.LayerNorm(hidden_dim_final_mlp_joint),
                                        activation,
                                        l4)
        if lattice_MLP:
            l1 = nn.Linear(self.input_dim_lattice,hidden_dim_final_mlp_lattice)
            l2 = nn.Linear(hidden_dim_final_mlp_lattice,6)
            # nn.init.xavier_uniform_(l1.weight, gain=0.8)
            # nn.init.xavier_uniform_(l2.weight, gain=0.8)
            self.lattice = nn.Sequential(l1,
                                         activation,
                                         l2)
        else:
            self.lattice = nn.Linear(self.input_dim_lattice, 6)

    
    
    def forward(self, atoms, frac_coords, num_atoms, L, t):

        L_features = L
        L_org = get_L_features(L, num_atoms)
        frac_coords, atoms, _, mask = transform_to_batch(frac_coords, atoms, L_org, num_atoms)



        if self.separate_embedding_atoms_lattice:
            atoms = self.embedding_atom_types(atoms)
            L_features = self.embedding_lattice(L_features)


        atoms = self.atom_norm(atoms)
        L_features = self.lattice_norm(L_features)


        if self.embedding_num_atoms:
            L_features = torch.concat([L_features, num_atoms.unsqueeze(1)], dim=1)


        if self.skip_coords_network:
            node_features = torch.concat([atoms, frac_coords], dim=2)

        else:

            L_features_ = L_features.unsqueeze(1)
            L_features_stacked = L_features_.repeat_interleave(torch.max(num_atoms), dim = 1)

            
            node_features = self.attention_sincos(frac_coords, atoms, L_features_stacked, mask)

        L_features = L_features.unsqueeze(1)
            
        for attention_layer_nodes, ln_nodes_nodes, ln_nodes_lattice,\
            attention_layer_lattice, ln_lattice_nodes, ln_lattice_lattice in zip(
            self.attention_nodes, self.pre_norm_attention_nodes_nodes, self.pre_norm_attention_nodes_lattice,
            self.attention_lattice, self.pre_norm_attention_lattice_nodes, self.pre_norm_attention_lattice_lattice):
            
            
            node_features = ln_nodes_nodes(node_features)
            # L_features = ln_nodes_lattice(L_features)
            L_features_stacked = L_features.repeat_interleave(torch.max(num_atoms), dim = 1)

            concat = torch.concat([node_features, L_features_stacked],dim=2)
            L_features = attention_layer_lattice(L_features, concat, concat, mask=mask)

            # node_features = ln_lattice_nodes(node_features)
            L_features = ln_lattice_lattice(L_features)
            L_features_stacked = L_features.repeat_interleave(torch.max(num_atoms), dim = 1)
            concat = torch.concat([node_features, L_features_stacked],dim=2)
            node_features = attention_layer_nodes(node_features, concat, concat, mask=mask)

        L_features = L_features.squeeze(1)



        ### MLP

        node_features = transform_from_batch_singular(node_features, num_atoms)

        
        nodes = self.joint_final_MLP(node_features)
        pred_frac_coord_diff = nodes[:, self.output_atom_type_dim:]
        # pred_frac_coord_diff = torch.tanh(pred_frac_coord_diff)
        pred_atom_types_diff = nodes[:, :self.output_atom_type_dim]
        pred_lattice_diff = self.lattice(L_features)


        return pred_atom_types_diff, pred_frac_coord_diff, pred_lattice_diff
    

