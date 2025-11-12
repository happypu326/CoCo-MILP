import pickle

import torch
import torch_geometric
import numpy as np
    
class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def process_sample(self,filepath):
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        BG = bgData
        varNames = solData['var_names']

        sols = solData['sols'][:50]#[0:300]
        objs = solData['objs'][:50]#[0:300]

        sols=np.round(sols,0)
        return BG, sols, objs, varNames


    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index])

        A, v_map, v_nodes, c_nodes, b_vars=BG

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features =A._values().unsqueeze(1)
        edge_features=torch.ones(edge_features.shape)
        
        constraint_features[torch.isnan(constraint_features)] = 1

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features.cpu()),
            torch.LongTensor(edge_indices.cpu()),
            torch.FloatTensor(edge_features.cpu()),
            torch.FloatTensor(variable_features.cpu())
        )
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)

        graph.objVals = torch.FloatTensor(objs)
        graph.nsols = sols.shape[0]
        graph.ntvars = variable_features.shape[0]
        graph.ntcons = constraint_features.shape[0]
        graph.varNames = varNames
        varname_dict={}
        varname_map=[]
        i=0
        for iter in varNames:
            varname_dict[iter]=i
            i+=1
        for iter in v_map:
            varname_map.append(varname_dict[iter])


        varname_map=torch.tensor(varname_map)

        graph.varInds = [[varname_map],[b_vars]]
        graph.n_constraints = constraint_features.shape[0]

        return graph

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features



    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
