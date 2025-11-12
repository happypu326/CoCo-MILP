import argparse
import os
import torch
import torch_geometric
import random
import time

from utils import TASKS

os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = "pytorch"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from gnn import GNNPolicy
from dataset.graph_dataset import GraphDataset

# Predefined task-specific parameters
TASK_BATCH_SIZE = {'CA': 4, 'WA': 4, 'IP': 4, 'SC': 1}
ENERGY_WEIGHT_NORM = {'CA': -10000, "WA": 100, "IP": 10, 'SC': 100}


def train(args, predict, data_loader, optimizer=None, weight_norm=1, device='cpu'):

    if optimizer:
        predict.train()
    else:
        predict.eval()

    mean_loss = 0.0
    n_samples_processed = 0

    temperature = args.tao
    with torch.set_grad_enabled(optimizer is not None):
        for _, batch in enumerate(data_loader):
            batch = batch.to(device)

            # 1) Extract per-instance target solutions & weights
            solInd = batch.nsols  # number of solutions per MILP instance in current mini‑batch
            target_sols = []      # list of [n_solutions, n_binary_vars] tensors
            target_vals = []      # list of objective values (energy) tensors
            solEndInd = 0
            valEndInd = 0

            for i in range(solInd.shape[0]):
                nvar = len(batch.varInds[i][0][0])
                solStartInd = solEndInd
                solEndInd = solInd[i] * nvar + solStartInd
                valStartInd = valEndInd
                valEndInd = valEndInd + solInd[i]

                sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
                vals = batch.objVals[valStartInd:valEndInd]

                target_sols.append(sols)
                target_vals.append(vals)

            # 2) Forward pass
            constraint_features_batch = torch.repeat_interleave(torch.arange(len(batch.ntcons), device=batch.ntcons.device), batch.ntcons.clone().detach().long())
            variable_features_batch = torch.repeat_interleave(torch.arange(len(batch.ntvars), device=batch.ntvars.device), batch.ntvars.clone().detach().long())

            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10  # sanitize

            BD = predict(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                batch.n_constraints,
                constraint_features_batch,
                variable_features_batch,
            ).sigmoid()  # (total_binary_vars, ) probabilities

            # 3) Multi‑Sample Contrastive Loss
            MSCL_loss = 0.0
            index_arrow = 0  # pointer into BD over concatenated graphs

            for sols, vals, n_var, varInds in zip(
                target_sols,
                target_vals,
                batch.ntvars,
                batch.varInds,
            ):
                # a) per-solution weights
                exp_weight = torch.exp(-vals / weight_norm)
                weight = exp_weight / exp_weight.sum()  # (n_solutions,)

                # b) map back to binary variable positions
                varname_map = varInds[0][0]
                b_vars = varInds[1][0].long()
                sols = sols[:, varname_map][:, b_vars]  # (n_solutions, |B|)

                # c) predicted probs & logits for this MILP instance
                pre_sols = BD[index_arrow : index_arrow + n_var].squeeze()[b_vars]  # (|B|,)
                index_arrow += n_var

                logits = torch.logit(pre_sols.clamp(min=1e-8, max=1 - 1e-8)) / temperature  # (|B|,)
                exp_logits = torch.exp(logits)
                partition = exp_logits.sum()  # scalar Z

                # d) MSCL loss over each feasible solution in this instance
                #    loop over solutions to accumulate weighted loss
                for sol_vec, w_q in zip(sols, weight):
                    pos_mask = sol_vec.bool()
                    if pos_mask.sum() == 0:  # no positive variables in this solution
                        continue
                    numerator = exp_logits[pos_mask].sum()
                    mscl_loss = -w_q * torch.log(numerator / partition)
                    MSCL_loss += mscl_loss

            Rank_loss = 0
            index_arrow = 0
            
            margin = args.margin
            for _,(sols,vals) in enumerate(zip(target_sols,target_vals)):
                n_vals = vals
                exp_weight = torch.exp(-n_vals/weight_norm)
                weight = exp_weight/exp_weight.sum()

                varInds = batch.varInds[_]
                varname_map=varInds[0][0]
                b_vars=varInds[1][0].long()

                sols = sols[:,varname_map][:,b_vars]
                
                n_var = batch.ntvars[_]
                pre_sols = BD[index_arrow:index_arrow + n_var].squeeze()[b_vars]
                index_arrow = index_arrow + n_var
                
                rank_loss = 0
                for sol_idx in range(sols.size(0)):
                    cur_sol = sols[sol_idx]
                    pos_mask = cur_sol == 1
                    neg_mask = cur_sol == 0
                    if not torch.any(pos_mask) or not torch.any(neg_mask):
                        continue
                    
                    pos_scores = pre_sols[pos_mask]
                    neg_scores = pre_sols[neg_mask]
                    
                    diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
                    pair_losses = torch.clamp(margin - diff, min=0)
                    sol_rank_loss = pair_losses.mean()
                    rank_loss += weight[sol_idx] * sol_rank_loss
                    
                Rank_loss += rank_loss
  
            Loss = MSCL_loss + args.alpha*Rank_loss
            
            # 4) Optimisation step & statistics
            if optimizer is not None:
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()

            mean_loss += Loss.item()
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    return mean_loss


def get_parser():
    """Create and return the argument parser with all configuration parameters"""
    parser = argparse.ArgumentParser(description="train for CoCo.")

    # Required parameters
    parser.add_argument("--problem_type", choices=TASKS, default='SC',
                       help="Problem type to train on (e.g., CA, WA, IP)")
    
    # Model parameters
    parser.add_argument("--gnn_type", default='gcn')
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--cons_nfeats", type=int, default=4)
    parser.add_argument("--edge_nfeats", type=int, default=1)
    parser.add_argument("--var_nfeats", type=int, default=6)
    parser.add_argument("--depth", type=int, default=2)
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=0.0001,
                       help="Learning rate for optimizer (default: %(default)s)")
    parser.add_argument("--num_epochs", type=int, default=5000,
                       help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of data loader workers (default: %(default)s)")
    
    # Data and paths
    parser.add_argument("--data_dir", default="./dataset",
                       help="Base directory for dataset (default: %(default)s)")
    parser.add_argument("--model_save_dir", default="./pretrain_models",
                       help="Directory to save models (default: %(default)s)")
    parser.add_argument("--log_save_dir", default="./train_logs",
                       help="Directory to save logs (default: %(default)s)")
    
    # Device configuration
    parser.add_argument("--device",default="cuda:1", help="cuda device")
    parser.add_argument('--Intra_Constraint_Competitive', default=False, action='store_true')

    parser.add_argument("--margin",type=float,default=0.9)
    parser.add_argument("--alpha",type=float,default=0.01)
    parser.add_argument("--tao",type=float,default=0.1)

    return parser


def main():
    # Parse arguments and setup configurations
    parser = get_parser()
    args = parser.parse_args()
    
    # Set device configuration
    DEVICE = args.device
    problem_type = args.problem_type
    
    save_name = f'Intra_Constraint_Competitive_{args.Intra_Constraint_Competitive}_margin_{args.margin}_alpha_{args.alpha}_tao_{args.tao}'

    # Set task-specific defaults
    batch_size = TASK_BATCH_SIZE.get(problem_type, 4)
    weight_norm = ENERGY_WEIGHT_NORM.get(problem_type, 100)
    
    # Create directories
    model_save_path = os.path.join(args.model_save_dir, problem_type)
    log_save_path = os.path.join(args.log_save_dir, problem_type)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    
    # Initialize logging
    log_file = open(f'{log_save_path}/{save_name}_train.log', 'wb')
    
    dir_bg = os.path.join(args.data_dir, problem_type, 'BG')
    dir_sol = os.path.join(args.data_dir, problem_type, 'solution')
    sample_names = os.listdir(dir_bg)
    sample_names = [name for name in sample_names if not name.endswith(".pkl")]
    sample_files = [(os.path.join(dir_bg, name), 
                    os.path.join(dir_sol, name).replace('bg', 'sol')) 
                  for name in sample_names]
    
    random.shuffle(sample_files)
    train_files = sample_files[:int(0.8*len(sample_files))]
    valid_files = sample_files[int(0.8*len(sample_files)):]
    
    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.loader.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = GNNPolicy(
        emb_size=args.emb_size,
        cons_nfeats=args.cons_nfeats,
        edge_nfeats=args.edge_nfeats,
        var_nfeats=args.var_nfeats,
        depth=args.depth,
        Intra_Constraint_Competitive=args.Intra_Constraint_Competitive
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        train_loss = train(args, model, train_loader, optimizer, weight_norm, device=DEVICE)
        valid_loss = train(args, model, valid_loader, None, weight_norm, device=DEVICE)
    
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, f'{save_name}_model_best.pth'))
        
        torch.save(model.state_dict(), os.path.join(model_save_path, f'{save_name}_model_last.pth'))
        
        log_entry =  f'@epoch{epoch}   Train loss:{train_loss}   Valid loss:{valid_loss}    TIME:{time.time() - start_time}\n'
        log_file.write(log_entry.encode())
        log_file.flush()
    
    log_file.close()
    print("Training completed successfully.")


if __name__ == '__main__':
    main()