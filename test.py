import pickle
import argparse
import random
import os
import logging
import torch

from utils import get_a_new2, TASKS
from solver.solver_utils import SOLVER_CLASSES
from gnn import GNNPolicy
from multiprocessing import Process, Queue


def setup_environment(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def configure_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(os.path.join(log_dir, "test.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_pretrained_model(args: argparse.Namespace, model_path: str, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = GNNPolicy(
        emb_size=args.emb_size,
        cons_nfeats=args.cons_nfeats,
        edge_nfeats=args.edge_nfeats,
        var_nfeats=args.var_nfeats,
        depth=args.depth,
        Intra_Constraint_Competitive=args.Intra_Constraint_Competitive
    ).to(device)
    
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    return model


def process_single_instance(args, ins_path, policy, device):
    A, v_map, v_nodes, c_nodes, b_vars=get_a_new2(ins_path)

    constraint_features = c_nodes.cpu()
    mask = torch.isnan(constraint_features)
    constraint_features[mask] = 1
    variable_features = v_nodes
    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1)
    edge_features=torch.ones(edge_features.shape)

    constraint_features_batch = torch.tensor([0]*len(constraint_features)).to(device)
    variable_features_batch = torch.tensor([0]*len(variable_features)).to(device)

    BD = policy(
        constraint_features.to(device),
        edge_indices.to(device),
        edge_features.to(device),
        variable_features.to(device),
        torch.tensor([constraint_features.shape[0]]).to(device),
        constraint_features_batch.to(device),
        variable_features_batch.to(device)
    )
    BD = BD.sigmoid().cpu().squeeze()

    # 对齐GNN输出和求解器之间的变量名
    all_varname=[]
    for name in v_map:
        all_varname.append(name)
    binary_name=[all_varname[i] for i in b_vars]
    
    # get a list of (index, VariableName, Prob, -1, type)
    scores=[]
    for i in range(len(v_map)):
        type="C"
        if all_varname[i] in binary_name:
            type='BINARY'
        scores.append([i, all_varname[i], BD[i].item(), -1, type])

    scores.sort(key=lambda x:x[2],reverse=True)

    scores=[x for x in scores if x[4]=='BINARY']
    return scores

def fix_pas(scores, task, args):
    # default hyperparameters {"IP": (60, 35, 55), "WA": (20, 200, 100), "CA": (400, 0, 40), "SC" : (1000, 0, 200)} 
    k0 = int(args.k0)
    k1 = int(args.k1)
    delta = int(args.delta)

    # fixing variable picked by confidence scores
    scores.sort(key=lambda x: x[2], reverse=True)
    for i in range(min(len(scores), k1)):
        scores[i][3] = 1

    scores.sort(key=lambda x: x[2], reverse=False)
    for i in range(min(len(scores), k0)):
        scores[i][3] = 0

    return scores, delta


def solve_mps(mps_file, log_dir, save_name, ins_name, scores, task, args):
    log_file = log_dir
    solver = SOLVER_CLASSES[args.solver]()
    solver.hide_output_to_console()

    solver.load_model(mps_file)
    solver.set_aggressive()

    scores, delta = fix_pas(scores, task, args)
    # trust region method implemented by adding constraints
    instance_variables = solver.get_vars()
    instance_variables.sort(key=lambda v: solver.varname(v))

    # Create a map from varname string to solver's variable object
    variables_map = {}
    for v in instance_variables:  
        variables_map[solver.varname(v)] = v

    alphas = []

    for i in range(len(scores)):
        tar_var = variables_map[scores[i][1]]
        x_star = scores[i][3]  # 1, 0, or -1 (don't fix)
        if x_star < 0:
            continue

        tmp_var = solver.create_real_var(name=f'alpha_{tar_var}')
        alphas.append(tmp_var)
        solver.add_constraint(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
        solver.add_constraint(tmp_var >= x_star - tar_var, name=f'alpha_down_{i}')

    if len(alphas) > 0:
        all_tmp = 0
        for tmp in alphas:
            all_tmp += tmp
        solver.add_constraint(all_tmp <= delta, name="sum_alpha")

    results = solver.solve(means=args.solver, log_file=log_file, time_limit=args.max_time, threads=args.threads)
    sol_save_path = os.path.join(os.path.dirname(log_dir), save_name + ins_name.split('.')[0] + '_node_info.pkl')
    with open(sol_save_path, 'wb') as f:
        pickle.dump(results, f)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="multitest for CoCo.")
    
    exp_group = parser.add_argument_group("Experiment Settings")
    exp_group.add_argument("--test_problem_type", type=str, choices=TASKS, default='SC',help="Problem type to train on (e.g., CA, WA, IP)")
    
    exp_group.add_argument("--test_num", type=int, default=100,
                         help="Number of test instances to process")
    
    model_group = parser.add_argument_group("Model Settings")

    model_group.add_argument("--model_dir", default="./pretrain_models",
                           help="Directory containing pretrained models")
    
    gcn_group = parser.add_argument_group("GCN Settings")
    gcn_group.add_argument("--emb_size", type=int, default=64,
                       help="Embedding size for GNN (default: %(default)s)")
    gcn_group.add_argument("--cons_nfeats", type=int, default=4, 
                       help="Number of features for constraint nodes (default: %(default)s)")
    gcn_group.add_argument("--edge_nfeats", type=int, default=1, 
                       help="Number of features for edge (default: %(default)s)")
    gcn_group.add_argument("--var_nfeats", type=int, default=6, 
                       help="Number of features for variable nodes (default: %(default)s)")
    parser.add_argument("--depth", type=int, default=2)
    
    solver_group = parser.add_argument_group("Solver Settings")
    solver_group.add_argument("--solver", choices=SOLVER_CLASSES.keys(), default="gurobi",
                            help="MILP solver implementation (default: %(default)s)")
    solver_group.add_argument("--max_time", type=int, default=1000,
                            help="Maximum solving time in seconds")
    solver_group.add_argument("--threads", type=int, default=1,
                            help="Number of threads for solving")
 
    sys_group = parser.add_argument_group("System Settings")
    sys_group.add_argument("--instance_dir", default="./instance/test",
                         help="Path to test instances directory")
    sys_group.add_argument("--scores_dir", default="./scores",
                         help="Path to store scores directory")
    sys_group.add_argument("--device", default="cuda:0",
                         help="Computation device (default: %(default)s)")
    sys_group.add_argument("--num_workers", type=int, default=1,
                         help="Number of parallel workers (default: %(default)s)")
    sys_group.add_argument("--log_dir", default="./test_logs/",
                         help="Path to test instances directory")
    
    parser.add_argument('--Intra_Constraint_Competitive', default=False, action='store_true')

    parser.add_argument("--margin",type=float,default=0.9)
    parser.add_argument("--alpha",type=float,default=0.01)
    parser.add_argument("--tao",type=float,default=0.1)

    parser.add_argument("--k0", type=int, default=1000)
    parser.add_argument("--k1", type=int, default=0)
    parser.add_argument("--delta", type=int, default=200)

    
    return parser.parse_args()


def worker_process(task_queue, args, scores_dir, log_dir, save_name):
    logger = configure_logging(log_dir=log_dir) 
    
    while True:
        ins_name = task_queue.get()
        if ins_name is None: 
            break
        
        ins_path = os.path.join(args.instance_dir, args.test_problem_type, ins_name)
        log_path = os.path.join(log_dir, f"{save_name}_{ins_name.split('.')[0]}.log")
        score_path = os.path.join(scores_dir, f"scores_{ins_name.split('.')[0]}.pkl")
        with open(score_path, "rb") as f:
            scores = pickle.load(f)
        
        logger.info(f"Start solving {ins_name} with {args.solver}")
        solve_mps(ins_path, log_path, save_name, ins_name, scores, args.test_problem_type, args)


def main():
    setup_environment()
    args = parse_arguments()
    
    save_name = f'Intra_Constraint_Competitive_{args.Intra_Constraint_Competitive}_margin_{args.margin}_alpha_{args.alpha}_tao_{args.tao}_k0_{args.k0}_k1_{args.k1}_delta_{args.delta}'

    model_path = os.path.join(args.model_dir, args.test_problem_type, f"Intra_Constraint_Competitive_{args.Intra_Constraint_Competitive}_margin_{args.margin}_alpha_{args.alpha}_tao_{args.tao}_model_best.pth")
    policy = load_pretrained_model(args, model_path, args.device) 
    

    log_dir = os.path.join(args.log_dir, args.test_problem_type, 
                         f"{save_name}")
    os.makedirs(log_dir, exist_ok=True)

    scores_dir = os.path.join(args.scores_dir, args.test_problem_type, 
                         f"Intra_Constraint_Competitive_{args.Intra_Constraint_Competitive}_margin_{args.margin}_alpha_{args.alpha}_tao_{args.tao}")
    os.makedirs(scores_dir, exist_ok=True)

    test_instances = sorted(os.listdir(os.path.join(args.instance_dir, args.test_problem_type)))
    for ins_name in test_instances[:args.test_num]:
        ins_path = os.path.join(args.instance_dir, args.test_problem_type, ins_name)
        file_path = os.path.join(scores_dir, f"scores_{ins_name.split('.')[0]}.pkl")
        if os.path.exists(file_path):
            continue

        scores = process_single_instance(args, ins_path, policy, args.device)
        with open(file_path, 'wb') as f:
            pickle.dump(scores, f)

    task_queue = Queue()
    
    for ins_name in test_instances[:args.test_num]:
        task_queue.put(ins_name)
    
    num_workers = args.num_workers
    
    for _ in range(num_workers):
        task_queue.put(None)
    
    processes = []
    for _ in range(num_workers):
        p = Process(
            target=worker_process,
            args=(task_queue, args, scores_dir, log_dir, save_name) 
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("Testing completed successfully.")


if __name__ == "__main__":
    main()