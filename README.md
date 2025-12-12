# CoCo-MILP: Inter-Variable Contrastive and Intra-Constraint Competitive MILP Solution Prediction
This is the code of **CoCo-MILP: Inter-Variable Contrastive and Intra-Constraint Competitive MILP Solution Prediction**. Tianle Pu, Jianing Li, Yingying Gao, Shixuan Liu, Zijie Geng, Haoyang Liu, Chao Chen, Changjun Fan. AAAI 2026 Oral.

Paper: [https://arxiv.org/abs/2511.09209](https://arxiv.org/abs/2511.09209)

## Dependencies

- **Python**: 3.8.13
- **Gurobi**: 11.0.3
- **NetworkX**: 2.8.4

To build the environment, use the provided Conda environment file:

```bash
conda env create -f environment.yaml
```

## Usage

Go to the root directory `CoCo-MILP`. Put the MILP instances under the `./instance` directory. Below is an illustration of the directory structure.

```
CoCo-MILP/
├── dataset/                # dataset directory
│   ├──graph_dataset.py     # graph dataset for MILP
├── instance/               # instances directory
├── test_logs/              # Output directory and log files
├── train_logs/             # Output directory and log files
├── pretrained_models       # pretrained models directory
├── solver                  # solvers directory, including guorbi and scip
├── gnn.py                  # GNN modules
├── process_data.py         # precess MILP instances
├── train.py                # training
├── test.py                 # testing
├── utils.py                # Utility functions
├── environment.yaml        # Conda environment file
└── README.md
```

The workflow of CoCo-MILP is as following.    

### 1. Data generation

We use [Ecole](https://www.ecole.ai/) library to generate Set Covering (SC) and Combinatorial Auction (CA) instance, and obtain the Balanced Item Placement (denoted by IP) and Workload Appointment (denoted by WA) instances from the ML4CO 2021 competition [generator](https://github.com/ds4dm/ml4co-competition-hidden). And you can download our data from Hugging Face:

https://huggingface.co/datasets/tianle326/L2O-MILP

For each benchmarks, we generate 300 instances for training and 100 instances for testing. We take SC for example, after generating the instances, place them in the `instance` directory following this structure: `instance/train/SC` and `instance/test/SC`.

### 2. Preprocessing

To preprocess a dataset (e.g., `SC`), run:

```bash
python process_data.py --problem_type "SC" --max_time 3600 --workers 10
```

The corresponding bipartite graph(BG) and solution will be automatically generated in the dataset folder.

### 3. Train

To train the model with default settings:

```bash
python train.py --problem_type "SC" --device "cuda:0"
```

### 4. Test

To evaluate the trained model, run: 

```bash
python test.py --test_problem_type "SC" --num_workers 10 --device "cuda:0"
```

Note: To avoid the impact of physical machine specifications on experimental results, it is recommended that the number of processes configured during preprocessing and testing does not exceed the maximum number of threads supported by the machine. For parallel testing, we pre-cache the scores of all instances in the `scores` folder. You can decide whether to enable this caching mechanism based on your specific use case.

## Citation

If you find CoCo-MILP useful or relevant to your research, please consider citing our paper.

```bash
@inproceedings{pu2026coco,
  title={CoCo-MILP: Inter-Variable Contrastive and Intra-Constraint Competitive MILP Solution Prediction},
  author={Tianle Pu and Jianing Li and Yingying Gao and Shixuan Liu and Zijie Geng and Haoyang Liu and Chao Chen and Changjun Fan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  year={2026}
}
```

