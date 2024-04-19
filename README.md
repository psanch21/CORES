# CORES


Welcome to the repository for our article "Improving the interpretability of GNN predictions through conformal-based graph sparsification". This repository contains all the source code used in our research. We appreciate your interest in our work and hope you find it valuable.




## Installation
To get started, please follow these installation instructions.

Clone the repository
```bash
git clone https://github.com/psanch21/CORES.git
```

Check out the `pyproject.toml` file to see the dependencies. This project uses [Poetry](https://python-poetry.org/) for dependency management. Install dependencies:

``` bash
poetry install
```

Make sure you are using the environment created by Poetry. Run the tests to make sure everything is working:
```bash
pytest tests
```

### Training


To train base GNN
```bash
python cores/delivery/cli/cores_train.py --config_file config/cores/train.yaml --opts logger_kwargs.enable=False cores.gnn_mode=True
```


To train TopK (SAGPool or GPool)

```bash
python cores/delivery/cli/top_k_train.py --config_file_data config/dataset/mutag.yaml --opts trainer.max_epochs=30

```

To train CORES
```bash
# Train with the default configuration
python cores/delivery/cli/cores_train.py

# Train with the default configuration + overwrite some values
python cores/delivery/cli/cores_train.py --config_file config/cores/train.yaml --opts cores.action_refers_to=node data=mutag loss=bce_logits device=mps

# Train with the default configuration + overwrite some values included in train_extra.yaml
python cores/delivery/cli/cores_train.py --config_file config/cores/train.yaml --config_extra config/cores/train_extra.yaml

# Train with the default configuration + overwrite some values included in other yaml files
python cores/delivery/cli/cores_train.py --config_file config/cores/train.yaml --config_file_data config/dataset/mutag.yaml --config_file_reward config/cores/reward_conformal.yaml --opts logger_kwargs.enable=False trainer.limit_val_batches=1.0 trainer.limit_train_batches=1.0 cores.warm_up_epochs=300 trainer.max_epochs=3 checkpoint=enabled

python cores/delivery/cli/cores_train.py --config_file config/cores/train.yaml --config_file_data config/dataset/ba2motifs.yaml --config_file_machine config/machine/laptop.yaml --opts cores.warm_up_epochs=300 logger_kwargs.project=cores_test


```


## Experiment creation

All the sweep configurations are in the folder `config/sweep`. All the files to create the sweeps are in the folder `cores/delivery/jobs`. To create the sweeps, run the following commands:


```bash
./cores/delivery/jobs/gnn_jobs.sh
./cores/delivery/jobs/topk_jobs.sh
./cores/delivery/jobs/cores_jobs.sh
./cores/delivery/jobs/cores_ablation_jobs.sh
```




## Exploratory Data Analysis (EDA)


#### BA2Motif
```bash
python cores/delivery/cli/eda.py --name ba2motifs --k_fold 2 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
python cores/delivery/cli/eda.py --name ba2motifs --k_fold 3 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
```
#### BZR
```bash
python cores/delivery/cli/eda.py --name bzr --k_fold 2 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
python cores/delivery/cli/eda.py --name bzr --k_fold 3 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
```

#### COX2
```bash
python cores/delivery/cli/eda.py --name cox2 --k_fold 2 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
python cores/delivery/cli/eda.py --name cox2 --k_fold 3 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
```

#### DD
```bash
python cores/delivery/cli/eda.py --name dd --k_fold 2 --node_proba 0.03 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
python cores/delivery/cli/eda.py --name dd --k_fold 3 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
```


#### ENZYMES
```bash
python cores/delivery/cli/eda.py --name enzymes --k_fold 2 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
python cores/delivery/cli/eda.py --name enzymes --k_fold 3 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
```

#### MUTAG
```bash
python cores/delivery/cli/eda.py --name mutag --k_fold 3 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
python cores/delivery/cli/eda.py --name mutag --k_fold 2 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
```

#### NCI1
```bash
python cores/delivery/cli/eda.py --name nci1 --k_fold 2 --node_proba 0.1 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
python cores/delivery/cli/eda.py --name nci1 --k_fold 3 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
```


#### NCI109
```bash
python cores/delivery/cli/eda.py --name nci109 --k_fold 2 --node_proba 0.2 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
python cores/delivery/cli/eda.py --name nci109 --k_fold 3 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
```


#### Proteins
```bash
python cores/delivery/cli/eda.py --name proteins --k_fold 2 --node_proba 0.5 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
python cores/delivery/cli/eda.py --name nci109 --k_fold 3 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
```



#### PTC_FM
```bash
python cores/delivery/cli/eda.py --name ptc_fm --k_fold 2 --node_proba 1.0 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
python cores/delivery/cli/eda.py --name ptc_fm --k_fold 3 --opts root=../Data splits="[0.8,0.1,0.1]" shuffle_train=False single_split=False
```
## Citation

If you use the code or findings from this repository and/or article in your work, please cite the following:

```
@misc{SnchezMartn2024CORES,
      title={Improving the interpretability of GNN predictions through conformal-based graph sparsification}, 
      author={Pablo Sanchez-Martin and Kinaan Aamir Khan and Isabel Valera},
      year={2024},
      eprint={2404.12356},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

## Contact Information

If you have any questions, feedback, or inquiries about the code or the research, feel free to reach out to [psanchez@tue.mpg.de](mailto:psanchez@tue.mpg.de).

For issues related to the repository or code, you can also create a GitHub issue or pull request.

We appreciate your interest in our research and code! Your feedback and collaboration are valuable to us.



