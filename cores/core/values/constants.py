from enum import Enum

from tueplots import bundles

cores_bundle = bundles.icml2022()

cores_bundle["font.size"] = 10
cores_bundle["axes.labelsize"] = 10
cores_bundle["legend.fontsize"] = 10
cores_bundle["xtick.labelsize"] = 8
cores_bundle["ytick.labelsize"] = 8
cores_bundle["axes.titlesize"] = 10

data_mapping = {
    "bzr": "BZR",
    "cox2": "COX2",
    "dd": "DD",
    "enzymes": "ENZYMES",
    "mutag": "MUTAG",
    "nci1": "NCI1",
    "nci109": "NCI109",
    "proteins": "PROTEINS",
    "ptc_fm": "PTC",
}

data_order = ["bzr", "cox2", "dd", "enzymes", "mutag", "nci1", "nci109", "proteins", "ptc_fm"]


model_style = {
    r"CORES$_N$": {"marker": "o", "linestyle": "solid"},
    r"CORES$_E$": {"marker": "^", "linestyle": "dashed"},
}

model_mapping = {
    "cores-node": r"CORES$_N$",
    "cores-edge": r"CORES$_E$",
    "gnn": "GNN",
    "topk_hard": "GPool",
    "topk_soft": "SAGPool",
    "sugar": "SUGAR",
    "diff_pool": "DiffPool",
}

model_order = ["gnn", "topk_soft", "diff_pool", "sugar", "topk_hard", "cores-node", "cores-edge"]


mutag_atoms_style = {
    "Carbon": {"color_id": 2},
    "Nitrogen": {"color_id": 3},
    "Oxigen": {"color_id": 4},
    "Chlorine": {"color_id": 7},
}


params_mapping = {
    "data_kwargs.splits": r"Split sizes",
    "dataloader_kwargs.batch_size": r"Batch size",
    "early_stopping_kwargs.clf_kwargs.patience": r"Early stopping clf. patience",
    "early_stopping_kwargs.patience": r"Early stopping clf. patience",
    "graph_clf_kwargs.dropout": r"Dropout rate",
    "graph_clf_kwargs.has_bn": r"Batch normalizing",
    "graph_clf_kwargs.heads": r"Number of heads",
    "graph_clf_kwargs.hidden_dim": r"Dimension of hidden layers",
    "graph_clf_kwargs.layers_gnn_num": r"Number of GNN layers",
    "graph_clf_kwargs.pooling": r"Global pooling type",
    "graph_clf_kwargs.eps": r"$\epsilon$",
    "graph_clf_kwargs.train_eps": r"Trainable $\epsilon$",
    "lr_scheduler_kwargs_clf.factor": r"Classifier scheduler factor",
    "lr_scheduler_kwargs.factor": r"Classifier scheduler factor",
    "lr_scheduler_kwargs_rl.factor": r"RL scheduler factor",
    "optimizer_kwargs_clf.lr": r"Classifier learning rate",
    "optimizer_kwargs.lr": r"Classifier learning rate",
    "optimizer_kwargs_rl.ratio_critic": r"Ratio of the critic learning rate",
    "reward_kwargs.alpha": r"Conformal error rate $\alpha$",
    "cores.action_refers_to": r"Removal mode",
    "ppo_kwargs.eps_clip": r"PPO clip value $\epsilon$",
    "ppo_kwargs.coeff_mse": r"PPO MSE coefficient",
    "ppo_kwargs.coeff_entropy": r"PPO entropy coefficient",
    "graph_env_kwargs.penalty_size": r"Environment penalty size",
    "cores.ppo_steps": r"Number of PPO epochs",
    "cores.env_steps": r"Number of environment steps",
    "early_stopping_kwargs.ppo_kwargs.patience": r"Early stopping PPO patience",
    "top_k.multiplier": r"TopK multiplier",
    "top_k.ratio": r"TopK ratio",
    "reward_kwargs.desired_ratio": r"$d$",
    "reward_kwargs.lambda_1": r"$\lambda$",
}


metrics_mapping = {
    "testing_test_end/1__accuracy": "Accuracy (\%)",
    "testing_test_end/1__num_nodes_ratio": "Node Ratio (\%)",
    "testing_test_end/1__num_edges_ratio": "Edge Ratio (\%)",
}


class ActionTypes(Enum):
    NODE = "node"
    EDGE = "edge"


class Cte:
    TAB = " --- "


class Datasets:
    BA2MOTIFS = "ba2motifs"
    BZR = "bzr"
    COX2 = "cox2"
    DD = "dd"
    ENZYMES = "enzymes"
    MUTAG = "mutag"
    NCI1 = "nci1"
    NCI109 = "nci109"
    PROTEINS = "proteins"
    PTC_FM = "ptc_fm"


class DataTypes(Enum):
    GRAPH = "graph"
    IMAGE = "image"


class Distributions(Enum):
    BERNOULLI = "bernoulli"
    CONT_BERNOULLI = "continous_bernoulli"
    CATEGORICAL = "categorical"


class Framework:
    PYTORCH = "torch"


class GraphEnvTypes(Enum):
    SINGLE = "single"
    MULTI = "multi"


class GraphFramework(Enum):
    NETWORKX = "networkx"
    PYG = "pyg"


class PoolingTypes(Enum):
    GLOBAL_ATT = "gatt"
    MEAN = "mean"
    # TOPK = "topk"
    MAX = "max"
    ADD = "add"


class ActivationTypes(Enum):
    ELU = "elu"
    IDENTITY = "identity"
    LEAKY_RELU = "lrelu"
    PRELU = "prelu"
    RELU = "relu"
    SELU = "selu"
    SIGMOID = "sigmoid"
    SIN = "sin"
    SOFTMAX = "softmax"
    TANH = "tanh"


class DataBalanceTypes(Enum):
    UPSAMPLE = "upsample"
    DOWNSAMPLE = "downsample"
    WEIGHT = "weight"


class InitFnTypes(Enum):
    XAVIER = "xavier"
    NORMAL = "normal"


class GNNLayers(Enum):
    GCN = "gcn"
    GAT = "gat"
    GIN = "gin"


class LoggerType(Enum):
    DUMMY = "dummy"
    FILE_SYSTEM = "file_system"
    PRINT = "print"
    WANDB = "wandb"


class LossType(Enum):
    L1 = "l1"
    L2 = "l2"
    CROSS_ENTROPY = "cross_entropy"
    BCE = "bce"
    BCE_LOGITS = "bce_logits"


class Models:
    CORES = "cores"
    TOP_K_RATIO = "top_k_ratio"
    GNN = "gnn"


class ScalerTypes(Enum):
    IDENTITY = "identity"
    STD = "std"


class LossTypes(Enum):
    BCELOGITS = "bce_logits"


class LRSchedulerType(Enum):
    EXP = "exponential"
    COS = "cosine"
    PLATEAU = "plateau"
    STEP = "step"
    NONE = None


class OptimizerType(Enum):
    ADAM = "adam"
    RADAM = "radam"
    SGD = "sgd"
    NONE = None


class RewardTypes(Enum):
    CORES_CONFORMAL = "cores_conformal"


class StageTypes(Enum):
    SKIPSUM = "skipsm"


class SplitNames:
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    CALIB = "calib"


class PolicyTypes(Enum):
    GRAPH_ACTOR_CRITIC = "graph_actor_critic"


class TaskDomains(Enum):
    NODE = "node"
    EDGE = "edge"
    LINK = "link"
    GRAPH = "graph"


class TaskTypes(Enum):
    CLF_BINARY = "clf_binary"
    CLF_MULTICLASS = "clf_multiclass"
    CLF_MULTILABEL = "clf_multilabel"
    REG = "reg"
    GEN = "gen"
