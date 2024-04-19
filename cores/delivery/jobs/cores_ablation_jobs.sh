datasets=("bzr" "cox2" "dd" "enzymes" "mutag" "nci1" "nci109" "proteins" "ptc_fm")
archis=("gin")
actions=("node" "edge")

ablation_studies=("desired_ratios" "lambdas")

CONFIG_FILE_REWARD="config/cores/reward_conformal.yaml"
CONFIG_FILE_LOGGER="config/logger/file_system.yaml"
CONFIG_FILE_MACHINE="config/logger/file_system.yaml"

for dataset in "${datasets[@]}"; do
  for archi in "${archis[@]}"; do
    for action in "${actions[@]}"; do
      CONFIG_EXTRA_GNN="config/best/gnn/${dataset}_${archi}.yaml"
      CONFIG_EXTRA_CORES="config/best/cores/${dataset}_${archi}_${action}.yaml"
      CONFIG_FILE_DATA=config/dataset/${dataset}.yaml
      for ablation in "${ablation_studies[@]}"; do
        python cores/delivery/cli/create_runs.py --sweep_file config/sweep/cores_${ablation}.yaml --main_file cores/delivery/cli/cores_train.py --count 0 --jobs_manager "output_file=jobs/shell/cores_ablation_${dataset}_${archi}_${action}_${ablation}.sh" --opts config_file=config/cores/train.yaml config_file_data=${CONFIG_FILE_DATA} config_file_machine=${CONFIG_FILE_MACHINE} config_file_logger=${CONFIG_FILE_LOGGER} config_file_reward=${CONFIG_FILE_REWARD} config_extra=$CONFIG_EXTRA_GNN+$CONFIG_EXTRA_CORES cores.gnn_mode=False
        done
      done
  done
done