datasets=("bzr" "cox2" "dd" "enzymes" "mutag" "nci1" "nci109" "proteins" "ptc_fm")
archis=("gin" "gat" "gcn")
actions=("node" "edge")

for dataset in "${datasets[@]}"; do
  for archi in "${archis[@]}"; do
    for action in "${actions[@]}"; do
      python cores/delivery/cli/create_runs.py --sweep_file config/sweep/cores.yaml --main_file cores/delivery/cli/cores_train.py --count 20 --jobs_manager "output_file=jobs/shell/cores_${dataset}_${archi}_${action}.sh" --opts config_file=config/cores/train.yaml config_file_data=config/dataset/${dataset}.yaml config_file_machine=config/machine/mps.yaml config_file_logger=config/logger/file_system.yaml config_file_reward=config/cores/reward_conformal.yaml config_extra=config/best/gnn/${dataset}_${archi}.yaml param_data=config/sweep/dataset_${dataset}.yaml param_archi=config/sweep/archi_${archi}.yaml param_action=config/sweep/action_${action}.yaml cores.gnn_mode=False
      done
  done
done