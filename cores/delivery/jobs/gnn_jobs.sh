datasets=("bzr" "cox2" "dd" "enzymes" "mutag" "nci1" "nci109" "proteins" "ptc_fm")
archis=("gin" "gat" "gcn")

for dataset in "${datasets[@]}"; do
  for archi in "${archis[@]}"; do
    python cores/delivery/cli/create_runs.py --sweep_file config/sweep/gnn.yaml --count 20 --jobs_manager "output_file=jobs/shell/gnn_${dataset}_${archi}.sh" --opts config_file=config/cores/train.yaml config_file_data=config/dataset/${dataset}.yaml config_file_machine=config/machine/mps.yaml config_file_logger=config/logger/file_system.yaml param_data=config/sweep/dataset_${dataset}.yaml param_archi=config/sweep/archi_${archi}.yaml
  done
done