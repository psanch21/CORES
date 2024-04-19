datasets=("bzr" "cox2" "dd" "enzymes" "mutag" "nci1" "nci109" "proteins" "ptc_fm")
archis=("gin" "gat" "gcn")
top_k_is_soft=("True" "False")



for dataset in "${datasets[@]}"; do
  for archi in "${archis[@]}"; do
    for is_soft in "${top_k_is_soft[@]}"; do
      if [ "$is_soft" = "True" ]; then
          is_soft_str="soft"
      else
          is_soft_str="hard"
      fi
      python cores/delivery/cli/create_runs.py --sweep_file config/sweep/top_k.yaml --main_file cores/delivery/cli/top_k_train.py --count 20 --jobs_manager "output_file=jobs/shell/topk${is_soft_str}_${dataset}_${archi}.sh" --opts config_file=config/top_k/train.yaml config_file_data=config/dataset/${dataset}.yaml config_file_machine=config/machine/mps.yaml config_file_logger=config/logger/file_system.yaml param_data=config/sweep/dataset_${dataset}.yaml param_archi=config/sweep/archi_${archi}.yaml param_mode=config/sweep/top_k_ratio.yaml model_kwargs.use_gnn=${is_soft}
    done
  done
done



