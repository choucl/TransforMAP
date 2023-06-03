#!/bin/bash
python ./TMAP_train.py $1 $2 $3 $4 $5 $6
# path_to_load_json path_of_data_dir path_to_save_model_to work_group ratio_of_training_data training_gpu_num

#train.sh /home/pengmiao/Disk/work/MLPrefComp/ChampSim/ML-DPC/LoadTraces/spec17/602.gcc-s2.txt.xz ./results/spec17/602.gcc-s2.model.pth 5

