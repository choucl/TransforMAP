#!/bin/bash
python ./TMAP_generate.py $1 $2 $3 $4 $5
#path_load_trace; model_save_path; path_to_prefetch_file; warm-up; total_instructions

#./generate.sh ../ChampSim/ML-DPC/LoadTraces/spec17/602.gcc-s2.txt.xz ./results/spec17/602.gcc-s2.model.pth ./results/spec17/602.gcc-s2 50 100
