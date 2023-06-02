# Instruction for Running TransforMAP, for ML-Based Data Prefetching Competition

## 0. File structure

* All codes are in the folder of `TransforMAP`, including python codes and `train.sh`, `generate.sh` two bash scripts following the guidance format.
* Dependencies are listed in `env.yaml`
* Pretrained models are stored in folder `Pretrained_models`

## 1. Model training

```
./train.sh path_to_load_json path_of_data_dir path_to_save_model_to ratio_of_training_data
```

* `train.sh` need to add executive permission using: `chmod +x ./train.sh`
* if  `path_to_save_model_to` exists, the existing model will be loaded and continue training
* Epochs: the default epoch=50  with early stop mechanism when converge. The time duration relies on the size of dataset, manually stop is supported.
* Manually stop:  For any reason to stop the model training manually, simply use `ctrl+c` to stop the terminal. We save the model during training. 

## 2. Model inference and prefetching file generation

```
./generate.sh path_to_load_trace path_to_saved_model path_to_prefetch_file num_warmup_instructions num_total_instructions
```

* If training for 100M program instructions and test on the next 100M instructions, then:
  * num_warmup_instructions=100
  * num_total_instructions=200

## 3. Pretrained models

* We provide some pretrained models for randomly picked applications in SPEC06 and SPEC17, but they are trained for only first 20M instructions for the workshop paper.  It can not be directly used for Competition but it can reduce the training time by loading and retraining.
* Model name is in the format of `<AppName>.bitmap.model.pth`

* SPEC17: 
  * 602.gcc-s0.trace.xz
  * 605.mcf-s1.trace.xz
  * 607.cactuBSSN-s0.trace.xz
  * 619.lbm-s2.trace.xz
  * 620.omnetpp-s0.trace.xz
  * 621.wrf-s3.trace.xz
* SPEC06:
  * 429.mcf-s1.trace.gz
  * 433.milc-s1.trace.gz
  * 437.leslie3d-s1.trace.gz
  * 459.GemsFDTD-s2.trace.gz
  * 471.omnetpp-s1.trace.gz
  * 473.astar-s1.trace.gz
  * 482.sphinx3-s1.trace.gz
