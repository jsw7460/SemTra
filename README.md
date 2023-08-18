# SemTra Training
Semtra consists of six modules. Due to the modularized structure, we separate the training codes.
(1) multi-modal skill encoder, (2) semantic skill sequence generator, (3) skill boundary detector, (4) context encoder, (5) online context encoder, and (6) behavior decoder.

## Setup
(1) Install environment using environment configure file:
```
conda env create --file semtra_env.yaml
```
(2) Install multi-stage Meta-World 
```
cd meta_world
python3 -m pip install -e .
```
(3) Install Franka Kitchen using D4RL.


## (2) Skill sequence generator
```
python3 translate_learning_comde.py mode=translate_learning save_suffix={FILELANE_FOR_SAVE}
```

## (4) Context encoder
```
python3 prompt_learning_comde.py mode=prompt_learning save_suffix={FILENAME_FOR_SAVE}
```

## (3) Skill boundary detector & (5) Online context encoder & (6) Behavior decoder
```
python3 train_comde.py env={env} dataset_path={PATH_FOR_DATASET} save_suffix={FILENAME_FOR_SAVE} mode={mode}
```
env: one of ['metaworld', 'kitchen', 'rlbench', 'carla']
mode: one of ['comde', 'comde_nonstationary']. If you want to train a contrastive-learning-based dynamics encoder (i.e., online context encoder), use nonstationary mode.

dataset_path: this has to indicate the parent directory of multiple sub-directories. \
E.g., dataset_path=/home/user_name/wind/4_target_skills/ for below example.

    └── /home/user_name/kitchen/
        ├── ./wind/4_target_skills/0
        ├── ./wind/4_target_skills/1
        ├── ./wind/4_target_skills/2

## Baseline training
If you want to train a baseline, use the following commands.
```
python3 train_comde.py env={env} dataset_path={PATH_FOR_DATASET} save_suffix={FILENAME_FOR_SAVE} mode=baseline baseline={baseline}
```
The placeholder {baseline} can be any baselines, which are showcased in config/train/baseline.
For instance, use baseline=bcz, or baseline=flaxvima, etc.

 
# Semtra Evaluation
```
python3 eval_comde.py \
env={env} \
pretrained_suffix={SAVED_FILE_NAME} \
step={STEP} \
date={2023-mm-dd} \
sequential_requirement={SEQ} \
non_functionality={NF} \
parameter={PRM}
```
pretrained_suffix: same with the save_suffix at the training of behavior decoder.\
Please fill config/eval/eval_base.yaml/composition and .../eval_base.yaml/prompt to indicate absolute path of pretrained skill sequence generator and context encoder.

step: pretrained model's number of iterations (so, integer value). Baseline is saved at every 50000step and semtra is saved at every 5000step.
date: the date when you run your training code.

sequential_requirement (Optional): one of ['sequential', 'reverse', 'replace x with y']. x and y are 'indices' of skills (integer value). You can see these values in environment codes, e.g., comde/rl/envs/franka_kitchen/franka_kitchen.py

non_functionality: metaworld -> 'speed', kitchen -> 'wind', rlbench -> 'weight'. 

parameter (Optional): speed -> ['slow', 'normal', 'fast'],  wind -> ['breeze', 'gust', 'flurry'], weight -> ['heavy', 'normal', 'light']. 


### Visualization
If you want to visualize the evaluation result, please add 'save_results=True' in your command line when you run evaluation. You can define your save path by save_suffix='{FILENAME_FOR_EVALUATION}'.
After that, a state-action trajectory is saved. Based on your evaluation environment, please see the codes in misc/{env}/{visualize_xxx}.
