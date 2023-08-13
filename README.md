# SemTra Training

## Skill sequence generator

## Behavior decoder & Skill boundary detector
```
python3 train_comde.py env={env} dataset_path={PATH_FOR_DATASET} save_suffix={FILENAME_FOR_SAVE}
```
env: one of ['metaworld', 'kitchen', 'rlbench', 'carla']

dataset_path: this has to indicate the parent directory of multiple sub-directories. \
E.g., dataset_path=/home/user_name/wind/4_target_skills/ for below example.

    └── /home/user_name/kitchen/
        ├── ./wind/4_target_skills/0
        ├── ./wind/4_target_skills/1
        ├── ./wind/4_target_skills/2

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
pretrained_suffix: same with the save_suffix at the training.\
step: pretrained model's number of iterations (so, integer value). Baseline is saved at every 50000step and semtra is saved at every 5000step.
date: the date when you run your training code.

sequential_requirement (Optional): one of ['sequential', 'reverse', 'replace x with y']. x and y are 'indices' of skills (integer value). You can see these values in environment codes, e.g., comde/rl/envs/franka_kitchen/franka_kitchen.py

non_functionality: metaworld -> 'speed', kitchen -> 'wind', rlbench -> 'weight'. 

parameter (Optional): speed -> ['slow', 'normal', 'fast'],  wind -> ['breeze', 'gust', 'flurry'], weight -> ['heavy', 'normal', 'light']. 


### Visualization
If you want to visualize the evaluation result, please add 'save_results=True' in your command line when you run evaluation. You can define your save path by save_suffix='{FILENAME_FOR_EVALUATION}'.
After that, a state-action trajectory is saved. According to your evaluation environment, please see the codes in misc/{env}/{visualize_xxx}.
