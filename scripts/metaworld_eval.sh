for j in {0..8};
do
  for i in {1..5};
  do
  ten=$(($j * 10))
  step=$(($(($ten + $i)) * 5000))
  echo $step
  CUDA_VISIBLE_DEVICES=1 python3 eval_comde.py \
  date=2023-06-03 \
  pretrained_suffix=mw_speed_skill_promptdt \
  env=metaworld \
  use_optimal_target_skill=False \
  use_optimal_next_skill=True \
  non_functionality=speed \
  save_suffix=mw_speed_skill_promptdt \
  n_eval=1 \
  step=$step &
  done
  wait
done
