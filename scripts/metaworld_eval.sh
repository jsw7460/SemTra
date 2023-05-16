for j in {1..4};
do
  for i in {1..10};
  do
  ten=$(($j * 10))
  step=$(($(($ten + $i)) * 5000))
  echo $step
  CUDA_VISIBLE_DEVICES=1 python3 eval_comde.py \
  date=2023-05-12 \
  pretrained_suffix=comde_mw_speed \
  env=metaworld \
  use_optimal_target_skill=True \
  use_optimal_next_skill=True \
  non_functionality=speed \
  save_suffix=comde_mw_speed \
  n_eval=1 \
  step=$step &
  done
  wait
done
