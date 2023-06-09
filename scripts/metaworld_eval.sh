for j in {0..8};
do
  for i in {1..5};
  do
  ten=$(($j * 10))
  step=$(($(($ten + $i)) * 5000))
  echo $step
  CUDA_VISIBLE_DEVICES=1 python3 eval_comde.py \
  date=2023-06-06 \
  pretrained_suffix=mw_speed_big_skpromptdt_easypr \
  env=metaworld \
  use_optimal_target_skill=False \
  use_optimal_next_skill=True \
  non_functionality=speed \
  sequential_requirement=sequential \
  save_suffix=mw_speed_big_skpromptdt_easypr \
  n_eval=1 \
  step=$step &
  done
  wait
done
