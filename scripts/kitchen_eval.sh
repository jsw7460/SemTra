for j in {1..4};
do
  for i in {1..5};
  do
  ten=$(($j * 10))
  step=$(($(($ten + $i)) * 5000))
  echo $step
  CUDA_VISIBLE_DEVICES=1 python3 eval_comde.py \
  date=2023-06-08 \
  pretrained_suffix=kt_skill_promptdt \
  env=kitchen \
  use_optimal_target_skill=False \
  non_functionality=wind \
  n_eval=1 \
  step=$step &
  done
  wait
done
