for j in {1..10};
do
  for i in {1..10};
  do
  ten=$(($j * 10))
  step=$(($(($ten + $i)) * 5000))
  echo $step
  CUDA_VISIBLE_DEVICES=1 python3 eval_comde.py \
  date=2023-05-15 \
  pretrained_suffix=comde_kitchen_wind_smallmlp \
  env=kitchen \
  use_optimal_target_skill=True \
  non_functionality=wind \
  n_eval=1 \
  step=$step &
  done
  wait
done
