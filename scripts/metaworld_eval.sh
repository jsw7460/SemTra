starts=50000
n_iter=50
n_parallel=3

for ((iter=0; iter<n_iter; iter++)); do
  for ((j=0; j<n_parallel; j++)); do
    offset=$((iter * (5000 * n_parallel)))
    step=$((starts + offset + 5000 * j))
    echo $step
    CUDA_VISIBLE_DEVICES=1 python3 eval_comde.py \
    date=2023-06-21 \
    pretrained_suffix=mw_speed_5dir \
    env=metaworld \
    use_optimal_target_skill=False \
    use_optimal_next_skill=False \
    non_functionality=speed \
    sequential_requirement=sequential \
    save_suffix=mw_speed_5dir \
    n_eval=1 \
    step=$step &
  done
  wait
done
