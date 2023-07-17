starts=100000
n_iter=50
n_parallel=3

for ((iter=0; iter<n_iter; iter++)); do
  for ((j=0; j<n_parallel; j++)); do
    offset=$((iter * (5000 * n_parallel)))
    step=$((starts + offset + 5000 * j))
    echo $step
    CUDA_VISIBLE_DEVICES=0 python3 eval_comde.py \
    date=2023-07-16 \
    pretrained_suffix=mw_semtra_7460rm \
    save_suffix=mw_semtra_7460rm \
    env=metaworld \
    use_optimal_target_skill=True \
    use_optimal_next_skill=False \
    non_functionality=speed \
    sequential_requirement=sequential \
    n_eval=1 \
    step=$step &
  done
  wait
done
