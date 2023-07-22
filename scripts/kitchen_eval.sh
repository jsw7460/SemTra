starts=100000
n_iter=150
n_parallel=3
gpu_idx=1

for ((iter=0; iter<n_iter; iter++)); do
  for ((j=0; j<n_parallel; j++)); do
    offset=$((iter * (5000 * n_parallel)))
    step=$((starts + offset + 5000 * j))
    echo $step
    CUDA_VISIBLE_DEVICES=$gpu_idx python3 eval_comde.py \
    date=2023-07-19 \
    pretrained_suffix=kt_5v \
    save_suffix=kt_5v \
    env=kitchen \
    use_optimal_target_skill=True \
    use_optimal_next_skill=True \
    non_functionality=wind \
    n_eval=1 \
    step=$step &
  done
  wait
done
