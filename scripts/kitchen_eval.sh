starts=50000
n_iter=20
n_parallel=3

for ((iter=0; iter<n_iter; iter++)); do
  for ((j=0; j<n_parallel; j++)); do
    offset=$((iter * (5000 * n_parallel)))
    step=$((starts + offset + 5000 * j))
    echo $step
    python3 eval_comde.py \
    date=2023-06-15 \
    pretrained_suffix=kt_wind_bigmlp_5dir \
    save_suffix=kt_wind_bigmlp_5dir \
    env=kitchen \
    use_optimal_target_skill=True \
    non_functionality=wind \
    n_eval=1 \
    step=$step &
  done
  wait
done
