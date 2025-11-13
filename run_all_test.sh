#!/bin/bash
# Run metrics pipeline for all TEST cities
# ----------------------------------------

# 1) activate env (uncomment if needed)
# source /srv/scratch/4547761/llama2-env/bin/activate

# 2) ensure python can find poi package
export PYTHONPATH="$(pwd)/src:$(pwd)"

# 3) loop through all *_test.yaml configs
for config in configs/*_test.yaml; do
    echo ">>> Running TEST pipeline for $config"
    python scripts/run_city.py --config "$config"
    echo ">>> Finished $config"
    echo "--------------------------------------"
done