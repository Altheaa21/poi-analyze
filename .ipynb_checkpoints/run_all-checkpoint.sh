#!/bin/bash
# Run metrics pipeline for all cities

# 激活环境（如果需要的话，可以取消下面注释）
# source /srv/scratch/z5447761/llama2-env/bin/activate

for config in configs/*.yaml; do
    echo ">>> Running pipeline for $config"
    python scripts/run_city.py --config "$config"
    echo ">>> Finished $config"
    echo "-----------------------------------"
done

