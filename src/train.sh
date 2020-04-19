python train.py --config_name bm25top30v2-hardneg-albert_xlarge \
                --localhost 23985 \
                --val_epoch_ratio 0.333 \
                --alias typeloss1 \
                --add_typeloss \
#                 --add_noise_labels \
#                 --typeloss_epsilon 1 \
#                 --add_gp \
#                 --gp_epsilon 0.2 \