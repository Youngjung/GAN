export CUDA_VISIBLE_DEVICES=0
python main.py \
--model TextGAN \
--dataset mscoco \
--nEpochs 2000 \
--save_every_batch 100 \
--print_every 10 \
--lr 0.002 \
--log_dir logs_TextGAN_withImage \
--inception_checkpoint_file checkpoint/inception_v3.ckpt
