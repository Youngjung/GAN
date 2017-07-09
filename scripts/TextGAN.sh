export CUDA_VISIBLE_DEVICES=0
python main.py \
--model TextGAN \
--dataset flowers \
--nEpochs 2000 \
--save_every_batch 100 \
--print_every 1 \
--lr 0.002
