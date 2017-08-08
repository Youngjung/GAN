export CUDA_VISIBLE_DEVICES=0
python main.py --model CycleGAN --batch_size 1 --nEpochs 600 --dataset horse2zebra
