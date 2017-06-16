export CUDA_VISIBLE_DEVICES=1
python main.py --model CycleGAN --batch_size 1 --nEpochs 600 --dataset horse2zebra
