export CUDA_VISIBLE_DEVICES=1
python main.py \
--dataset flowers \
--input_height 64 \
--model CrossModalRepr \
--nEpochs 200 
