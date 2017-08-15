export CUDA_VISIBLE_DEVICES=0
python main.py \
--dataset flowers \
--input_height 64 \
--model CrossModalRepr \
--use_TextNLL False \
--use_GAN False \
--note captionNLL_only \
--nEpochs 200 
