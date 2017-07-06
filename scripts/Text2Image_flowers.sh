export CUDA_VISIBLE_DEVICES=0
python main.py \
--dataset flowers \
--input_height 64 \
--model Text2Image \
--nEpochs 200 \
--textEncoder DSSJE
