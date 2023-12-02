export CUDA_VISIBLE_DEVICES=0
python3.7 -m paddle.distributed.launch train.py --cfg configs/ddhrnet.yaml --use_gpu --do_eval 

# nohup ./train.sh > train_www.log 2>&1 &