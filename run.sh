set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=4,5
#export CUDA_VISIBLE_DEVICES=6,7
#export CUDA_VISIBLE_DEVICES=0,1,2,4
export LD_LIBRARY_PATH=./lib64/
export PYTHONIOENCODING=utf8
python main_apex.py