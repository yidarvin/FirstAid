declare -r path_train=/home/dnr/Documents/data/DDSM-seg-1024/training
declare -r path_test=/home/dnr/Documents/data/DDSM-seg-1024/testing
declare -r path_val=/home/dnr/Documents/data/DDSM-seg-1024/validation

declare -r name=DDSM_seg_test

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt

python train_CNNsegmentation.py --pTrain $path_train --pTest $path_test --pVal $path_val --pModel $path_model --pLog $path_log --name $name --nGPU 4 --bs 24 --ep 1000
