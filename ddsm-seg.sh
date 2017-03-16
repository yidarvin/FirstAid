declare -r path_train=/home/dnr/Documents/data/ddsm/training
#declare -r path_test=/home/dnr/Documents/data/ddsm/testing
declare -r path_val=/home/dnr/Documents/data/ddsm/validation
declare -r path_inf=/home/dnr/Documents/data/ddsm/inference

declare -r name=ddsm-seg

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt

python train_CNNsegmentation.py --pTrain $path_train --pInf $path_inf --pVal $path_val --pModel $path_model --pLog $path_log --name $name --nGPU 4 --bs 64 --ep 500
