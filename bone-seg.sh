declare -r path_train=/home/dnr/Documents/data/bone-512/training
#declare -r path_test=/home/dnr/Documents/data/bone-512/testing
declare -r path_val=/home/dnr/Documents/data/bone-512/validation
declare -r path_inf=/home/dnr/Documents/data/bone-512/inference

declare -r name=bone-seg

declare -r path_model=/home/dnr/modelState/$name.ckpt
declare -r path_log=/home/dnr/logs/$name.txt

#python train_CNNsegmentation.py --pTrain $path_train --pInf $path_inf --pVal $path_val --pModel $path_model --pLog $path_log --name $name --nGPU 4 --bs 128 --ep 500 --lr 0.0001

python train_CNNsegmentation.py --pInf $path_inf --pVal $path_val --pModel $path_model --pLog $path_log --name $name --nGPU 4 --bs 128 --ep 500 --lr 0.0001
