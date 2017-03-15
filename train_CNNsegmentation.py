import argparse
import sys

from utils.segmentation import segmentor

def main(args):
    """
    Main function to parse arguments.
    INPUTS:
    - args: (list of strings) command line arguments
    """
    # Reading command line arguments into parser.
    parser = argparse.ArgumentParser(description = "Do CNN Segmentation.")

    # Paths: arguments for filepath to misc.
    parser.add_argument("--pTrain", dest="path_train", type=str, default=None)
    parser.add_argument("--pVal", dest="path_validation", type=str, default=None)
    parser.add_argument("--pTest", dest="path_test", type=str, default=None)
    parser.add_argument("--pInf", dest="path_inference", type=str, default=None)
    parser.add_argument("--pModel", dest="path_model", type=str, default=None)
    parser.add_argument("--pLog", dest="path_log", type=str, default=None)
    parser.add_argument("--pVis", dest="path_visualization", type=str, default=None)

    # Experiment Specific Parameters (i.e. architecture)
    parser.add_argument("--name", dest="name", type=str, default="noname")
    parser.add_argument("--net", dest="network", type=str, default="GoogLe")
    parser.add_argument("--nClass", dest="num_class", type=int, default=2)
    parser.add_argument("--nGPU", dest="num_gpu", type=int, default=1)
    
    # Hyperparameters
    parser.add_argument("--lr", dest="lr", type=float, default=0.001)
    parser.add_argument("--dec", dest="lr_decay", type=float, default=1.0)
    parser.add_argument("--l2", dest="l2", type=float, default=0.0000001)
    parser.add_argument("--l1", dest="l1", type=float, default=0.0)
    parser.add_argument("--bs", dest="batch_size", type=int, default=12)
    parser.add_argument("--ep", dest="max_epoch", type=int, default=10)
    parser.add_argument("--time", dest="max_time", type=int, default=1440)

    # Switches
    parser.add_argument("--bLo", dest="bool_load", type=int, default=0)
    parser.add_argument("--bDisp", dest="bool_display", type=int, default=1)

    # Creating Object
    opts = parser.parse_args(args[1:])
    CNN_obj = segmentor(opts)
    CNN_obj.train_model() #Train/Validate the Model
    CNN_obj.test_model() #Test the Model.
    CNN_obj.do_inference() #Do inference on inference set.

    # We're done.
    return 0
    

if __name__ == '__main__':
    main(sys.argv)
