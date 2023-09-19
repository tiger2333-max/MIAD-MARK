import argparse

#Argments
parser = argparse.ArgumentParser(description='train models')
parser.add_argument('--root_path',default='E:/MIAD-MARK')
parser.add_argument('--model_name', default='resnet50',type=str, help='model_name')
parser.add_argument('--dataset',default='messidor', type=str, help='dataset')
parser.add_argument('--cla_num',default=4,type=int,help='class number')
parser.add_argument('--max_epoch', default=1, type=int, help='number of total epochs to run')
parser.add_argument('--bs',default=4,type=int,help='batch size')
parser.add_argument('--lr_init',default=0.001,type=float,help='initial learning rate')
parser.add_argument('--optimizer',default='SGD',type=str,help='optimizer')
parser.add_argument('--loss',default='CE',type=str,help='loss function')
parser.add_argument('--lr_step',default=5,type=int,help='lr decay step size')
parser.add_argument('--lr_decay',default=0.1,type=float,help='lr decay factor')
parser.add_argument('--seed',default=2023,type=int,help='random seed')
args = parser.parse_args()