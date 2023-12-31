import argparse

#Argments
parser = argparse.ArgumentParser(description='train models')
parser.add_argument('--root_path',default='E:/MIAD-MARK')
parser.add_argument('--model_name', default='resnet50',type=str, help='model_name')
parser.add_argument('--dataset',default='messidor', type=str, help='dataset')
parser.add_argument('--cla_num',default=4,type=int,help='class number')
parser.add_argument('--seed',default=2023,type=int,help='random seed')
parser.add_argument('--down_val',default=110,type=int,help='down value of threshold')
parser.add_argument('--up_val',default=190,type=int,help='up value of threshold')
parser.add_argument('--step',default=20,type=int,help='the step of threshold')
parser.add_argument('--sample_num', type=int, help='sampling numbers of each class', required=False)
args = parser.parse_args()
