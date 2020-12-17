import argparse

parser = argparse.ArgumentParser(description="Process some Command")
parser.add_argument('--p', type=int, default= 1 ,help='is plot')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--test', type = str, default='zlz')