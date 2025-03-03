import argparse


def parser():
    data = argparse.ArgumentParser()
    data.add_argument('--dataset', type=str, help='dataset name')
    data.add_argument('--backbone', type=str, help='backbone model')
    return data.parse_args()


print(parser().dataset)