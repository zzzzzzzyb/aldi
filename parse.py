import argparse


def parser():
    data = argparse.ArgumentParser()
    data.add_argument('--dataset', type=str, help='dataset name')
    data.add_argument('--backbone', type=str, help='backbone model')
    data.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    data.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')

    data.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    data.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    data.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    data.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    data.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')

    data.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    data.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')

    data.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    data.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    data.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    data.add_argument('--model_type', nargs='?', default='sngcf',
                        help='Specify the name of model (sngcf).')
    data.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    data.add_argument('--alg_type', nargs='?', default='rw',
                        help='Specify the type of the graph convolutional layer from {rw, rw_single, rw_fixed, rw_single_svd, rw_svd, rw_final, rw_linear}.')

    data.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    data.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    data.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    data.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    data.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    data.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    return data.parse_args()

