import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="GNNDiff Model")

    parser.add_argument('--data_name', default='c')
    parser.add_argument('--TEST_MODE', default=False)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--save', type=bool, default=True)

    args = parser.parse_args()

    return args
