import argparse

def add_args(parser: argparse.ArgumentParser):
    """Add common arguments that are used for every type of run."""
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--momentum', type=float, default=0.0, help="Momentum value for training")
    parser.add_argument('--weight-decay', type=float, default=1e-4, help="Weight decay for training")
    parser.add_argument('-e', '--num-epochs', type=int, default=1)
    parser.add_argument('--num-features', type=int, default=784)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--acc-test', action='store_true', help='Option for accuracy guard test in CH regression.')
    parser.add_argument('--ffn-dim-1', type=int, default=32)
    parser.add_argument('--ffn-dim-2', type=int, default=32)


def add_run_args(parser: argparse.ArgumentParser):
    """Add run arguments."""
    parser.add_argument('--data-folder',
                        type=str,
                        default='mnist_data',
                        help="The folder to download the MNIST dataset to.")
