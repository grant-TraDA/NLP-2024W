from argparse import ArgumentParser, Namespace


def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data", choices=["coaid", "isot", "liar"])
    parser.add_argument("--model", choices=["roberta", "ernie"])
    parser.add_argument("--mask", choices=["yes", "no"])
    parser.add_argument("--seed", type=int)
    return parser.parse_args()
