
from models.eval_model import ModelEvaluator, ComparedResultStorer
from argparse import Namespace
from models.supports.utils import path_to_dict

def run_eval(args: Namespace) -> None:
    """
    """
    evaluator = ModelEvaluator(args)
    evaluator.run()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    
    parser.add_argument(
        '--phases', 
        default=[0.0, 0.005, 0.01, 0.05, 0.1],
        dest='phases'
    )

    parser.add_argument(
        '--density',
        default=False,
        dest='density'
    )

    parser.add_argument(
        '--acc',
        default=False,
        dest='acc'
    )

    parser.add_argument(
        '--data_dir',
        default='data',
        dest='data_dir'
    )

    parser.add_argument(
        '--config',
        default='config.yaml',
        dest='config'
    )

    parser.add_argument(
        '--ref',
        default='ref',
        dest='ref'
    )
    parser.add_argument(
        '--seed',
        default=0,
        dest='seed'
    )



    args = parser.parse_args()
    
    run_eval(args)
