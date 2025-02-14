from args import parse_args
from evolution import ape, ga_evo, de_evo, pso_evo
from utils import set_seed
from evaluator import CLSEvaluator, SimEvaluator, SumEvaluator

def run(args):
    print("Hello from run.run() 1")
    set_seed(args.seed)
    task2evaluator = {
        "cls": CLSEvaluator,
        "sum": SumEvaluator,
        "sim": SimEvaluator,
    }
    print("Hello from run.run() 2")
    print(args.task)
    evaluator = task2evaluator[args.task](args)
    evaluator.logger.info(
        "---------------------Evolving prompt-------------------\n"
    )
    print("Hello from run.run() 3")
    if args.evo_mode == "para" and args.para_mode == "topk":
        ape(args=args, evaluator=evaluator)
    elif args.evo_mode in 'de':
        de_evo(args=args, evaluator=evaluator)
    elif args.evo_mode == 'ga':
        print("Hello before calling ga_evo from run.run()")
        ga_evo(args=args, evaluator=evaluator)
    elif args.evo_mode == 'pso':
        print("Hello before calling pso_evo from run.run()")
        pso_evo(args=args, evaluator=evaluator)

if __name__ == "__main__":
    print("Hello from run.main()")
    args = parse_args()
    run(args)
