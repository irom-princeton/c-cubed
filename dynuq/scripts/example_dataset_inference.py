import fire

from dynuq.evaluators.bridge_evaluator import BridgeEvaluator
if __name__ == "__main__":
    # example usage
    fire.Fire(BridgeEvaluator().evaluate)