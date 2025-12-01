import argparse
from tqdm import tqdm
import hydra
from hydra.utils import instantiate

def main(cfg):

    # ----------------------------------------------------------------------- #
    #  Instantiate evaluators (Not need now)
    # ----------------------------------------------------------------------- #
    # evaluators = [
    #     instantiate(evaluator)
    #     for evaluator in cfg.evaluators.values()
    # ]
    
    # ----------------------------------------------------------------------- #
    #  Instantiate metrics
    # ----------------------------------------------------------------------- #
    # TODO: modify perceptual metrics cfg to support hydra instantiation
    metrics = [
        instantiate(metric) 
        for metric in cfg.metrics.values()
    ]

    for metric in tqdm(metrics, desc="Computing metrics"):
        # compute metrics
        metric.compute(cfg)

if __name__ == "__main__":
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str, 
        required=False,
        default="bridge_compute_metrics",
        help="Name of the config file"
    )
    
    # parse arguments
    args = parser.parse_args()
    
    # load the configs from file
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=f"../configs", version_base="1.1")
    cfg = hydra.compose(config_name=args.config_name)
    
    # run the main function
    main(
        cfg=cfg
    )