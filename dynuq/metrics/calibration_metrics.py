import os
from typing import List, Dict
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from natsort import natsorted

from dynuq.metrics.base_metrics import BaseMetricConfig, BaseMetric
from dynuq.utils.load_utils import _load_config

class CalibrationMetricsConfig(BaseMetricConfig):
    # option to enable verbose print
    verbose_print: bool = True
    
    # base output path
    base_output_path: Path | str = "outputs/metrics/"

    # number of bins
    num_bins: int = 20

class CalibrationMetrics(BaseMetric):
    def __init__(
        self,
        cfg: CalibrationMetricsConfig = CalibrationMetricsConfig(),
    ):
        # init super-class
        super().__init__(cfg=cfg)
        
        # TODO: Add some other functionality

    def compute(self, cfg: dict = None):

        # # load config
        # cfg = _load_config(cfg_path)

        # create output directory, if needed
        base_output_path = Path(cfg.calibration.output_path)
        base_output_path.mkdir(parents=True, exist_ok=True)
        
        # error thresholds for calibration
        err_threshold_conf_pred = np.linspace(
            cfg.calibration.err_threshold_conf_pred_range[0], 
            cfg.calibration.err_threshold_conf_pred_range[1],
            num=cfg.calibration.num_err_threshold_conf_pred_steps,
        )
        
        # number of bins for calibration
        num_bins = cfg.calibration.num_bins

        # TODO: temporary split
        split_idx = os.environ.get("BRIDGE_SPLIT_IDX", None)
        
        if split_idx is None:
            split_idx = 0
        else:
            split_idx = int(split_idx)
        
        # TODO: temporary split size
        split_size = os.environ.get("BRIDGE_SPLIT_SIZE", None)
        
        if split_size is not None:
            split_size = int(split_size)
        
        self.console.log(f"Using split index: {split_idx}")
        
        # all eval videos
        eval_videos = natsorted(list(Path(cfg.dataset.gen_video_base_path).iterdir()))
        
        if split_size is not None:
            eval_videos = eval_videos[split_idx * split_size:(split_idx + 1) * split_size]

        def _compute_stats_per_thresh(err_thresh):
            # print
            self.console.log(f"Computing stats for error threshold: {err_thresh}")
            
            # disable logging:
            with self.console.capture() as capture:
                # all stats
                all_stats: Dict[str, Dict] = {}
            
                # update the base output path and create directory
                base_output_path_err_thresh = base_output_path / f"err_thresh_{err_thresh:.2f}"
                base_output_path_err_thresh.mkdir(parents=True, exist_ok=True)
            
                for in_vid in tqdm(eval_videos, desc="Running eval on the dataset"):
                    in_vid_dir = Path(cfg.dataset.gen_video_base_path) / in_vid.stem
                    stats_dict = self.compute_metrics(
                        data_dir=in_vid_dir,
                        err_threshold_conf_pred=err_thresh,
                        path_to_save_the_stats=base_output_path_err_thresh / f"per_video/{in_vid.stem}",
                        save_calibration_error_plots= cfg.calibration.save_plots, # TODO: refactor # cfg.save_calibration_error_plots,
                    )
                    all_stats[in_vid.stem] = stats_dict

                # path to save overall video stats
                overall_stats_path: Path = base_output_path_err_thresh / f"overall_calibration_stats_split_{split_idx}.json"
                
                # save overall video stats
                self.save_to_json(data=all_stats, file_path=overall_stats_path)
            
                # compute dataset stats
                abs_calib_error_per_bin_aggregate = np.zeros(num_bins)
                num_samples_per_bin = np.zeros(num_bins)
                total_num_samples = 0
                total_accuracy = 0.0
                total_accuracy_per_bin = np.zeros(num_bins)

                for vid_id, stats in all_stats.items():
                    abs_calib_error_per_bin_aggregate += stats["abs_calibration_error_per_bin"] * stats["number_of_samples_per_bin"]
                    num_samples_per_bin += stats["number_of_samples_per_bin"]
                    total_num_samples += np.sum(stats["number_of_samples_per_bin"])
                    total_accuracy += (stats["mean_accuracy"] * stats["total_num_samples"])
                    proc_acc_per_bin = stats["accuracy_per_bin"].copy()
                    proc_acc_per_bin[stats["accuracy_per_bin"] == None] = np.inf
                    total_accuracy_per_bin += proc_acc_per_bin.astype(float) * stats["number_of_samples_per_bin"]
                    bin_intervals = stats["bin_intervals"]

                # mean_accuracy /= len(all_stats)
                # recompute accuracy per bin
                merged_accuracy_per_bin = np.divide(
                    total_accuracy_per_bin,
                    num_samples_per_bin,
                    out=np.zeros_like(total_accuracy_per_bin),
                    where=num_samples_per_bin > 0,
                )
                merged_accuracy_per_bin[merged_accuracy_per_bin == np.inf] = None
                
                # mean accuracy
                mean_accuracy = total_accuracy / total_num_samples
                
                abs_calib_error_per_bin = np.divide(
                    abs_calib_error_per_bin_aggregate,
                    num_samples_per_bin,
                    out=np.zeros_like(abs_calib_error_per_bin_aggregate),
                    where=num_samples_per_bin > 0,
                )

            # compute dataset stats
            dataset_stats = self._compute_ece_mce(
                abs_calib_error_per_bin=abs_calib_error_per_bin,
                num_samples_per_bin=num_samples_per_bin,
                mean_accuracy=mean_accuracy,
                accuracy_per_bin=merged_accuracy_per_bin,
                total_num_samples=total_num_samples,
                bin_intervals=bin_intervals,
                path_to_save_the_stats=base_output_path_err_thresh,
                save_calibration_error_plots=cfg.calibration.save_plots,
            )
            
            
            # path to save overall video stats
            dataset_stats_path: Path = base_output_path_err_thresh / f"dataset_calibration_stats.json"
            
            # save dataset stats across all videos
            self.save_to_json(data=dataset_stats, file_path=dataset_stats_path)
                    
        # compute stats per threshold with multiprocessing
        all_run_outputs = Parallel(n_jobs=self.config.max_workers, prefer="threads")(
            delayed(_compute_stats_per_thresh)(seq_chunk)
            for seq_chunk in err_threshold_conf_pred
        )           
        
        # for err_thresh in err_threshold_conf_pred:
            
           
    def _compute_ece_mce(
        self,
        abs_calib_error_per_bin: np.ndarray,
        num_samples_per_bin: np.ndarray,
        mean_accuracy: float,
        accuracy_per_bin: np.ndarray,
        total_num_samples: int,
        bin_intervals: np.ndarray,
        video_id: str = None,
        path_to_save_the_stats: Path | str = "results/metrics",
        save_calibration_error_plots: bool = False,
    ):
        """
        Compute ECE and MCE
        """

        # Expected Calibration Error (ECE)
        ece: float = np.sum(abs_calib_error_per_bin * num_samples_per_bin / total_num_samples)

        # Maximum Calibration Error (MCE)
        mce_argmax: float = np.argmax(abs_calib_error_per_bin)
        mce: float = abs_calib_error_per_bin[mce_argmax]

        # bin associated with the MCE
        mce_bin: int = [bin_intervals[mce_argmax], bin_intervals[mce_argmax + 1]]

        # Print the results
        if self.config.verbose_print:
            # print the results
            terminal_width = shutil.get_terminal_size((80, 20)).columns
            if video_id is not None:
                print(f"Calibration Stats for Video ID: {video_id}")
            else:
                print(f"Overall Calibration Stats".center(terminal_width))
            print(f"Mean Accuracy: {mean_accuracy:.5f}")
            print(f"Expected Calibration Error (ECE): {ece:.5f}")
            print(f"Maximum Calibration Error (MCE): {mce:.5f}, Bin: {np.round(np.array(mce_bin), decimals=3)}")
            print(f"".center(terminal_width, "*"))

        # save the stats
        stats_dict = {
            "mean_accuracy": mean_accuracy,
            "ece": ece.item(),
            "mce": mce.item(),
            "bin_intervals": bin_intervals,
            "number_of_samples_per_bin": num_samples_per_bin,
            "accuracy_per_bin": accuracy_per_bin,
            "abs_calibration_error_per_bin": abs_calib_error_per_bin,
            "total_num_samples": total_num_samples,
        }

        if video_id is not None:
            stats_dict["video_id"] = video_id
            
        if save_calibration_error_plots:
            self.save_calibration_error_plots(
                bin_intervals=bin_intervals,
                num_samples_per_bin=num_samples_per_bin,
                accuracy_per_bin=accuracy_per_bin,
                video_id=video_id,
                path_to_save_the_figures=path_to_save_the_stats,
            )
            
        # if enable_conf_calibration:
        #     # perform conformal calibration
        #     conf_stats = conf_calibration.calibrate(
        #         data=stats_dict,
        #         stats_output_path=stats_output_path.parent,
        #         **conf_calibration_config,
        #     )

        return stats_dict


    def compute_metrics(
        self,    
        data_dir: Path,
        err_threshold_conf_pred: float = 0.85,
        num_bins: int = None,
        path_to_save_the_stats: Path | str = "results/metrics",
        save_calibration_error_plots: bool = False,
    ):
        """
        Compute the Calibration Error
        """
        # input-checking
        if num_bins is None:
            num_bins = self.config.num_bins
            
        # load data
        gt_latents = np.load(data_dir / f"gt_latents_conf_thresh_{err_threshold_conf_pred}.npy")
        generated_latents = np.load(data_dir / f"generated_latents_conf_thresh_{err_threshold_conf_pred}.npy")
        raw_confidences = np.load(data_dir / f"raw_conf_pred_conf_thresh_{err_threshold_conf_pred}.npy")

        # TODO: confirm first frame of gt not included in generation
        gt_latents = gt_latents[1:]
        
        # --- Compute accuracy --- #
        abs_errors = np.abs(gt_latents - generated_latents) # L1 error
        accuracy_per_samp = (abs_errors <= err_threshold_conf_pred).astype(np.float32)

        # mean accuracy
        mean_accuracy = np.mean(accuracy_per_samp)

        # --- Compute calibration error --- #
        calib_error_per_samp = accuracy_per_samp - raw_confidences
        
        # Save raw stats
        np.savez(
            Path(path_to_save_the_stats) / f"raw_stats.npz",
            abs_errors=abs_errors,
            raw_confidences=raw_confidences,
        )

        # --- partition the data into bins --- #

        # bin intervals [min, max)
        bin_intervals = np.linspace(0, 1, num_bins + 1)

        # total number of samples
        total_num_samples: int = raw_confidences.size

        # number of samples per bin
        num_samples_per_bin: List[int] = []

        # calibration error per bin (unweighted by the total number of samples)
        calib_error_per_bin: List[float] = []
        
        # accuracy per bin
        accuracy_per_bin: List[float] = []
        
        # bin-accuracy per sample
        bin_acc_per_sample = np.zeros_like(raw_confidences)
        
        # binned results
        for bin_idx in range(len(bin_intervals) - 1):
            # associated mask
            bin_mask = np.logical_and(
                raw_confidences > bin_intervals[bin_idx],
                raw_confidences <= bin_intervals[bin_idx + 1],
            )

            # number of samples per bin
            num_samples_per_bin.append(np.count_nonzero(bin_mask))

            if num_samples_per_bin[-1] > 0:
                # associated component of the calibration error
                calib_error_per_bin.append(np.mean(calib_error_per_samp[bin_mask]))
                
                # accuracy for the bin
                accuracy_per_bin.append(np.mean(accuracy_per_samp[bin_mask]))
            else:
                # empty bin
                calib_error_per_bin.append(0)
                
                # accuracy for the bin
                accuracy_per_bin.append(None)
                
            # cache the bin-accuracy per sample
            bin_acc_per_sample[bin_mask] = accuracy_per_bin[-1]
            
        # Cast to NumPy Array
        num_samples_per_bin = np.array(num_samples_per_bin)
        calib_error_per_bin = np.array(calib_error_per_bin)
        accuracy_per_bin = np.array(accuracy_per_bin)
        
        # absolute calibration error
        abs_calib_error_per_bin = np.abs(calib_error_per_bin)

        stats_dict = self._compute_ece_mce(
            abs_calib_error_per_bin=abs_calib_error_per_bin,
            num_samples_per_bin=num_samples_per_bin,
            mean_accuracy=mean_accuracy,
            accuracy_per_bin=accuracy_per_bin,
            total_num_samples=total_num_samples,
            bin_intervals=bin_intervals,
            video_id=data_dir.stem,
            path_to_save_the_stats=path_to_save_the_stats,
            save_calibration_error_plots=save_calibration_error_plots,
        )

        return stats_dict

     
    def save_calibration_error_plots(
        self,
        bin_intervals,
        num_samples_per_bin,
        accuracy_per_bin,
        num_bins: int = None,
        video_id: str=None,
        show_title=True,
        path_to_save_the_figures: Path = Path("results/dataset"),
        eps=1e-12,
    ):
        """
        Visualize the Calibration Error Curve
        """
        # input-checking
        if num_bins is None:
            num_bins = self.config.num_bins
          
        if video_id is None:
            video_id = ""
        # path to save the figures
        calib_figure_path: Path = Path(
            f"{path_to_save_the_figures}/{video_id}calib_curve.pdf"
        )

        # create parent directory, if necessary
        calib_figure_path.parent.mkdir(exist_ok=True, parents=True)

        # fontsize property
        # prop = {'size': 18}
        labelsize = 18
        plt.rc("axes", labelsize=labelsize)
        plt.rc("xtick", labelsize=labelsize)
        plt.rc("ytick", labelsize=labelsize)
        plt.rc("legend", fontsize=labelsize)

        # figure
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))

        if not (type(axs) is list or isinstance(axs, np.ndarray)):
            axs = [axs]

        # axis counter
        ax_counter = 0
        ax = axs[ax_counter]

        # optimal accuracy for each bin
        opt_acc_per_bin = (bin_intervals[:-1] + bin_intervals[1:]) / 2.0

        # calibration gap (bottom of the bar)
        calib_gap_bottom = accuracy_per_bin[num_samples_per_bin > 0]

        # calibration gap
        calib_gap = opt_acc_per_bin[num_samples_per_bin > 0] - calib_gap_bottom
        
        # plot the curve
        ax.bar(
            x=bin_intervals[:-1][num_samples_per_bin > 0],
            height=accuracy_per_bin[num_samples_per_bin > 0],
            width=1 / num_bins,
            align="edge",
            edgecolor="k",
            linewidth=2.2,
            label="Outputs",
        )

        # plot the calibration gap
        ax.bar(
            x=bin_intervals[:-1][num_samples_per_bin > 0][calib_gap > 0],
            height=calib_gap[calib_gap > 0],
            bottom=calib_gap_bottom[calib_gap > 0],
            width=1 / num_bins,
            align="edge",
            hatch="*",
            color=[1.0, 0.0, 0.0, 0.7],
            edgecolor=[0.8, 0.0, 0.0, 0.8],
            linewidth=2.2,
            label="Gap",
        )

        # plot the ideal trend
        ideal_calib_line = np.linspace(0, 1, 100)
        ax.plot(
            ideal_calib_line,
            ideal_calib_line,
            "-.k",
            linewidth=2.2,
        )
        
        # plot the density of samples per bin
        ax.scatter(
            x=(bin_intervals[1:] + bin_intervals[:-1]) / 2,
            y=np.asarray(num_samples_per_bin) / (np.sum(num_samples_per_bin) + eps),
            marker="P",
            linewidth=2.2,
            edgecolor=[0.35, 0.2, 0.86, 0.9],
            s=420,
            label="Density",
        )

        # axis label
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")

        # axis options
        ax.grid()
        ax.set_xlim(left=0, right=1)
        ax.set_ylim(bottom=0, top=1)
        ax.legend()
        # ax.legend(prop=prop)
        
        if show_title:
            # file path
            fpath = Path(path_to_save_the_figures)
            ax.set_title(f"{fpath.stem}{os.path.splitext(fpath)[-1]}")
            
        # increment counter
        ax_counter += 1
        
        # save the figure
        fig.savefig(fname=calib_figure_path, transparent=True, bbox_inches="tight", dpi=300)
        
        # close figure
        plt.close("all")
        
    def save_to_json(self, data: dict, file_path: str | Path):
        """
        Save data to JSON
        """
        
        with open(file_path, "w") as fh:
            # convert dictionary valiues to list
            data = self.dict_val_to_list_recursive(data)
                 
            # save data
            json.dump(data, fh, indent=4, default=float)
            
            # print
            self.console.log(f"Saved stats to {file_path}!")
            
    def dict_val_to_list_recursive(self, data):
        """
        Recursively map dictionary values to list

        Args:
            data (dict): The nested dictionary to process.

        Returns:
            dict: The nested dictionary with values converted to list.
        """
        # new data
        new_data = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                # If the value is a dict, recurse
                new_data[key] = self.dict_val_to_list_recursive(value)
            else:
                # Otherwise, convert the value to a  array
                new_data[key] = np.asarray(value).tolist()
                
        return new_data
        
        
# %%
