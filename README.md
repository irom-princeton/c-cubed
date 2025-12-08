<h1 align="center">
    <div style="display: flex; align-items: center; justify-content: center;">
        <div style="flex: 0 0 14%; text-align: center;">
            <img src="assets/icon.gif" alt="Icon" style="height: 3.5em; max-width: 100%;">
        </div>
        <div style="flex: 0 0 86%; text-align: left;">
            World Models That Know When They Don't Know:
            <br /> <span style="font-size: xx-large;">Controllable Video Generation<span>
            <br /> <span style="font-size: xx-large;">with Calibrated Uncertainty<span>
        </div>
    </div>
</h1>
<p align="center"> 
    <span class="author-block"><a href="https://may0mei.github.io/">Zhiting&nbsp;Mei*</a></span>,
    <span class="author-block"><a
            href="https://tenny-yinyijun.github.io/">Tenny&nbsp;Yin</a></span>,
    <span class="author-block"><a href="#">Micah&nbsp;Baker</a></span>,
    <span class="author-block"><a href="#">Ola&nbsp;Shorinwa*</a></span>,
    <span class="author-block"><a
            href="https://irom-lab.princeton.edu/majumdar/">Anirudha&nbsp;Majumdar</a></span>
                
</p>
<p align="center">
    <sup>&#42;</sup>Equal Contribution.
</p>
<p align="center">
  <!-- <a href="">
    <img src="assets/irom_princeton.png" width="80%">
  </a> -->
  <h4 align="center">
  <a href="https://c-cubed-uq.github.io/">Project Page</a> 
  | <a href= "https://arxiv.org/abs/2512.05927">arXiv</a>
  <div align="center"></div>
</p>

<br>

Recent advances in generative video models have led to significant breakthroughs in high-fidelity video
synthesis,
specifically in controllable video generation where the generated video is conditioned on text and
action inputs.
This impressive leap in performance has paved the way for broad applications from instruction-guided
video editing
to world modeling in robotics.
Despite these exceptional capabilities, controllable video models often
_hallucinate_ generating future video frames that are misaligned with physical
reality; which raises
serious
concerns
in many tasks such as robot policy evaluation and planning. However, state-of-the-art video models lack
the
ability to assess and express their confidence, further impeding hallucination mitigation.
To rigorously address this challenge, we propose
$C^{3}$
an uncertainty quantification method for training _continuous-scale_
_calibrated_ _controllable_ video models
for _dense_ confidence estimation at the _subpatch_ (channel) level,
precisely localizing the uncertainty
in each generated video frame. The effectiveness of our UQ method is underpinned by three core
innovations:

1. Our method introduces a novel framework that trains video models for _correctness_
   and _calibration_
   via strictly proper scoring rules.

2. We estimate the video model's uncertainty in latent space, avoiding training instability and
   prohibitive training
   costs associated with pixel-space approaches.

3. We map the dense latent-space uncertainty to _interpretable_ pixel-level
   uncertainty in the RGB space for
   intuitive visualization, providing high-resolution uncertainty heatmaps that identify untrustworthy regions.

Through extensive experiments on large-scale robot learning datasets (Bridge and DROID) and real-world
evaluations,
we demonstrate that our method not only provides calibrated uncertainty estimates within the training
distribution,
but also enables effective out-of-distribution detection.

## Installation

Please follow the following steps:

1. Clone this repo.

```bash
git clone https://github.com/irom-princeton/c-cubed.git
```

2. Install your virtual environment and package manager, e.g., `uv`.

```bash
# specify environment variables
UV_INSTALL_DIR="/custom/path"
DEST_PATH="/custom/path"
# install UV
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh
source "${UV_INSTALL_DIR}/env"
# install virtual environment
uv venv "${DEST_PATH}/envs/dynuq" --python 3.13
# activate UV
source "${DEST_PATH}/envs/dynuq/bin/activate"
```

3. Install `dynuq` as a Python package.

```bash
uv pip install -e .
```

## Training

You can use the example config file `dynuq/configs/train_bridge_256_example.yaml` as a guide.

```bash
sbatch slurm_scripts/train/bridge_train_8gpu.sh
```

You may find the evaluation scripts in the `bash_scripts` directory useful for generating videos along with dense confidence predictions and measuring the calibration of the uncertainty estimates. You may refer to the config files in the `configs` directory as a guide.

### Generating Videos with Dense Confidence Predictions

You may run:

```bash
bash bash_scripts/test/bridge_inference_example.sh
```

### Measuring the Calibration of the Uncertainty Estimates

You may run:

```bash
bash bash_scripts/test/bridge_compute_metrics_example.sh
```

## Using Slurm

By default, the scripts run on a Slurm cluster.
You can disable this option by replacing the `sbatch` command with `bash` in any of the bash scripts.

Please refer to this [guide](https://researchcomputing.princeton.edu/support/knowledge-base/slurm#Serial-Jobs) for more help on Slurm.
