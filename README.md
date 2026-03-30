# BlurBall

Code and dataset for the paper: **[BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking]([https://cogsys-tuebingen.github.io/blurball/](https://arxiv.org/abs/2509.18387))**

![demo](demo.gif)

This repository builds upon [WASB: Widely Applicable Strong Baseline for Sports Ball Detection and Tracking](https://github.com/nttcom/WASB-SBDT/) with additional training scripts, dataset support, and modifications for blur-aware ball detection.


## Features at a Glance 🚀

- **New dataset**: 64k frames from real table tennis games, annotated with ball positions and explicit motion blur attributes (length + orientation).  
- **Blur-aware labeling**: A new annotation convention placing the ball at the blur center, improving accuracy across detectors.  
- **BlurBall model**: An HRNet-based multi-frame detector extended with attention mechanisms (Squeeze-and-Excitation) to jointly predict ball position and blur.  
- **Pretrained models**: Ready-to-use weights for BlurBall and other baselines (WASB, TrackNetV2, Monotrack, DeepBall, BallSeg).  


## Dataset

We release a **table tennis ball dataset** that provides both ball positions and motion blur attributes.  
Download here: [NextCloud](https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7).

After downloading, update the dataset config:

```yaml
# src/configs/dataset/tabletennis.yaml
root_dir: <path_to_dataset>
```

## Weights

All trained model weights for BlurBall, WASB, TrackNetv2, ResTrackNetv2, BallSeg, DeepBall, DeepBall large and Monotrack are available here: [Nextcloud](https://cloud.cs.uni-tuebingen.de/index.php/s/6Z8TpM3sXRKHzGC)

## Installation

```
pip install -r requirements.txt
```

Set the root of the repository correctly in src/config/global:
```
WASB_ROOT=<path to root of repo>
```

## Evaluation
Once the weights and dataset are downloaded, you can evaluate the different models as follows:

### BlurBall

#### 3-step inference
```
python src/main.py --config-name=eval_blurball detector.model_path=<path_to_blurball>
```

#### 1-step inference (recommended with threshold=0.7):
```
python src/main.py --config-name=eval_blurball detector.model_path=<path_to_blurball> detector.step=1 detector.postprocessor.score_threshold=0.7
```

### For the other models
```
python src/main.py --config-name=eval_<model_name> runner=eval detector.model_path=<path_to_model>
```

## Inference

BlurBall uses a multi-frame MIMO setup and is sensitive to duplicated frames (common in re-encoded online videos). During inference, the script will automatically generate a directory of unique frames.

Run inference on a video:

```
python main.py --config-name=inference_<model> detector.model_path=<path to corresponding model> +input_vid=<path to vid>
```

### BlurBall parameters
- Step size: trade off between accuracy and speed
- Score threshold: recommended 0.7 for 1-step inference

Example:
```
python main.py --config-name=inference_blurball detector.model_path=<path to corresponding model> +input_vid=<path to vid> detector.step=1 detector.postprocessor.score_threshold=0.7
```

## Training
Train BlurBall from scratch:

```
python src/main.py --config-name=train_blur
```

To fine-tune on your own dataset:

- Add images and GT CSV labels in the same format.  
- Update `src/configs/dataset/tabletennis.yaml` with your dataset name.

## Citation

If you use this work, please cite:

```bibtex
@article{gossard2025blurball,
  title   = {BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking},
  author  = {Thomas Gossard and Filip Radovic and Andreas Ziegler and Andreas Zell},
  journal = {arXiv preprint arXiv:2509.18387},
  year    = {2025}
}
