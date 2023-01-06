# Regotron

## Repo Description
Source code for the paper "[Regotron: Regularizing the Tacotron2 architecture via monotonic alignment loss](https://arxiv.org/abs/2204.13437)"

## Regotron Paper

### Regotron = Regularized Tacotron2
Regotron is a  regularized Tacotron2 version. Specifically, it penalizes the weights in the attention mechanism in order to be monotonic. The essential modification is an additional loss function term which acts as a regularizer. 

### Why use Regotron
Our results in LJSpeech Dataset show that Regotron 
- builds a monotonic alignment quicker (compared to Taco2)
- is more stable during training (no spily behavior)
- is more robust (less common TTS mistakes)
- improves MOS (compared to Taco2)
- minimal training overhead (+1 loss term) and same inference cost/time

## Setup

This repo is built upon Nvidia's DeepLearningExamples Tacotron2 [implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2). We use an english pretrained WaveGlow vocoder. 


### Requirements

The following components are required:

- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch 21.05-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)

The [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) dataset is also required (or any other speech dataset in the LJSpeech filelist format).

## Regotron Training

To train Regotron use the following steps 

1. Clone the repository.
   ```bash
   git clone https://github.com/efthymisgeo/regotron
   ```

2. Download and preprocess the dataset.
Use the `./scripts/prepare_dataset.sh` download script to automatically
download and preprocess the training, validation and test datasets. To run
this script, issue:
   ```bash
   bash scripts/prepare_dataset.sh
   ```

   Data is downloaded to the `./LJSpeech-1.1` directory (on the host).  The
   `./LJSpeech-1.1` directory is mounted to the `/workspace/tacotron2/LJSpeech-1.1`
   location in the NGC container.

1. Build the Regotron/Tacotron2/WaveGlow container.
   ```bash
   # FIX ME
   bash scripts/docker/build.sh
   ```

2. Start an interactive session in the NGC container to run training/inference.
After you build the container image, you can start an interactive CLI session with:

   ```bash
   # FIX ME
   bash scripts/docker/interactive.sh
   ```

   The `interactive.sh` script requires that the location on the dataset is specified.
   For example, `LJSpeech-1.1`. 
   
1. To preprocess raw speech data and produce mels for Regotron training, use
   the `./scripts/prepare_mels.sh` script:
   ```bash
   bash scripts/prepare_mels.sh
   ```

   The preprocessed mel-spectrograms are stored in the `./LJSpeech-1.1/mels` directory.

2. Train Regotron
   ```bash
   bash scripts/multi_regotron.sh 
   ```
   For training Tacotron2 with the setup in the paper
   ```bash
   bash scripts/multi_taco2.sh
   ```

3. Inference (Generate Speech)

   You will need to have already trained Regotron/Tacotron2 by this step,
   or download a pretrained version from the Nvidia hub or the link in this
   repo. For vocoder we use pretained WaveGlow. Store Regotron checkpoint under
   `pretrained_rego`, Tacotron2 checkpoint under `pretrained_tacotron2` and
   WaveGlow under `vocoder` folder. 
   
   This script generates speech based on the Regotron model
   ```bash
   bash generate_wav.sh \
        en_phrases \
        rego_output_folder \
        pretrained_regotron/checkpoint_Tacotron2_1500.pt \
        vocoder/nvidia_waveglowpyt_fp32_20190427.pt
   ```

## Repo Structure & Details

### Imporant Changes
- `tacotron2`: has the source code for the Tacotron2/Regotron architecture
- `tacorton2/loss_function.py`: has the Regotron loss

### Hyper-Parameters

* `--epochs` - number of epochs (default: 1501)
* `--learning-rate` - learning rate (default: 1e-3)
* `--batch-size` - batch size (default FP16: 104)
* `--amp` - use mixed precision training
* `--cpu` - use CPU with TorchScript for inference
* `--sampling-rate` - sampling rate in Hz of input and output audio (22050)
* `--filter-length` - (1024)
* `--hop-length` - hop length for FFT, i.e., sample stride between consecutive FFTs (256)
* `--win-length` - window size for FFT (1024)
* `--mel-fmin` - lowest frequency in Hz (0.0)
* `--mel-fmax` - highest frequency in Hz (8.000)
* `--anneal-steps` - epochs at which to anneal the learning rate (500 1000 1500)
* `--anneal-factor` - factor by which to anneal the learning rate (FP16/FP32: 0.3/0.1)

#### Regotron additional parameters

* `--enable-align-loss` - use this argument to enable Regotron loss
* `--delta-align` - $\delta$, relaxation hyperparam, default=0.01
* `--weight-align` - $\lambda$, monotonic loss weight, default=1e-5
