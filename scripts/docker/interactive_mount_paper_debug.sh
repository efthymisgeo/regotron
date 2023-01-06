#!/bin/bash

nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --ipc=host -v $PWD:/workspace/tacotron2/ -v /home/efthygeo/projects/speech-synthesis/nvidia_tacotron2/LJSpeech-1.1:/workspace/tacotron2/LJSpeech-1.1 -v /data/efthygeo/tts/en/regotron/best/checkpoint_Tacotron2_last.pt:/workspace/tacotron2/pretrained_regotron/checkpoint_Tacotron2_last.pt -v /data/efthygeo/tts/en/waveglow_nvidia/nvidia_waveglowpyt_fp32_20190427.pt:/workspace/tacotron2/vocoder/nvidia_waveglowpyt_fp32_20190427.pt tacotron2 bash
