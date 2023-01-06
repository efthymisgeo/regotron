#!/bin/bash
bash generate_wav.sh \
        en_phrases \
        taco2_output_folder \
        pretrained_tacotron2/checkpoint_Tacotron2_1500.pt \
        vocoder/nvidia_waveglowpyt_fp32_20190427.pt

bash generate_wav.sh \
        en_phrases \
        rego_output_folder \
        pretrained_regotron/checkpoint_Tacotron2_1500.pt \
        vocoder/nvidia_waveglowpyt_fp32_20190427.pt