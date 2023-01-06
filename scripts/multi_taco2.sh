# Tacotron2 training script
mkdir -p tacotron2

python -m multiproc train.py \
       -d ./ \
       -m Tacotron2 \
       -o ./tacotron2/ \
       -lr 1e-3 \
       --epochs 1501 \
       -bs 154 \
       --weight-decay 1e-6 \
       --grad-clip-thresh 1.0 \
       --cudnn-enabled \
       --log-file taco2.json \
       --anneal-steps 500 1000 1500 \
       --anneal-factor 0.3 \
       --load-mel-from-disk \
       --sampling-rate 22050 \
       --training-files=filelists/ljs_mel_text_train_filelist.txt \
       --validation-files=filelists/ljs_mel_text_val_filelist.txt \
       --amp \
       --epochs-per-checkpoint 100