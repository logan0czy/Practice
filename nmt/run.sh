#!/bin/bash

if [ "$1" = "train" ]; then
    python run.py train --use_word2vec --max_epoch=20 --vocab_size=10000 --batch_size=64 --lr=0.001 --dev_info_interval=1000 --loss_info_interval=40
elif [ "$1" = "test" ]; then
    python run.py test
elif [ "$1" = "sanity" ]; then
    python run.py train --use_word2vec --loss_info_interval=10 --dev_info_interval=100 --max_epoch=20 --corpus_limit=100 --vocab_size=1000 --freq_cutoff=0 --batch_size=64
else
    echo "Invalid Option Selected"
fi