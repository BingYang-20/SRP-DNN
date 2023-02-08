# SRP-DNN
A python implementation of “**<a href="https://ieeexplore.ieee.org/document/9746624" target="_blank">SRP-DNN: Learning direct-path phase difference for multiple moving sound source localization</a>**”, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

## Datasets
+ **Source signals**: from <a href="http://www.openslr.org/12/" target="_blank">LibriSpeech database</a> 
+ **Real-world multi-channel microphone signals**: from <a href="https://www.locata.lms.tf.fau.de/datasets/" target="_blank">LOCATA database</a> 
  
## Quick start
+ **Preparation**
  - copy the train-clean-100, dev-clean and test-clean folders of LibriSpeech database to SRP-DNN/data/SouSig/LibriSpeech
  - install: numpy, scipy, soundfile, tqdm, matplotlib, <a href="https://github.com/DavidDiazGuerra/gpuRIR" target="_blank">gpuRIR</a>, <a href="https://github.com/wiseman/py-webrtcvad" target="_blank">webrtcvad</a>, etc.
 
+ **Training**
  ```
  python RunSRPDNN.py --train --gpu-id [*] (--use-amp)
  ```
+ **Evaluation**
  - use GPU
  ```
  python RunSRPDNN.py --test --gpu-id [*] --time 00000000 --eval-mode locata pred eval (--use-amp)
  ```
  - use CPU
  ```
  python RunSRPDNN.py --test --no-cuda --time 00000000 --eval-mode locata pred eval (--use-amp)
  ```
+ **Pretrained models**
  - exp/00000000/best_model.tar 

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{yang2022srpdnn,
    author = "Bing Yang and Hong Liu and Xiaofei Li",
    title = "SRP-DNN: Learning direct-path phase difference for multiple moving sound source localization",
    booktitle = "Proceedings of {IEEE} International Conference on Acoustics, Speech and Signal Processing (ICASSP)",
    year = "2022",
    pages = "721-725"}
```

## Licence
MIT
