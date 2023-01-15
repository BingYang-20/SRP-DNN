# SRP-DNN
A python implementation of “**<a href="https://ieeexplore.ieee.org/document/9746624" target="_blank">SRP-DNN: Learning direct-path phase difference for multiple moving sound source localization</a>**”, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

## Datasets
+ **Source signals**: from <a href="http://www.openslr.org/12/" target="_blank">LibriSpeech database</a> 
+ **Real-world multi-channel microphone signals**: from <a href="https://www.locata.lms.tf.fau.de/datasets/" target="_blank">LOCATA database</a> 
  
## Quick start

+ **Preparation**
  - 

+ **Training**
  ```
  python RunSRPDNN.py --train --gpu-id [*]
  ```
+ **Test**
  ```
  python RunSRPDNN.py --test --gpu-id [*] 
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
