# SRP-DNN
A python implementation of “**<a href="https://ieeexplore.ieee.org/document/9746624" target="_blank">SRP-DNN: Learning direct-path phase difference for multiple moving sound source localization</a>**”, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

+ **Contributions** 
  - **Learning competing and time-varying direct-path inter-channel phase differences (or IPD sequence) for multiple moving sources**
    - avoids the assignment ambiguity and the problem of uncertain output-dimension encountered when simultaneously predicting multiple targets
    - exhibits reliable peaks around the actual directions of sources by the constructed spatial spectrum
    <div align=center>
    <img src=https://user-images.githubusercontent.com/74909427/218167346-1bd30853-3a6c-4d94-995e-8f8b9cde2cba.png width=55% />
    </div>
  - **Iterative source detection and localization**
    - separates the merged peaks of spatial spectrum caused by the interaction between sources
    - achieves superior performance for the azimuth and elevation estimation of multiple moving sound sources
    <div align=center>
    <img src=https://user-images.githubusercontent.com/74909427/218168115-19187bbe-05fc-449c-b7ff-ad27bf2168e4.png width=65% />
    </div>
+ **Suited cases** 
  - good or adverse noisy and reverberant scenario
  - single or multiple sound sources
  - static or moving source sources
  - the number of sound sources is known or unknown
  - different topologies of microphone arrays
        

## Datasets
+ **Source signals**: from <a href="http://www.openslr.org/12/" target="_blank">LibriSpeech database</a> 
+ **Real-world multi-channel microphone signals**: from <a href="https://www.locata.lms.tf.fau.de/datasets/" target="_blank">LOCATA database</a> 
  
## Quick start
+ **Preparation**
  - copy the train-clean-100, dev-clean and test-clean folders of LibriSpeech database to SRP-DNN/data/SrcSig/LibriSpeech
  - install: numpy, scipy, soundfile, tqdm, matplotlib, <a href="https://github.com/DavidDiazGuerra/gpuRIR" target="_blank">gpuRIR</a>, <a href="https://github.com/wiseman/py-webrtcvad" target="_blank">webrtcvad</a>, etc.
 
+ **Training**
  ```
  python RunSRPDNN.py --train --gen-on-the-fly --gpu-id [*] (--use-amp)
  ```
+ **Evaluation**
  - use GPU
  ```
  python RunSRPDNN.py --test --gpu-id [*] --time 00000001 --eval-mode locata pred eval (--use-amp)
  ```
  - use CPU
  ```
  python RunSRPDNN.py --test --no-cuda --time 00000001 --eval-mode locata pred eval (--use-amp)
  ```
+ **Pretrained models**
  - exp/00000002/best_model.tar 

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

## Reference code 
+ <a href="https://github.com/DavidDiazGuerra/Cross3D" target="_blank">Cross3D</a> 

## Licence
MIT
