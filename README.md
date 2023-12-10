# MDNet: Multi Decoder Network for Polyp Segmentation in Colonoscopy

## Overview
**MDNet** is an automatic polyp segmentation method for colorectal polyps, helping endoscopists to standardize colonoscopy examinations and evaluation tasks. It is an encoder-decoder network that uses the pre-trained ResNet50 as the encoder and multiple different decoder networks. Each decoder network is connected to a different part of the encoder via some convolutional layers. With each decoder network, we aim to increase the depth of the network to predict three different segmentation masks. The proposed MDNet effectively refines the segmentation mask with a high dice coefficient approx **85\%** for most datasets and achieves a real-time processing speed of $\textbf{64}$ frames per second. Extensive experiments on four publicly available polyp datasets demonstrate that our proposed MDNet outperforms all other baseline approaches and is more interpretable and robust compared to the other baseline models. 

## Key Features:

- **Multi-Decoder Learning:** MDNet features multiple decoders, each linked to different stages of the encoder, enhancing the detail and accuracy of the segmentation.
- **High Performance:** The architecture achieves approximately 85% dice coefficient, indicating a high level of precision in polyp detection.
- **Real-Time Processing:** Capable of processing at 64 frames per second, MDNet is optimized for real-time applications in medical diagnostics.
- **Robust Testing:** The architecture has been thoroughly tested on four publicly available polyp datasets, showing its robustness and superiority over baseline models.
- **Interpretability through Visualization:** MDNet provides visual insights into its decision-making process, aiding medical professionals in understanding and trusting its outputs.
- **State-of-the-Art Results:** The system demonstrates exceptional performance, setting new benchmarks in the field of medical image analysis.



## MDNet Architecture 
<p align="center">
<img src="Img/MDNet.jpg">
</p>


## Architecture Advantages:
- Improved accuracy for medical image segmentation.
- Efficient learning of hierarchical features.
- Ability to capture long-range spatial dependencies.

  
## Uses of MDNet:
- Medical Image Segmentation 
- General Image Segmentation
- Anomaly Detection in Medical Images 
- Comparative Studies

## Dataset 
Kvasir dataset(https://datasets.simula.no/kvasir-seg/)


## Results
 **Qualitative results comparison of the SOTA methods** <br/>
<p align="center">
<img src="Img/MDNet_Qualitative.jpg">
</p>

## Citation
Please cite our paper if you find the work useful: 
<pre>
@INPROCEEDINGS{8959021,
  author={D. {Jha},  K. {Biswas}, N. {Tomar}, U. {Bagci}},
  title={MDNet: Multi Decoder Network for Polyp Segmentation in Colonoscopy}, 
  year={2023}
</pre>

## Contact
Please contact debeshjha1@gmail.com for any further questions.
