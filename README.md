# MDNet: Multi-Decoder Network for Abdominal CT Organs Segmentation

## Overview
**MDNet** Accurate segmentation of organs from abdominal CT scans is essential for clinical applications such as diagnosis, treatment planning, and patient monitoring. To handle the challenges of heterogeneity in organ shapes, sizes, and complex anatomical relationships, we propose a Multi decoder network (MDNet), an encoder decoder network that uses the pre-trained MiT-B2 as the encoder and multiple different decoder networks. Each decoder network is connected to a different part of the encoder via a multi-scale feature enhancement dilated block. With each decoder, we increase the depth of the network iteratively and refine
segmentation masks, enriching feature maps by integrating the previous decoders’ feature maps. To refine the feature map further, we also utilize the predicted masks from the previous decoder to the current decoder to provide spatial attention across foreground and background regions. MD-Net effectively refines the segmentation mask with a high dice similarity coefficient (DSC) of 0.9013 and 0.9169 on the Liver Tumor segmentation (LiTS) and MSD Spleen datasets. Additionally, it reduces Hausdorff distance (HD) to 3.79 for the LiTS dataset and 2.26 for the spleen segmentation dataset, underscoring the precision of MDNet in capturing the complex contours. Moreover, MDNet is more interpretable and robust compared to the other baseline models. 

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
