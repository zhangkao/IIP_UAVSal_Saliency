# IIP_UAVSal_Saliency

It is a re-implementation code for the UAVSal model. 

Related Project
* Kao Zhang, Zhenzhong Chen, Shan Liu. A Spatial-Temporal Recurrent Neural Network for Video Saliency Prediction. IEEE Transactions on Image Processing (TIP), vol. 30, pp. 572-587, 2021. <br />
Github: https://github.com/zhangkao/IIP_STRNN_Saliency

* Kao Zhang, Zhenzhong Chen. Video Saliency Prediction Based on Spatial-Temporal Two-Stream Network. IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), vol. 29, no. 12, pp. 3544-3557, 2019. <br />
Github: https://github.com/zhangkao/IIP_TwoS_Saliency


## Installation 
### Environment:
The code was developed using Python 3.6+ & pytorch 1.4+ & CUDA 10.0+. There may be a problem related to software versions.
* Windows10/11 or Ubuntu20.04
* Anaconda latest, Python 
* CUDA, CUDNN

### Python requirements
You can try to create a new environment in anaconda, as follows

    *For GEFORCE RTX 10 series, such as GTX1080, xp, etc. (Pytorch 1.4.0~1.7.1, python=3.6~3.8)

        conda create -n uavsal python=3.8
        conda activate uavsal
        conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
        pip install numpy hdf5storage h5py==2.10.0 scipy matplotlib opencv-python scikit-image torchsummary

    *For GEFORCE RTX 30 series, such as RTX3060, 3080, etc.
        
        conda create -n uavsal python=3.7
        conda activate uavsal
        conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
        pip install numpy hdf5storage h5py==2.10.0 scipy matplotlib opencv-python scikit-image torchsummary


### Pre-trained models
Download the pre-trained models and put the pre-trained model into the "weights" file.

* **UAVSal-UAV2** 
coming soon

* **UAVSal-AVS1K** 
coming soon       

### Train and Test

**The parameters**

* Please change the working directory: "dataDir" to your path in the "Demo_Test.py" and "Demo_Train_DIEM.py" files, like:

        dataDir = 'E:/DataSet'
        
* More parameters are in the "train" and "test" functions.
* Run the demo "Demo_Test.py" and "Demo_Train_DIEM.py" to test or train the model.

**The full training process:**


* We initialize the SRF-Net with the pretrained MobileNet V2 and fine-tune the model on SALICON dataset. Then we train the whole model on EyeTrackUAV2  and AVS1K, respectively.

**The training and testing datasets:**

* **Training dataset**: 
[SALICON(2015)](http://salicon.net/), 
[UAV2](https://www.mdpi.com/2504-446X/4/1/2/), and 
[AVS1K](http://cvteam.buaa.edu.cn/papers.html/)
* **Testing dataset**: 
[UAV2-TE](https://www.mdpi.com/2504-446X/4/1/2/) and
[AVS1K-TE](http://cvteam.buaa.edu.cn/papers.html/)


**The training and test data examples:**
* **Training data example**: 
[UAV2](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/ET1Fa3CqLyxCrpsCwF8gM-8BTJye0OLztTl5vigg-Kr7gw?e=iFMLga) (376M)
[AVS1K](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EbxQR0fnsppEnVD4Y7SCELIBgYSuAjYct1stVXQcxAGivQ?e=6g5QOc) (147M)
* **Testing data example**:
[UAV2-TE](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EaAkpNbZ0YxCtEKnLid4BpwBtWfm4KcrsM3qDmAn4jNX_A?e=jBd8Df) (483M)
[AVS1K-TE](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EeHjqpW3aetAqRmtKt7UCU8BtZirn1PsfIhT8GgWRlPzPQ?e=WiyANH) (29M)




### Output
And it is easy to change the output format in our code.
* The results of video task is saved by ".mat"(uint8) formats.
* You can get the color visualization results based on the "Visualization Tools".
* You can evaluate the performance based on the "EvalScores Tools".


**Results**: [ALL](https://whueducn-my.sharepoint.com/:f:/g/personal/zhangkao_whu_edu_cn/EucCA9ArT1NIqpEokhDjzSMBivD86OFdKrtuzUvHw9UIJA?e=J2ZdBh) (6.2G):

The model is trained using Adam optimizer with lr=0.0001 and weight_decay=0.00005    
* **Version V1** : 
[UAV2-TE](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/ET2r9UlJ4R1Dkc5eLIq_qr0BfwEy9VIXreb5zElzPAy9vQ?e=oMepX9) (707M)
[AVS1K-TE](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EWjG4vOefPZItLWE0L1eGbkBAsHgVUsbK1AU6tbbXwWZNA?e=KAiTSQ) (1.8GM), 

The model is trained using Adam optimizer with lr=0.00001 and weight_decay=0.000005 
* **Version V2** : 
[UAV2-TE](https://whueducn-my.sharepoint.com/:u:/g/personal/zhangkao_whu_edu_cn/EUuGfQaPiVFAi42YUnyzHzgBVyqhG2InQXKIyupJxUuEYw?e=wQodTB) (654M), 
[AVS1K-TE]() (1.9G)

Time cost: [PNGs](https://whueducn-my.sharepoint.com/:f:/g/personal/zhangkao_whu_edu_cn/Eka_swtqjChAh9JycJRD1PYBDR6HjFUqbXREOvGRuEDokw?e=FIjd6h) (repeat 4 times for test.)

## Paper & Citation

If you use the STRNN video saliency model, please cite the following paper: 
```
@article{zhang2021an,
  title={An Efficient Saliency Prediction Model for Unmanned Aerial Vehicle Video},
  author={Zhang, Kao and Chen, Zhenzhong and Li, Songnan},
  journal={xxxx},
  volume={xxxx},
  pages={xxxx},
  year={xxxx}
}
```

## Contact
Kao ZHANG  <br />
Laboratory of Intelligent Information Processing (LabIIP)  <br />
Wuhan University, Wuhan, China.  <br />
Email: zhangkao@whu.edu.cn  <br />

Zhenzhong CHEN (Professor and Director) <br />
Laboratory of Intelligent Information Processing (LabIIP)  <br />
Wuhan University, Wuhan, China.  <br />
Email: zzchen@whu.edu.cn  <br />
Web: http://iip.whu.edu.cn/~zzchen/  <br />