# Wide-Range Image Blending - PyTorch Implementation
<img src="https://github.com/arunsandy1309/Image-Blending/assets/50144683/6cd80dcd-b547-4e85-b3db-9b8718fd086e"></br>
The objective of proposed model of wide-range image blending is learning to generate new content for the intermediate region which connects two different input photos, thus leading to a semantically coherent and spatially smooth panoramic image. Our full model is shown below, where in the following we will sequentially describe our model designs, including the image context encoder-decoder, the bidirectional content transfer module, and the contextual attention mechanism on skip connection, as well as the training details.</br></br>
This repository contains the Pytorch implementation of the following paper:
>**Bridging the Visual Gap: Wide-Range Image Blending**</br>
>Chia-Ni Lu, Ya-Chu Chang, Wei-Chen Chiu</br>
>https://arxiv.org/abs/2103.15149
>
>**Abstract:** _In this paper we propose a new problem scenario in image processing, wide-range image blending, which aims to smoothly merge two different input photos into a panorama by generating novel image content for the intermediate region between them. Although such problem is closely related to the topics of image inpainting, image outpainting, and image blending, none of the approaches from these topics is able to easily address it. We introduce an effective deep-learning model to realize wide-range image blending, where a novel Bidirectional Content Transfer module is proposed to perform the conditional prediction for the feature
representation of the intermediate region via recurrent neural networks. In addition to ensuring the spatial and semantic consistency during the blending, we also adopt the contextual attention mechanism as well as the adversarial learning scheme in our proposed method for improving the visual quality of the resultant panorama. We experimentally demonstrate that our proposed method is not only able to produce visually appealing results for wide-range image blending, but also able to provide superior performance with respect to several baselines built upon the state-of-theart image inpainting and outpainting approaches._

## Architecture
<img src="https://github.com/arunsandy1309/Image-Blending/assets/50144683/26a4dcc2-d16d-4922-b490-f91e8b4606d8"></br>
**(a) Full Model:** Our full model takes I<sub>left</sub> and I<sub>right</sub> as input, and compresses them into compact representations ˜f<sub>left</sub> and ˜f<sub>right</sub> individually via the encoder. Afterwards, our novel Bidirectional Content Transfer (BCT) module is used to predict ˜f<sub>mid</sub> from ˜f<sub>left</sub> and ˜f<sub>right</sub>. Lastly, based on the feature ˜f, which is obtained by concatenating {˜f<sub>left</sub>, ˜f<sub>mid</sub>, ˜f<sub>right</sub>} along the horizontal direction, the decoder generates our final result ˜I. Noting that there is a contextual attention mechanism on the skip connection between the encoder and decoder, which helps to enrich the texture and details of our blending result. </br>
**(b) LSDTM Encoder:** The architecture of the LSTM encoder EBCT in our BCT module, which encodes the information of ˜f<sub>left</sub> or ˜f<sub>right</sub> to generate c˜<sub>left</sub> or c˜<sub>right</sub>. </br>
**(c) LSTM Decoder:** The architecture of the conditional LSTM decoder DBCT in our BCT module, which takes the condition c˜<sub>right</sub> (respectively c˜<sub>left</sub>) as well as the input ˜f<sub>left</sub> (respectively ˜f<sub>right</sub>) to predict the feature map <sup>--></sup>f<sub>mid</sub> (respectively <sup><--</sup>f<sub>mid</sub>). The prediction of ˜f<sub>mid</sub> related to the intermediate region, which blends between ˜f<sub>left</sub> and ˜f<sub>right</sub>, is then obtained via concatenating <sup>--></sup>f<sub>mid</sub> and <sup><--</sup>f<sub>mid</sub> along the channel dimension followed by passing through a 1 × 1 convolutional layer.</br>
  
## Two-Stage Training
1. **Self-Reconstruction Stage:** We adopt the objective of self-reconstruction, where the two input photos {I<sub>left</sub>, I<sub>right</sub>} and the intermediate region are obtained from the same image. This is achieved by first splitting a wide image vertically and equally into three parts, then taking the leftmost one-third and the rightmost one-third as I<sub>left</sub> and I<sub>right</sub> respectively, while the middle one-third can be treated as the ground truth I<sub>mid</sub> for the generated intermediate region ˜I<sub>mid</sub>.</br></br>
    - We adopt the scenery dataset proposed by [Very Long Natural Scenery Image Prediction by Outpainting](https://github.com/z-x-yang/NS-Outpainting) for conducting our experiments, in which we split the dataset to 5040 training images and 1000 testing images.
    - Download the dataset with our split of train and test set from [here](https://drive.google.com/file/d/1wi3s9-_4b-UnPbkqjAXjuA4-bVyn1IAw/view) and put them under `data/`.</br></br>
    - Run the training code for self-reconstruction stage
    ```
    python train_SR.py
    ```
    - If you want to run train the model with your own dataset
    ```
    python train_SR.py --train_data_dir YOUR_DATA_PATH
    ```
2. **Fine-Tuning Stage:** We keep using the objective of self-reconstruction as the previous training stage, but additionally consider another objective which is based on the training samples of having I<sub>left</sub> and I<sub>right</sub> obtained from different images (i.e. different scenes). As there is no ground truth of ˜I<sub>mid</sub> now for such training samples, this additional training objective is then based on the adversarial learning.</br></br>
    - After finishing the training of self-reconstruction stage, move the latest model weights from `checkpoints/SR_Stage/` to `weights/` (or use [pre-train weights](https://drive.google.com/drive/folders/1CHvW6KHeLXVJSslvahEylDW70nJowWHR) from self-reconstruction stage), and run the training code for fine-tuning stage (second stage)
    ```
    python train_FT.py --load_pretrain True
    ```
    
## Testing
Download our pre-trained model weights from [here](https://drive.google.com/drive/folders/1dRpBBFAYHlbOrRjpKiW6SvdU8AnTepqy?usp=share_link) and put them under `weights/`. 

Test the sample data provided in this repo:
```
python test.py
```
Or download our paired test data from [here](https://drive.google.com/file/d/1d01cgpaEG4F1drGAE38gz_LsvUCTM6l_/view?usp=share_link) and put them under `data/`.  
Then run the testing code:
```
python test.py --test_data_dir_1 ./data/scenery6000_paired/test/input1/
               --test_data_dir_2 ./data/scenery6000_paired/test/input2/
```

Run your own data:
```
python test.py --test_data_dir_1 YOUR_DATA_PATH_1
               --test_data_dir_2 YOUR_DATA_PATH_2
               --save_dir YOUR_SAVE_PATH
```
If your test data isn't paired already, add `--rand_pair True` to randomly pair the data.
