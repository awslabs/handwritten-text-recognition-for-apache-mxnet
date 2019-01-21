# Handwritten Text Recognition (OCR) with MXNet Gluon 

These notebooks have been created by [Jonathan Chung](https://github.com/jonomon), as part of his internship as Applied Scientist @ Amazon AI, in collaboration with [Thomas Delteil](https://github.com/ThomasDelteil) who built the original prototype.

![](https://cdn-images-1.medium.com/max/1000/1*nJ-ePgwhOjOhFH3lJuSuFA.png)

The pipeline is composed of 3 steps:
- Detecting the handwritten area in a form [[blog post](https://medium.com/apache-mxnet/page-segmentation-with-gluon-dcb4e5955e2)], [[jupyter notebook](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/1_b_paragraph_segmentation_dcnn.ipynb)], [[python script](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/ocr/scripts/paragraph_segmentation_dcnn.py)]
- Detecting lines of handwritten texts [[blog post](https://medium.com/apache-mxnet/handwriting-ocr-line-segmentation-with-gluon-7af419f3a3d8)], [[jupyter notebook](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/2_line_word_segmentation.ipynb)], [[python script](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/word_and_line_segmentation.py)]
- Recognising characters and applying a language model to correct errors. [[blog post](https://medium.com/apache-mxnet/handwriting-ocr-handwriting-recognition-and-language-modeling-with-mxnet-gluon-4c7165788c67)], [[jupyter notebook](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/3_handwriting_recognition.ipynb)], [[python script](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/ocr/scripts/handwriting_line_recognition.py)]

The entire inference pipeline can be found in this [notebook](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/0_handwriting_ocr.ipynb). See the *pretrained models* section for the pretrained models.

A recorded talk detailing the approach is available on youtube. [[video](https://www.youtube.com/watch?v=xDcOdif4lj0)]

The corresponding slides are available on slideshare. [[slides](https://www.slideshare.net/apachemxnet/ocr-with-mxnet-gluon)]

## Pretrained models:


You can get the models by running `python get_models.py`:

- {[deletion](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/deletion_costs.txt), [insertion](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/insertion_costs.txt), [substitute](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/substitute_costs.txt)}_costs.txt: text files containing matrixes of weights used by the weighted edit distance (in class [OcrDistanceMeasure](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/ocr/utils/lexicon_search.py)). Files were generated with [this notebook](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/5_a_character_error_distance.ipynb).
- [paragraph_segmentation2.params](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/paragraph_segmentation2.params) generates a region of handwritten text. It was generated with [this file](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/ocr/scripts/paragraph_segmentation_dcnn.py) with the following commands:

    * `python -m ocr.paragraph_segmentation_dcnn -g 0 -r 0.001 -e 181 -n cnn_mse.params -y 0.15`
    * `python -m ocr.paragraph_segmentation_dcnn -g 0 -r 0.0001 -l iou -e 150 -n cnn_iou.params -f cnn_mse.params`
    
- [word_segmentation2.params](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/word_segmentation2.params) generates word crops. It was generated with [this  file](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/ocr/scripts/word_and_line_segmentation.py). With the following command: `python -m ocr.word_and_line_segmentation--min_c 0.01 --overlap_thres 0.10 --topk 150 --epoch 401`

- [handwriting_line_sl_160_a_512_o_2.params](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/handwriting_line_sl_160_a_512_o_2.params): pre-trained models for CNN-biLSTM for handwriting detection. Model was generated with [this file](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/ocr/scripts/handwriting_line_recognition.py). With the following command: `python -m ocr.handwriting_line_recognition --epochs 501 -n handwriting_line.params -g 1 -l 0.0001 -x 0.1 -y 0.1 -j 0.15 -k 0.15 -p 0.75 -o 2 -a 512 -sl 160 -g 1`

## Sample results

![](https://cdn-images-1.medium.com/max/2000/1*8lnqqlqomgdGshJB12dW1Q.png)

The greedy, lexicon search, and beam search outputs present similar and reasonable predictions for the selected examples. In Figure 6, interesting examples are presented. The first line of Figure 6 show cases where the lexicon search algorithm provided fixes that corrected the words. In the top example, “tovely” (as it was written) was corrected “lovely” and “woved” was corrected to “waved”. In addition, the beam search output corrected “a” into “all”, however it missed a space between “lovely” and “things”. In the second example, “selt” was converted to “salt” with the lexicon search output. However, “selt” was erroneously converted to “self” in the beam search output. Therefore, in this example, beam search performed worse. In the third example, none of the three methods significantly provided comprehensible results. Finally, in the forth example, the lexicon search algorithm incorrectly converted “forhim” into “forum”, however the beam search algorithm correctly identified “for him”.

## Dataset:
* To use test_iam_dataset.ipynb, create credentials.json using credentials.json.example and editing the appropriate field. The username and password can be obtained from http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php.

## Appendix

### 1) Handwritten area

#####  Model architecture

![](https://cdn-images-1.medium.com/max/1000/1*AggJmOXhjSySPf_4rPk4FA.png)

##### Results

![](https://cdn-images-1.medium.com/max/800/1*HEb82jJp93I0EFgYlJhfAw.png) 

### 2) Line Detection

##### Model architecture

![](https://cdn-images-1.medium.com/max/800/1*jMkO7hy-1f0ZFHT3S2iH0Q.png)

##### Results

![](https://cdn-images-1.medium.com/max/1000/1*JJGwLXJL-bV7zsfrfw84ew.png)

### 3) Handwritten text recognition

##### Model architecture

![](https://cdn-images-1.medium.com/max/800/1*JTbCUnKgAySN--zJqzqy0Q.png)

##### Results

![](https://cdn-images-1.medium.com/max/2000/1*8lnqqlqomgdGshJB12dW1Q.png)

## SClite installation
1) Download sctk-2.4.10 ftp://jaguar.ncsl.nist.gov/pub/sctk-2.4.10-20151007-1312Z.tar.bz2
2) Put it into the utils folder
3) Untar sctk-2.4.10
4) Install sctk-2.4.10 by following sctk-2.4.10/INSTALL
5) Check sctk-2.4.10/bin contains built programs
