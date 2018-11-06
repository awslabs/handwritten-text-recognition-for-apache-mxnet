# Handwritten Text Recognition (OCR) with MXNet Gluon 

These notebooks have been created by [Jonathan Chung](https://github.com/jonomon), as part of his internship as Applied Scientist @ Amazon AI, in collaboration with [Thomas Delteil](https://github.com/ThomasDelteil) who built the original prototype.

![](https://cdn-images-1.medium.com/max/1000/1*nJ-ePgwhOjOhFH3lJuSuFA.png)

The pipeline is composed of 3 steps:
- Detecting the handwritten area in a form [[blog post](https://medium.com/apache-mxnet/page-segmentation-with-gluon-dcb4e5955e2)], [[jupyter notebook](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/paragraph_segmentation_dcnn.ipynb)], [[python script](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/paragraph_segmentation_dcnn.py)]
- Detecting lines of handwritten texts [[blog post](https://medium.com/apache-mxnet/handwriting-ocr-line-segmentation-with-gluon-7af419f3a3d8)], [[jupyter notebook](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/line_segmentation.ipynb)], [[python script](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/line_segmentation.py)]
- Recognising characters and applying a language model to correct errors. [[blog post](https://medium.com/apache-mxnet/handwriting-ocr-handwriting-recognition-and-language-modeling-with-mxnet-gluon-4c7165788c67)], [[jupyter notebook](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/handwriting_recognition.ipynb)], [[python script](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/handwriting_line_recognition.py)]

The entire inference pipeline can be found in this [notebook](https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet/blob/master/handwriting_ocr.ipynb). See the *pretrained models* section for the pretrained models.

A recorded talk detailing the approach is available on youtube. [[video](https://www.youtube.com/watch?v=xDcOdif4lj0)]

The corresponding slides are available on slideshare. [[slides](https://www.slideshare.net/apachemxnet/ocr-with-mxnet-gluon)]

## Pretrained models:

- {[deletion](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/deletion_costs.txt), [insertion](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/insertion_costs.txt), [substitute](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/substitute_costs.txt)}_costs.txt: text files containing matrixes of weights used by the weighted edit distance (in class [OcrDistanceMeasure](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/language_model/utils/lexicon_search.py)). Files were generated with [this notebook](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/language_model/model_distance.ipynb).
- [handwriting_line_recognition3.params](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/handwriting_line_recognition3.params): pre-trained models for CNN-biLSTM for handwriting detection. Model was generated with [this file](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/language_model/handwriting_line_recognition.py). With the following command: `python handwriting_line_recognition.py --epochs 501 -n handwriting_line.params -g 1 -l 0.0001 -x 0.1 -y 0.1 -j 0.15 -k 0.15 -p 0.75 -o 2 -a 128`
- [word_segmentation.params](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/word_segmentation.params) generates word crops. It was generated with [this  file](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/language_model/word_segmentation.py). With the following command: `python line_segmentation.py --min_c 0.01 --overlap_thres 0.10 --topk 150 --epoch 401`
- [paragraph_segmentation.params](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/paragraph_segmentation.params) generates a region of handwritten text. It was generated with [this file](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/language_model/paragraph_segmentation_dcnn.py) with the following commands:
        * 1) `python paragraph_segmentation_dcnn.py -r 0.001 -e 301 -n cnn_mse.params`
        * 2) `python paragraph_segmentation_dcnn.py -r 0.0001 -l iou -e 150 -n cnn_iou.params -f cnn_mse.params`
 - [GoogleNews-vectors-negative300.bin.gz](https://s3.us-east-2.amazonaws.com/gluon-ocr/models/GoogleNews-vectors-negative300.bin.gz): a word2vec model that is used by pycontractions. This was saved here for convenience purposes. 

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