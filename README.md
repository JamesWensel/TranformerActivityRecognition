# TranformerActivityRecognition
Transformer and Vision Transformer Implementions (according to [^1] and [^2]) used to perform Activity Recognition alongside LSTM and ResNet50 using TensorFlow

### Datasets
Activity Recognition Training done using [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset
Image Classification Training (note: ResNet was pretrained) done using [CIFAR-101](https://www.cs.toronto.edu/~kriz/cifar.html)

<br />

## File Descriptions: 
#### ***transformer.py***
Positional Encoding and Encoder custom layers for transformer (no decoder layer as outlined in the report). <br />
Also includes a Build function to build a Transformer with only an Encoder.

#### ***image_transformation.py***
Custom layer to split images into patches for Vision Transformer (as outlined in report and Vision Transformer paper [^2]) 

#### ***build_models.py***
Creates one of four Neural Network Models: 
  - ResNet Model 
  - LSTM Model
  - Transformer Model 
  - Vision Transformer Model 
These can be chained together to create a model capable of performing Activity Recognition, or used on their own for individual testing. 

#### ***run_model***
Performs training and predicting on a provided model using provided data, and outputs the results. Also used to print a Plot of the model used

#### ***output_data***
Helper file to output training, testing, and timing results

#### ***format_video_files.py***
Uses OpenCV to extract a specified number of frames from each Activity Recognition video in the [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset and
resizes them. <br /> 
Saves formatted video files to a Numpy array of shape (# of Videos, # of Frames, Image Height, Image Width, 3 [Only allows RGB videos]) <br /> 
Also creates a Numpy array containing associated label for each video file. 

#### ***feature_extractor.py***
Creates a ResNet50 Model and performs feature extraction on the formatted video files, saving them to a seprate Numpy array. <br /> 
This is used primarily to show the effects and speed of a Transformer vs a LSTM network for classification, and allows us to skip the time required to perform
feature extraction on each frame, as would be necessary in a complete Activity Recognition Classifier. 

#### ***cifar100_preprocessing.py***
Performs preprocessing steps for [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and saves result to a Numpy array for image classification
training. 

<br /> 

## Important Papers
[^1]: A. Viswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention Is All You Need,” 31st Conference on Neural Information Processing Systmes (NIPS), Long Beach, CA, USA. , 2017. 
[^2]: A. Dosovitskiy , L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An Image Worth 16x16 Words: Transformers for Image Recognition at Scale,” in International Conference on Learning Representation, 2021. 
