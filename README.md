<div align="center">
  <p>
    <a href="https://icaerus.eu" target="_blank">
      <img width="50%" src="https://icaerus.eu/wp-content/uploads/2022/09/ICAERUS-logo-white.svg"></a>
    <h3 align="center">Xylella Fastidiosa Classification🦚</h3>
    
   <p align="center">
    Training and validation of a classifier for Xylella Fastidiosa in olive trees
    <br/>
    <br/>
    <a href="https://github.com/icaerus-eu/repo-title/wiki"><strong>Explore the wiki »</strong></a>
    <br/>
    <br/>
    <a href="https://github.com/icaerus-eu/repo-title/issues">Report Bug</a>
    -
    <a href="https://github.com/icaerus-eu/repo-title/issues">Request Feature</a>
  </p>
</p>
</div>

![Downloads](https://img.shields.io/github/downloads/icaerus-eu/repo-title/total) ![Contributors](https://img.shields.io/github/contributors/icaerus-eu/repo-title?color=dark-green) ![Forks](https://img.shields.io/github/forks/icaerus-eu/repo-titlee?style=social) ![Stargazers](https://img.shields.io/github/stars/icaerus-eu/repo-title?style=social) ![Issues](https://img.shields.io/github/issues/icaerus-eu/repo-title) ![License](https://img.shields.io/github/license/icaerus-eu/repo-title) 


## Table Of Contents
- [Table Of Contents](#table-of-contents)
- [Summary](#summary)
- [Installation](#installation)
- [Documentation](#documentation)
  - [classifier.py](#classifierpy)
  - [training\_classifier.py](#training_classifierpy)
  - [validation\_classifier.py](#validation_classifierpy)
- [Acknowledgements](#acknowledgements)

## Summary
The code loads a custom dataset of images and CSV files, applies transformations to the images, and splits the data into training and validation sets. It then initializes and trains a custom YOLO model using PyTorch Lightning, with balanced class weights. The training results, including metrics and models, are saved and logged using TensorBoard and CSV loggers

## Installation
Install python 3.10 and required libraries listed in ```requirements.txt```.
The dataset is available at https://zenodo.org/records/13120763
The trained model is available at https://zenodo.org/records/13150032

## Documentation

The directory ```dataset/labels``` contains 3 CSV files, one for each health status: 1, 2, and 3+4 (we decided to merge classes 3 and 4 because they are less represented, thus balancing the dataset better):

Health Status total labels:
|Label ID|Num Trees| Description |
|:---:   | :----:  |   :----:    |
|1       |3218     |Asymptomatic olive tree|
|2       |702      |Olive tree with mild symptoms (identifiable through photos or NDVI)|
|3       |125      |Olive tree with evident symptoms|
|4       |229      |Olive tree with compromised canopy or dead|



The CSV files are organized as follows: the first column, ```health_status```, contains the health status of the olive tree (values from 1 to 3, since 3 and 4 are combined). The second column, ```NDVI_string```, contains the name of the NDVI photo identified by the tree, and ```RGB_string``` does the same for the RGB photo.

During the training sessions, we reduced the first class to around 1500 instances from 3218. These are indicated in the ```health_stat1.csv``` file in the ```dataset/labels``` directory.

In the ```dataset/labels``` directory, we have labels referring to the balanced dataset. To obtain a balanced dataset, we augmented health label 2 from 702 to 1404 and the combined class (3+4) from 354 elements to 1416 using classic geometric transformations.

Regarding the validation set, in the ```CustomDataset``` class inside the training script, we randomly select 400 samples for each class.
The photo names follow this format: ```1_DJI_20240525124035_0032_NDVI_0.JPG```. In this naming convention, the leading number ```(1_)``` indicates that the image has been generated through augmentation. This means that the original image has been modified to create this new version. The original unmodified image would be named ```DJI_20240525124035_0032_NDVI_0.JPG```, without the leading number.

For example:
```
Original image: DJI_20240525124035_0032_NDVI_0.JPG

Augmented images: 1_DJI_20240525124035_0032_NDVI_0.JPG, 2_DJI_20240525124055_0042_RGB_0.JPG, ...
```
In these examples ```1_DJI_20240525124035_0032_NDVI_0.JPG``` and ```2_DJI_20240525124055_0042_RGB_0.JPG``` are augmented versions of their respective original images.
The leading numbers ```(1_, 2_, ...)``` indicate different augmentation instances applied to the original images.
In dataset/images, all the health status 1 and augmented 2, 3+4 images are collected.

It's important to note that while the images in the augmented datasets have already been resized to ```640x640``` pixels, the other images retain their original dimensions. These will be resized later during the training process.

### classifier.py

The ```classifier.py``` file contains the implementation of the ```DualYOLO``` class, a custom neural network model built using ```PyTorch``` and ```PyTorch Lightning```. In addition to this main class, the file also defines the ```CustomDataset``` class, which is used for loading and preprocessing dataset images and labels.

The ```CustomDataset``` class is a subclass of ```torch.utils.data.Dataset``` and is designed to handle the loading of images and labels, both for training and validation of the model. It takes as parameters the paths to the CSV files containing labels (```health_dir_path```), NDVI images (```ndvi_path```), and RGB images (```rgb_path```). Additionally, you can specify the number of validation samples to extract from each CSV file (```val_samples_per_file```) and apply a series of transformations to the images via the transform object. The class also manages saving the validation set to a specified directory (```save_dir```).

A key feature of the ```CustomDataset``` class is that it handles label scaling. The labels in the dataset, originally ranging from 1 to 3, are scaled down by 1 (i.e., 1 becomes 0, 2 becomes 1, and 3 becomes 2) because the model expects input labels to be 0, 1, 2. When saving the ```validation_set.csv``` file, these labels are scaled back up by 1 to restore them to their original range.

CustomDataset includes several essential methods. The ```__len__()``` method returns the total number of samples in the dataset, while ```__getitem__(index)``` allows you to retrieve a specific sample from the dataset, returning the NDVI image, the corresponding RGB image, and the associated label. The ```get_train_labels()``` method provides the labels used for training, which can be utilized to calculate class weights or for other purposes. Finally, the ```save_validation_set()``` method allows you to save the validation set, ensuring the consistency of validation data across different training sessions.

The ```DualYOLO``` class is a PyTorch Lightning module that integrates two YOLO models and a classification head. This model is designed to take in two images as input, extract features using the YOLO models, and classify the combined features. The class parameters include class weights for handling class imbalance (```class_weights```), scaling factors for the depth and width of the YOLO model (```depth_scale``` and ```width_scale```), the learning rate (```lr```), the number of classes to classify (```num_classes```), and the directory where model checkpoints and logs will be saved (```save_dir```).

The ```forward(img1, img2)``` method performs the forward pass through the model, processing the two input images. During training, the ```training_step(batch, batch_idx)``` method handles loss calculation and logs precision, recall, and accuracy metrics. In the validation phase, the ```validation_step(batch, batch_idx)``` method evaluates the model on the validation data and updates the confusion matrix, while ```validation_epoch_end(outputs)``` processes the outputs at the end of each validation epoch, logging the confusion matrix results and accuracy. Finally, the ```configure_optimizers()``` method configures the optimizer (```Adam```) and the learning rate scheduler.

### training_classifier.py
This script is used to train the DualYOLO model using PyTorch Lightning.

Workflow:
Dataset and DataLoader Initialization:

Defines the image directories and transformation pipelines: Paths to image data (including CSV labels, NDVI, and RGB images) are set, and a series of image transformations are applied to standardize the input size and format.
Data Loading: The script creates instances of CustomDataset for training and ValDataset for validation, ensuring data is loaded efficiently and prepared for the model.

Shuffling:
```DataLoader``` class shuffles the training set to ensure that the model does not learn the order of the data, and importantly, the validation set is also shuffled. This is because the labels are ordered, and without shuffling, this could cause issues in metric calculation, especially with metrics that have problems with division by zero (```zero_division``` errors).

Class Weights Calculation:
Handles class imbalance using ```compute_class_weight``` from sklearn: Class weights are calculated to counteract any imbalance in the dataset, ensuring that underrepresented classes are given appropriate importance during training.

Model Initialization:
Initializes the ```DualYOLO``` model: The model is set up with specific configurations such as depth and width scaling, class weights, learning rate, and the number of output classes. These settings optimize the model architecture for the specific task and dataset.


Logging and Checkpointing:
Configures TensorBoard and CSV loggers: These loggers track training progress, including loss and accuracy metrics, and save them for later visualization and analysis.
Defines callbacks for model checkpoints: The script sets up callbacks to save the best model checkpoints based on validation loss, ensuring that only the most performant models are retained.


Training:
Utilizes the PyTorch Lightning Trainer: The Trainer is configured to handle GPU acceleration, mixed precision training, and other settings that streamline the training process. The training loop is initiated, where the model is iteratively trained and evaluated on the validation set.

### validation_classifier.py
This script is responsible for validating a trained DualYOLO model. The process involves loading a pre-trained model, running it on a validation dataset, and then saving the predictions for further analysis.

To begin, the script loads the DualYOLO model from a saved checkpoint and moves it to the appropriate device, whether that’s a GPU or CPU. This ensures the model is ready for evaluation.

Next, the script sets up the validation dataset using the ```CustomDataset``` class, which handles the organization and preprocessing of the data. A ```DataLoader``` is then initialized to batch the validation data, facilitating efficient processing during model inference.

Once the model is prepared and the data is loaded, the script switches the model to evaluation mode. This step is crucial as it adjusts the model’s behavior to suit the evaluation phase, such as disabling gradient computations and turning off any training-specific features like dropout layers. The script then iteratively feeds the validation data through the model, collecting predictions for each batch of images.

After completing the predictions, the script saves the results to a CSV file, including the paths to the images and the corresponding predictions. It is important to note that the output labels generated by the model are not altered or remapped to the input format like 1, 2, 3. Instead, the predictions are kept in their original form as produced by the model. 

## Acknowledgements
This project is funded by the European Union, grant ID 101060643.

<img src="https://rea.ec.europa.eu/sites/default/files/styles/oe_theme_medium_no_crop/public/2021-04/EN-Funded%20by%20the%20EU-POS.jpg" alt="https://cordis.europa.eu/project/id/101060643" width="200"/>
