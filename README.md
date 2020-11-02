# Using Neural Networks to Classify X-Ray Images of Pneumonia

#### Author: Lauren Esser

The contents of this repository detail an analysis of the module four project. This analysis is detailed in hopes of making the work accessible and replicable.

## Business Problem

1. **Objective:** Build a neural network that classifies x-ray images of pediatric patients to identify whether or not they have pneumonia.

2. **Project plan:** I worked daily on this project for the allowed time (one week). While working, the goal was to test parameters of basic neural networks and convolutional neural networks to see which model would provide the best accuracy and precision when classifying if an X-Ray Image has pneumonia or not. 

3. **Success criteria:** A successful model will predict correct X-Ray images above 50%. The goal I set for myself is to create a model that predicts the image correctly 80% of the time. I will know I have achieved this success criteria when I can see the classification report and confusion matrix show numbers over 80%. I also am making sure to pay attention to the loss, accuracy, precision, and recall visualizations in order to monitor gradient descent.


## Data

Data set was downloaded from Chest X-Ray Images (Pneumonia) from Kaggle. [Linked Here.](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) The provided dataset was already organized into three folders: train, test, validation. The subfolders for each group include: Pneumonia and Normal. All images come from pediatric patients aged 1 to 5 years old in Guangzhou Women and Children's Medical Center. In this notebook I uploaded the zip file from Kaggle into my Google Drive. You may access the files in my Drive [here.](https://drive.google.com/file/d/1Qy9c2iboOfnmbu8uoJ41_Rm1DamQg4i8/view?usp=sharing) 



## Methods

I used the OSEMN Method while working on this project. The steps are outlined in detail below:

<img src="https://docs.google.com/drawings/d/e/2PACX-1vS3jxe9W38P4Or4fOi2pfwCtqeWqRFTboNa4is7rkzWXYxxkcF9mWjGhFEMsOLSy4lrpCPdHuLhO-iU/pub?w=924&amp;h=487">


### Obtain:

Data was obtained from the Chest X-Ray Images (Pneumonia) dataset from Kaggle and uploaded into my Google Drive. Since I chose to complete my work on Google Colab, I had to mount my drive and show the source folder to access the correct zip file. I then unzipped the images for use. 

Example of code:

``` source_folder = r'/gdrive/My Drive/Colab Notebooks/DataSets/'
target_folder = r'/content/'
file = glob.glob(source_folder+ 'chest-xray-pneumonia-jmi.zip', recursive = True)
file = file[0]
file 
```

``` #upzip data
zip_path = file
!cp '{zip_path}' .

!unzip -q chest-xray-pneumonia-jmi.zip
!rm chest-xray-pneumonia-jmi.zip 
```

### Scrub:

To Scrub the data I first organized it into proper directories. These included a train, test, and validation directory. 

Example of code:

```
# Create Data Directory
data_dir = Path(r'chest_xray/')

# Train directory
train_dir = data_dir/'train'
pneumonia_train_dir = data_dir/'train'/'PNEUMONIA'
normal_train_dir = data_dir/'train'/'NORMAL'
```

I then checked the length of the directory to see what I was working with. For this project I went back and forth between Explore and Scrub so you will see farther details in the next section.

### Explore:

To begin my exploration section I wanted to first see the images I was working with. I created a function to view images and took a look at a normal train image and a pneumonia train image. 

Example of code:

```
# See image function
def see_image(img_file):
  filename = img_file
  img1 = load_img(filename, target_size = (150, 150))
  return plt.imshow(img1)
  ```

<img src="https://docs.google.com/drawings/d/e/2PACX-1vSuZQbSZWEn3U95pgTljtv6vzf_PX8HBcNfrQR0aF5eEZicqj3Kgud6kj5aObVv54uiM2WLiy9s6YMH/pub?w=543&amp;h=256">

Once I viewed the X-Ray iamges I began to prepare the data for modeling. Since I knew I wanted to work with Basic Neural Networks and Convolutional Neural Networks I knew I would have to arrange the data differently. For the CNN I used an Image Data Generator to rescale the images, set the image size to 64 x 64, the batch size to the full dataset, and class mode was set to binary. I then used the next() function to iterate through the images to create the needed datasets. 

Example of code:

```
# Example with train set
train_set = ImageDataGenerator(rescale= 1./255).flow_from_directory(train_dir, target_size = (64, 64), batch_size = 5216, class_mode = "binary")
train_images, train_labels = next(train_set)
```

I then continued to reformat the dataset for my Basic Neural Network. First I indentified the shape of my images and labels.

Example of code:

```
# Identify shapes of image/labels
print('Train Images Shape: ', str(train_images.shape))
print('Train Labels Shape: ', str(train_labels.shape))
print("Number of Training Samples: ", str(train_images.shape[0]))
```
Then I reshaped the images using the .reshape method.

Example of code:

```
# Reshape images for basic NN modeling
train_img = train_images.reshape(train_images.shape[0], -1)
```

To finish scrubbing/exploring I looked at another image, identified the class indices, and renamed my train_labels as train_y to help avoid confusion later on.

### Model:

Within my modeling section I created three Basic Neural Networks and three Convolutional Neural Networks. I will discuss the Basic Neural Networks first.

##### Basic Neural Networks
**Baseline Model**
To begin Modeling I created a baseline Neural Network Model. I intializaed a random seed, created three layers, and compiled the model using tanh as the activation, binary_crossentropy as the loss, and sgd for the optimizer. 

<img src="https://docs.google.com/drawings/d/e/2PACX-1vTYytWhBjqqQJyvguNSLhIRktYMk-CDhRSELtG79SHWorJogkpA8sEIVK4t-lM5w5-y-iclRRTt0vqO/pub?w=606&amp;h=246">

While creating this model I set a callback and created my own functions for running the history, time of model, results, visualizations. (Can view in Google Colab Notebook)

Results:

<img src="https://docs.google.com/drawings/d/e/2PACX-1vSAiME48OJzru1aYDYd3rmnsignJSUzZxdRfuHiQAo3v00xhLAOOU_EM3-0odOm4ynkJFauLbFvgOCG/pub?w=760&amp;h=413">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vSIpGv3ahu7aJzJWbUK8onqSly6PGuuNxUprZvPZtp0TVrxzAUqr3m4d-IsJ8Uq1GH9BaQUy1RpCDcA/pub?w=517&amp;h=551">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vSIpGv3ahu7aJzJWbUK8onqSly6PGuuNxUprZvPZtp0TVrxzAUqr3m4d-IsJ8Uq1GH9BaQUy1RpCDcA/pub?w=517&amp;h=551">

In the following Basic Neural Networks I experimented with changing image size, using different activations, optimizers, changing the batch size, and adding additional Dense layers. The third Neural Network model had the best results with 85% accuracy, high precision/recall, and visualizations that were relatively smooth. 

Third Basic Neural Network Model and Results:
<img src="https://docs.google.com/drawings/d/e/2PACX-1vTNBPQglsFkEoYEyTbMfpb1E0b_h4mA0MkMjAzuKsqTwBQoVTF-pglj-jU_Dpu6dhZ_f6t9kFAkNhT8/pub?w=665&amp;h=296">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vSFpOCeXoK_gf8m3LaA9Uvtzb14a5pLoxbU4hQNGMUvJBmpNVtK4PRN5PF9mg1DrvBmgIT2f0Rcp3W2/pub?w=682&amp;h=559">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vSZj9mIAsFaY39hEOh9K35ut4HOi45JxXaayLGOqt0okfgPW96gSSbp1-XGVrt7CQZzLa_HaIEPi-6O/pub?w=539&amp;h=691">

##### Convolutional Neural Networks
 Convolutional Neural Networks are popular in deep learning where analyzing visual imagery is required. For my baseline CNN Model I used the same parameters as our final Basic Neural Network and simply added the Conv2D, MaxPooling, and Dropout Layers.  

<img src="https://docs.google.com/drawings/d/e/2PACX-1vRH-fnAoCLdoIss9a5pcP-cdJ3Mda6q5fmN31pEGjZSP9hJG4vBh496VqxVxivU5DWZXYFW2Qaq2XHc/pub?w=735&amp;h=422">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vRp4O-MGmN0D_r9wlyLpZtC0xj_Nm5qOTXpbKic_5SL1H3sG8zRzh8cXFKFDY3npwOrSIf5TfAKKUkq/pub?w=796&amp;h=407">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vTpWpK1d5GhzkWM2IPWDTWgAcsCyuf57DiFAG5cKIPH-IvJYADoMUohfEYKvAcZbARyKo6f7edy6RbM/pub?w=718&amp;h=551">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vS2kgOU6lLOZvHXkreMCv_0dR5DJ8lFoZ8qQA2ktDRzb4Na9VxjAGalltKyGNe_rxqq39cI_CSnpT3l/pub?w=675&amp;h=686">

With our first Convolutional Neural Network our accuracy has dropped, our gradient descent steps are more extreme, and our precision has changed. The next steps taken were to tweak the parameters and layers to build a better CNN model. 

### Interpret:




Model Include:
Similar to the Mod 3 project, the focus is on prediction. Good prediction is a matter of the model generalizing well. Steps we can take to assure good generalization include: testing the model on unseen data, cross-validation, and regularization. What sort of model should you build?


Evaluation Include:
Recall that there are many different metrics we might use for evaluating a classification model. Accuracy is intuitive, but can be misleading, especially if you have class imbalances in your target. Perhaps, depending on you're defining things, it is more important to minimize false positives, or false negatives. It might therefore be more appropriate to focus on precision or recall. You might also calculate the AUC-ROC to measure your model's *discrimination*.

## Recommendations




## For Further Information

Please review the narrative of my analysis in my jupyter notebok or review my presentation. For any additional questions please contact via e-mail at Lauren.Esser02@gmail.com or Lauren Esser on LinkedIn.









