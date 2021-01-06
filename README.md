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

<img src="https://docs.google.com/drawings/d/e/2PACX-1vRpT2mCacPWWNKdJtpXD1xaoSmYwEqmOgs8eWCDN3uAQeqwCz3sy52eJ9LMBY9Lb7gJNC26ldmjswTk/pub?w=456&amp;h=690">

In the following Basic Neural Networks I experimented with changing image size, using different activations, optimizers, changing the batch size, and adding additional Dense layers. The third Neural Network model had the best results with 85% accuracy, high precision/recall, and visualizations that were relatively smooth. 

Third Basic Neural Network Model and Results:
<img src="https://docs.google.com/drawings/d/e/2PACX-1vTNBPQglsFkEoYEyTbMfpb1E0b_h4mA0MkMjAzuKsqTwBQoVTF-pglj-jU_Dpu6dhZ_f6t9kFAkNhT8/pub?w=665&amp;h=296">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vTBh2_RXXirLHghkQmdtWDnsoScTYzZuzunWT4k5WhndL847Xj-eK4gjEL7jM-07MoRMJuCvbo6fEDZ/pub?w=746&amp;h=415">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vSFpOCeXoK_gf8m3LaA9Uvtzb14a5pLoxbU4hQNGMUvJBmpNVtK4PRN5PF9mg1DrvBmgIT2f0Rcp3W2/pub?w=682&amp;h=559">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vSZj9mIAsFaY39hEOh9K35ut4HOi45JxXaayLGOqt0okfgPW96gSSbp1-XGVrt7CQZzLa_HaIEPi-6O/pub?w=539&amp;h=691">

##### Convolutional Neural Networks
 Convolutional Neural Networks are popular in deep learning where analyzing visual imagery is required. For my baseline CNN Model I used the same parameters as our final Basic Neural Network and simply added the Conv2D, MaxPooling, and Dropout Layers.  

<img src="https://docs.google.com/drawings/d/e/2PACX-1vRH-fnAoCLdoIss9a5pcP-cdJ3Mda6q5fmN31pEGjZSP9hJG4vBh496VqxVxivU5DWZXYFW2Qaq2XHc/pub?w=735&amp;h=422">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vRp4O-MGmN0D_r9wlyLpZtC0xj_Nm5qOTXpbKic_5SL1H3sG8zRzh8cXFKFDY3npwOrSIf5TfAKKUkq/pub?w=796&amp;h=407">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vTpWpK1d5GhzkWM2IPWDTWgAcsCyuf57DiFAG5cKIPH-IvJYADoMUohfEYKvAcZbARyKo6f7edy6RbM/pub?w=718&amp;h=551">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vS2kgOU6lLOZvHXkreMCv_0dR5DJ8lFoZ8qQA2ktDRzb4Na9VxjAGalltKyGNe_rxqq39cI_CSnpT3l/pub?w=675&amp;h=686">

With our first Convolutional Neural Network our accuracy has dropped, our gradient descent steps are more extreme, and our precision has changed. The next steps taken were to tweak the parameters and layers to build a better CNN model. The final CNN Model I will show in the interpret section because this is the choice model that I would select.

### Interpret:

##### Final CNN Model
<img src="https://docs.google.com/drawings/d/e/2PACX-1vQNRRpZG7376IWL6AFUIMw0KrzOxkvXYAYGBTsspA6ih6dviDCFd226C5bdtgTGZshmW-En-ocTzAlj/pub?w=796&amp;h=598">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vSg7wiHtIGHSiUEQu0EexPwd33LC53TJ8ZtK2nIUQQGAx3Z0Ij_LfK4S2xqJtPos3IzCeEf_ZXo0WlN/pub?w=867&amp;h=407">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vRsxlVJhL3-_se_a1J4_1BWkSFhQ9sZQaLN9C3LiLTusQ0nC-t0BX9VD2rjcBUvnegD1xVcrg3CvsRY/pub?w=794&amp;h=558">

<img src="https://docs.google.com/drawings/d/e/2PACX-1vR0hCP2hQtP4PvZyv3Y1Pd8ch6heKo_BMTAdtVa0xv19P2CqJAxwLEgV5S86fBfZmlQBa5UHu_0AaWl/pub?w=763&amp;h=682">

This model hits our goal of predicting with 88% accuracy. Our precision and recall scores are also above 80%. 


## Recommendations
1. Use an image size of 64x64 for faster processing.
2. If focusing on highest Accuracy use CNN Model 3.
3. If focusing on highest Recall use CNN Model 2. 
4. If focusing on highest Precision use the Baseline Neural Network Model.
5. Overall the best Model to use would be CNN Model 3.

## Future Work
1. Use a larger dataset to improve accuracy and precision.
2. Look at X-Rays of adult lungs with and without Pneumonia.
3. Use a premade CNN to see if they could potentially work better.

## Reproduction Instructions
This project uses:
* Anaconda, a package and environmental management tool
* Python 3.6.9
* Numpy
* Pandas
* Seaborn
* Matplotlib
* Sklearn
* Tensorflow
* Datetime
* Scipy
* Os, Glob
* Yellowbrick

If you would like to follow the analysis locally and have the above tools:

1. Fork and clone this repository
2. Obtain data [Linked Here.](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and upload into your Google Drive under /gdrive/My Drive
3. Download and run notebook into Google Colab making sure you connect to your own Google Drive.

You should then be able to run the exploration and analysis in the provided X-Ray-Prediction-Notebook.ipynb.

## For Further Information

Please review the narrative of my analysis in my jupyter notebook or review my presentation. For any additional questions please contact via e-mail at Lauren.Esser02@gmail.com or on [LinkedIn.](https://www.linkedin.com/in/laurenesser/)

## Repository Structure:
README.md <- README for reviewers of this project.
X-Ray-Prediction-Notebook.ipynb <- narrative documentation of analysis in jupyter notebook
Presentation.pdf <- pdf version of project presentation







