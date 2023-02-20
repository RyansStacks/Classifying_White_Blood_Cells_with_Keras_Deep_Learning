# Classifying White Blood Cells with Keras Deep Learning

<img src="https://media.sciencephoto.com/image/c0215391/800wm/C0215391-Monocyte_white_blood_cell,_SEM.jpg" width=400 height=300>

## A deep learning classifier is created that classifiers lymphocytes (white blood cells) as cancerous or benign. 
#### Cancer is one of the leading causes of death in America and diagnosis cells through microscopy is limited by human error.
#### Normal white blood cells can be distinguised visually from cancerous cells


## The model using a Convolutional Neural Networks (CNN) using Keras - one of the most popular deep learning platforms being used today.
#### Convolutional Neural Networks work similarly to Artificial Neural Networks were weights are assigned to particular features in a data set
#### Features in a particular data set in this particular project where limited to what value was assigned to the red, green, and blue image values
#### Each red, green, and blue component of an image contains a value from 0-255 where for a specific pixel within an image - a red pixel would have a red value of 255 but zero for both green and blue
#### The Convolutional Neural Network works applying weights to each color value at a particular pixel where cancerous lymphocytes exhibit a specific colorization as compared to benign versions
#### Furthermore, the Convolutional Neural Networks works in convolutons, producing reduced matrices from prior matrices by scanning the data set in for values that "stand out"
#### In this particular project, a 2 x 2 filter was used to scan the a matrix containing 2 rows and 2 columns at a time and the maximum value within each one of these squares is then projected to a new matrix space continuously updating the most prominent values in the data set and reducing the data size down (great for high memory images!).
## Image Preprocessing is Tantamount to Performing CNN
#### Images must be properly constructed by created by indvidual matrices for each red, blue, and green values that are shaped according to pixel dimensions.
#### For example, one picture would have three matrices (red, blue, green) that would be 25 x 25 if images were 625 pixes.
#### To further complicate the process, images that are input into Keras neural networks must be flatten where the original input shape is noted in a parameter
## Final Results were Achieve with Great Success
#### The CNN classifier was able to predict malignant vs non-malignant with 99% accuracy
#### Note of importance is that a small sample size of 200 images were used so the model may definitely be over-generalizing (performing well during training but may not perform as well in the real-world).
#### This model was strictly for demonstraton purposes and in the real world a image set of thousands or millions of images would be used (a feat only accomplished by large enterprises)




