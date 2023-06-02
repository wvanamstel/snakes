### Identify snake species from images
This repo contains code to classify the species of snakes given an images of varied sources. This part of the [Snake species identification challenge](https://www.aicrowd.com/challenges/snakeclef2021-snake-species-identification-challenge) on the [AICrowd](https://www.aicrowd.com/) platform.

The goal of the challenge is to provide a classification algorithm to better determine which antivenom to administer to a victim of a snakebite, given a photo of the snake.

The input images are very diverse (urban/nature background) and a snake can be located all over image. Here are some examples of input images:
<img
src="./imgs/snakes_examples.jpg"
alt="Example of raw input images">


#### Modelling approach
* Create a segmentation model that will create a bounding box around the snake in a given image.
* Create a classification model with input the segmented snake image.

##### Segmentaion model
To create the segmentation model there was the need to create a segmented training set first. Using [Label Studio](https://github.com/heartexlabs/label-studio) the snakes in 900 images were annotated.

<img 
src="./imgs/Segmentation_train_loss.svg"
alt="Segmentation training loss"
width="500">
