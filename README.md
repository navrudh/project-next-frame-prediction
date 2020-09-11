
# Self-Supervised Model for Video Prediction and Unsupervised Action Recognition

## Task Scripts:

 - ```train_video_prediction.py``` - Trains a model for the task of video prediction on the UCF101
 - ```build_classification_dataset.py``` - Runs videos through the trained video prediction model and stores the intermediate tensors to be used for classifying actions
 - ```train_video_classification.py``` - Use the saved intermediate tensors and classify actions.
 - ```generate_gif.py``` - Iterates through a few videos of the dataset and generates 6 image frames :- 3 seed frames and 3 predicted frames.

## Configuration:

  - ```resources/config/user-{linux_or_windows_user}.json``` - create a similar configuration file specifying the UCF101 dataset location, number of workers for processing, classification dataset directory, etc

## Sample Output
![YOYO](resources/yoyo-gif-output.gif)