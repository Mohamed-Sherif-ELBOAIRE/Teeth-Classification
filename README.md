# Teeth Classification - Preprocessing, Visualization, and Model Training

## Project Overview

This project aims to develop a comprehensive teeth classification solution, including preprocessing, visualizing dental images, and building a robust computer vision model capable of accurately classifying teeth into seven distinct categories. This solution is crucial for our company's AI-driven dental solutions, as accurate teeth classification aligns with our strategic goals in the healthcare industry, enhancing diagnostic precision and improving patient outcomes.

## Project Structure

The repository is organized as follows:

- **CaS/**: Contains data and results related to the CaS class.
- **CoS/**: Contains data and results related to the CoS class.
- **Gum/**: Contains data and results related to the Gum class.
- **MC/**: Contains data and results related to the MC class.
- **OC/**: Contains data and results related to the OC class.
- **OLP/**: Contains data and results related to the OLP class.
- **OT/**: Contains data and results related to the OT class.

## Key Components

### 1. Preprocessing

The preprocessing step includes preparing dental images for analysis through normalization and augmentation. This ensures the images are in optimal condition for model training and evaluation.

- **Augmentation:** We applied various augmentation techniques, including rotation, width and height shift, shear, zoom, and flipping, to enhance the dataset's diversity.

### 2. Visualization

Visualization of the dataset helps understand the distribution of classes and evaluate the effectiveness of preprocessing techniques.

- **Class Distribution:** The class distribution in the training dataset is visualized to ensure a balanced dataset.
- **Augmentation Visualization:** Images before and after augmentation are displayed to evaluate the effectiveness of preprocessing techniques.

### 3. Model Architecture and Training

We used TensorFlow/Keras to build a model tailored for the classification of dental images. The model architecture includes several convolutional layers followed by fully connected layers. 

- **Training:** The model was trained on the preprocessed dataset, and early stopping was used to prevent overfitting.
- **Hyperparameter Tuning:** Hyperparameters like filter size and number of nodes were tuned to achieve optimal model performance.
- **Evaluation:** The model was evaluated on a test dataset, and the best accuracy achieved was 91.75%.

## Results

### Confusion Matrix

The confusion matrix shows the classification performance across different classes.

### Training and Validation Accuracy/Loss

Training and validation accuracy and loss curves provide insights into the model's performance during training.

### Test Loss and Accuracy

Test loss and accuracy curves are plotted to evaluate the model's generalization capability.

## TensorBoard Visualization

TensorBoard was used to log the training process and visualize the performance of different hyperparameter settings.

## Model Saving and Evaluation

The trained model was saved for future use in inference, evaluation, or further training. The model achieved a test accuracy of 98.44% on the test dataset.

## How to Run the Code

1. Clone the repository.
2. Install the required libraries: TensorFlow, Keras, Matplotlib, Seaborn, etc.
3. Place the dataset in the appropriate folders.
4. Run the preprocessing script to prepare the data.
5. Train the model by executing the training script.
6. Use TensorBoard to visualize the training process.

## Conclusion

This project successfully developed a model for accurate teeth classification. The results demonstrated that the model can effectively distinguish between different classes with high accuracy, making it a valuable tool for AI-driven dental solutions.
