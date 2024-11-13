
# Face Mask Classification Project

This project involves training and evaluating four convolutional neural network (CNN) models to classify images of faces with and without masks. It explores different model architectures, including both custom and pre-trained models, to assess their performance in a mask detection context, relevant during the COVID-19 pandemic.

## Project Structure

- **rapport.pdf**: Project report, summarizing objectives, methods, model architectures, and results.
- **model1.py**: Script for a simple CNN model with basic convolutional layers.
- **model2.py**: Script for a modular CNN model using Keras' functional API.
- **model3.py**: Script for a VGG16-based model using transfer learning.
- **model4.py**: Script for a MobileNetV2-based model using transfer learning.
- **evaluation.py**: Script for evaluating the models on test data.

## Objectives

The main objective is to develop models that can differentiate between masked and unmasked faces, aiming for high accuracy, computational efficiency, and generalizability across variations in image data.

## Model Architectures

1. **Model 1**: Simple CNN with three convolutional layers and max-pooling.
   - *Strengths*: Simplicity, fast execution.
   - *Limitations*: Lower accuracy, struggles with partially obscured masks.

2. **Model 2**: Modular CNN using Keras functional API.
   - *Strengths*: Improved accuracy, better at detecting partially obscured masks.
   - *Limitations*: More complex, slower than Model 1.

3. **Model 3**: VGG16 model with transfer learning.
   - *Strengths*: High accuracy, effective at detecting masks.
   - *Limitations*: High computational resource requirements.

4. **Model 4**: MobileNetV2 model with transfer learning.
   - *Strengths*: Balanced accuracy and efficiency, suitable for resource-constrained environments.
   - *Limitations*: Slightly lower accuracy than Model 3 on masked faces.

## Data Preprocessing

- Data augmentation is applied using Keras' `ImageDataGenerator`, which includes rescaling, horizontal flipping, zoom, width and height shifts, and rotation to improve model generalization.

## Dataset
 
The dataset is available here on Google Drive(https://drive.google.com/file/d/1KAn2og3IbVNoBM06Ck8yNxUXHvUULXsQ/view). Download and extract it into the appropriate folder to run the project.
It contains labeled images of faces with and without masks, organized into training and validation sets. The dataset has been preprocessed with data augmentation to enhance model robustness and generalization.

## Training and Evaluation

- Each model is trained for up to 100 epochs using augmented training data, with checkpoints and early stopping to save the best-performing model.
- Evaluation is performed on validation data to assess each model's accuracy in classifying faces with and without masks.

## Results Summary

| Model        | Overall Accuracy | Face Class Accuracy | Maskface Class Accuracy |
|--------------|------------------|---------------------|--------------------------|
| **Model 1**  | 59.82%           | 21.94%             | 97.69%                   |
| **Model 2**  | 89.78%           | 94.25%             | 85.30%                   |
| **Model 3**  | 97.73%           | 97.47%             | 97.98%                   |
| **Model 4**  | 92.41%           | 94.05%             | 90.77%                   |

## Key Findings

- Models 3 and 4 (transfer learning) outperform Models 1 and 2.
- VGG16 (Model 3) achieves the highest accuracy but is resource-intensive.
- MobileNetV2 (Model 4) balances performance with efficiency, suitable for deployment on devices with limited computational power.

## How to Run

1. **Install Dependencies**:
   Ensure `tensorflow`, `keras`, `matplotlib`, and other dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Training the Models**:
   Run each model's script to train. Example for Model 1:
   ```bash
   python model1.py
   ```

3. **Evaluating Models**:
   Use `evaluation.py` to evaluate a trained model, e.g.:
   ```bash
   python evaluation.py
   ```

## Conclusion

- Model 3 (VGG16) is recommended for applications requiring high accuracy.
- Model 4 (MobileNetV2) is an efficient alternative for resource-constrained applications.

## Author

- Hentati Amin (2nd-year Data Engineering, 2023-2024)
