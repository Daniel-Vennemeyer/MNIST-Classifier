
# Deep Learning Model for MNIST Classification

This project implements various deep learning models (DNN, ConvNet, VGG, ResNet) to classify the MNIST handwritten digit dataset using PyTorch. 

## Requirements

Make sure you have the following packages installed:

- Python (3.6 or higher)
- PyTorch
- torchvision
- scikit-learn
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install torch torchvision scikit-learn numpy matplotlib
```

## Dataset

The project uses the MNIST dataset, which is automatically downloaded when running the code.

## Running the Model

1. **Modify the Main Function**

   You can load and test a saved model by modifying the parameters in the `test_model` function. The Highest weights are already saved and ready to be run, just modify the file path:

   ```python
   test_model('path_to_your_model.pth', 'ModelName', test_loader)
   ```

   Alternatively, to train the model, in the `if __name__ == '__main__':` block of the code, choose a model to train by uncommenting the relevant lines. For example:

   ```python
   model_names = ['DNN', 'ConvNet', 'VGG', 'ResNet']
   for model_name in model_names:
       print(f"\nTesting model: {model_name}")
       run_experiment(model_name)
   ```

2. **Run the Script**

   Execute the script:

   ```bash
   python your_script.py
   ```

   Replace `your_script.py` with the name of your Python file containing the code.

3. **View Results**

   The script will print the results of the test set, including the average loss, accuracy, F1 score, and AUC score. It will also plot the training and validation metrics, as well as the ROC and Precision-Recall curves.

## Saving Model Weights

The model weights will be saved after each macro epoch in the format `HW2_weights_ModelName.pth`. You can load these weights later using the `test_model` function.

