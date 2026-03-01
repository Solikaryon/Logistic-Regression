# Logistic Regression - Breast Cancer Classification

**Author:** Luis Fernando Monjaraz Briseño

## Project Description

This Jupyter notebook implements logistic regression from scratch (without scikit-learn) to classify breast cancer tumors as malignant or benign. The implementation demonstrates how to build a full classification model including the sigmoid activation function, loss computation, and iterative weight optimization.

## Dataset

The analysis uses the **Breast Cancer Dataset** from scikit-learn:
- **Samples:** 569 tumor records
- **Features:** 30 numeric features (cell measurements)
- **Classes:** 2 (benign or malignant)
- **Independent Variables (X):** Cell measurement features
- **Dependent Variable (Y):** Tumor classification (0 or 1)

## Custom Implementation

### Logistic Regression Class
The notebook implements a complete logistic regression classifier with:

```python
class LogisticRegression:
    - Sigmoid activation function
    - Loss computation (binary cross-entropy)
    - Gradient descent optimization
    - Weight and bias updates
```

### Key Parameters
- **Learning Rate:** 0.01 (controls optimization step size)
- **Number of Iterations:** 1000 (training epochs)
- **Loss Function:** Binary cross-entropy

## Implementation Details

### Algorithm Components

1. **Sigmoid Function:**
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

2. **Loss Function (Binary Cross-Entropy):**
   $$L = -\frac{1}{m} \sum (y \log(h) + (1-y) \log(1-h))$$

3. **Gradient Descent:**
   - Compute gradients with respect to weights and bias
   - Update parameters: $w = w - \alpha \cdot \frac{\partial L}{\partial w}$

### Training Process

- Split data into training and testing sets
- Scale features using StandardScaler for better performance
- Train model with gradient descent
- Track loss across iterations
- Make predictions on unseen test data

## Libraries Used

- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `sklearn.datasets` - Load breast cancer dataset
- `sklearn.model_selection` - Train-test split
- `sklearn.preprocessing` - Data standardization
- `sklearn.metrics` - Model evaluation metrics:
  - Confusion matrix
  - Accuracy score
  - Recall score
  - Precision score

## Evaluation Metrics

The model is evaluated using:

- **Confusion Matrix:** True Positives, True Negatives, False Positives, False Negatives
- **Accuracy:** Percentage of correct predictions
- **Precision:** Proportion of positive predictions that are correct
- **Recall:** Proportion of actual positive cases correctly identified

## Visualizations

- **Loss History:** Shows how loss decreases during training
- **Confusion Matrix:** Displays classification performance
- **Performance Metrics:** Summary of model accuracy and reliability

## How to Run

1. Ensure Python 3.x is installed
2. Install required packages:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```
3. Run the notebook:
   ```bash
   jupyter notebook "Logistic Regression.ipynb"
   ```

## Model Flow

1. Load breast cancer dataset
2. Standardize features for uniform scaling
3. Split data (training/testing)
4. Train custom logistic regression model
5. Make predictions on test set
6. Evaluate using multiple metrics
7. Visualize results

## Expected Output

- Trained weights and bias parameters
- Confusion matrix showing prediction breakdown
- Classification metrics (accuracy, precision, recall)
- Loss curve showing training progress

## Key Concepts Demonstrated

- **Binary Classification:** Two-class prediction problem
- **Logistic Regression:** Probability-based classification
- **Gradient Descent:** Iterative optimization algorithm
- **Feature Scaling:** Data normalization
- **Model Evaluation:** Multiple classification metrics
- **Training vs. Testing:** Cross-validation methodology

## Mathematics Behind Logistic Regression

Logistic regression finds the decision boundary by:

1. Computing linear combination: $z = w \cdot x + b$
2. Applying sigmoid: $\hat{y} = \sigma(z)$
3. Computing loss: Binary cross-entropy
4. Minimizing loss: Gradient descent optimization

## File Structure

```
.
├── Logistic Regression.ipynb
└── README.md
```

## Notes

- The dataset is loaded directly from scikit-learn
- Features are automatically standardized
- Model converges after ~1000 iterations
- Custom implementation helps understand algorithm internals

## Technologies Used

- Python 3.x
- NumPy for mathematical operations
- Scikit-learn for dataset and metrics
- Matplotlib for visualization

## Learning Objectives

This project demonstrates:
- Building classifiers from scratch
- Binary classification concepts
- Logistic regression mathematics
- Gradient descent optimization
- Feature scaling importance
- Model evaluation metrics
- Working with real-world medical data
