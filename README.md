# Logistic Regression and SVM Classifier on Iris and Breast Cancer Datasets

## GPT histry
[SVM](https://chatgpt.com/share/6725c4ce-22ac-800c-8ed4-cb34c657510b)
[Deploy](https://chatgpt.com/share/6725c4ea-4d14-800c-878b-75ee2c37c1d2)

## Overview
This Streamlit application allows users to interactively train and visualize Logistic Regression and Support Vector Machine (SVM) classifiers on the Iris and Breast Cancer datasets. The app provides functionalities for parameter tuning, model evaluation, and visual representation of decision boundaries.

## Features
- **Dataset Selection**: Choose between the Iris and Breast Cancer datasets for model training and evaluation.
- **Model Selection**: Select either Logistic Regression or SVM for classification.
- **Parameter Tuning**:
  - **Logistic Regression**: Customize `C` (regularization strength) and `max_iter` (maximum iterations).
  - **SVM**: Adjust `C` and choose the kernel type (`linear`, `rbf`, `poly`, or `sigmoid`).
- **Model Training and Evaluation**:
  - Display accuracy and a comprehensive classification report.
  - Visualize a confusion matrix for the test set.
- **Decision Boundary Visualization**: View a graphical representation of the decision boundary plotted against the feature space for both the training and test sets.

## Getting Started

### Prerequisites
Ensure you have Python 3.7 or later installed along with `pip`. You will need the following Python libraries:
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

Install the required packages using:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

### Installation
1. Clone this repository or download the `app.py` file:
   ```bash
   git clone https://github.com/IdONTKnowCHEK/HW3-SVM.git
   cd HW3-SVM
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL provided by Streamlit, typically `http://localhost:8501`.

## How to Use the App
1. **Select Dataset**: Use the sidebar to choose either the Iris or Breast Cancer dataset.
2. **Select Model**: Choose between Logistic Regression and SVM.
3. **Set Parameters**:
   - Adjust `C` and `max_iter` for Logistic Regression.
   - Set `C` and choose the kernel type for SVM.
4. **View Results**:
   - The app will display the accuracy score, classification report, and a confusion matrix plot.
5. **Visualize Decision Boundary**:
   - A plot will show the decision boundary of the selected model along with the training and test data points.

## Example Screenshots
![image](https://github.com/user-attachments/assets/8aa86615-5623-4653-8c69-76933010ca5b)
![image](https://github.com/user-attachments/assets/de71a939-91c1-472d-948e-236db9423226)


## Customization
Feel free to modify the `app.py` file to add more features, such as:
- Additional dataset options.
- Other model types like Decision Trees or Random Forest.
- Hyperparameter optimization options.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or suggestions, please reach out to [your-email@example.com] or open an issue on GitHub.
