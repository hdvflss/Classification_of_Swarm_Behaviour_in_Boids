# Classification of Swarm Behaviour in Boids

## Overview
This project aims to classify whether a group of boids (computer-generated agents mimicking collective animal movement) exhibits swarm behavior. Using Machine Learning (ML) techniques, we analyze patterns in boid movement and apply classification models to determine the presence of flocking behavior.

### Key Objectives:
This project explores how swarm behavior is perceived by humans and applies Supervised Learning to classify boid groups as either "flocking" (1) or "not flocking" (0). We compare the performance of Logistic Regression (LoR) and Random Forest (RF) models.

## Dataset
The dataset used in this project comes from Kaggle, originally sourced from the University of New South Wales (UNSW):  
[Swarm Behaviour Classification Dataset](https://www.kaggle.com/datasets/deepcontractor/swarm-behaviour-classification/data)  
Since the dataset is too large for GitHub, please accquire it directly from Kaggle.

### Dataset Details:
The dataset, `Swarm_Behaviour.csv`, consists of 23,309 rows, each representing a group of boids with 2,400 attributes. These attributes include position (`x, y` coordinates), velocity (`xVel, yVel`), and alignment, cohesion, and separation vectors. The target variable, `Swarm_Behaviour`, is labeled as 1 for showing swarm behavior and 0 for not showing swarm behavior.

## Project Files
- Project Report: `Project Report.pdf` - Detailed explanation of methodology and findings.
- Jupyter Notebook: `Classification_Boids.ipynb` - The main code implementation.
- Dataset: `Swarm_Behaviour.csv` - The dataset used for training and evaluation.

## Installation & Dependencies
To run this project, you need Python and the required libraries installed.

### Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Required Libraries:
- `numpy` (Numerical computations)
- `pandas` (Data manipulation)
- `matplotlib` & `seaborn` (Data visualization)
- `scikit-learn` (Machine Learning models and evaluation)

## Running the Notebook
1. Clone the repository or download the files.
2. Ensure `Swarm_Behaviour.csv` is in the same directory as `Classification_Boids.ipynb`.
3. Open Jupyter Notebook:
   ```bash
   jupyter notebook Classification_Boids.ipynb
   ```
4. Follow the steps in the notebook to:
   - Load the dataset.
   - Preprocess the data (handle missing values, class balancing, feature scaling).
   - Train and evaluate Logistic Regression and Random Forest models.
   - Compare model performance and view results.

## Results & Findings
Logistic Regression (LoR) was selected as the final model due to its balanced performance, while Random Forest (RF) initially overfitted but improved after hyperparameter tuning. The final model accuracy results are:

- **LoR:** Training 91.9%, Validation 90.6%, Test 90.7%
- **RF:** Training 97.3%, Validation 89.6%, Test 89.7%.

## Future Improvements
Future enhancements could include feature selection to reduce the 2,400 attributes, exploring alternative models such as XGBoost or LightGBM, adjusting decision thresholds for better precision, and experimenting with different feature engineering techniques.

## Contributors
- **Khanh Pham**  
  tpc.khanhpham83@gmail.com  
- **Huy Vu**  
  hdvflss@gmail.com

## References
- University of New South Wales Swarm Survey
- UCI Machine Learning Repository
- Kaggle Dataset by Deep Contractor
- Research papers and ML references cited in `Project Report.pdf`

## License
This project is for academic and research purposes. The dataset is publicly available under Kaggle’s data-sharing policies.

For further details, refer to `Project Report.pdf`. If you have any questions, feel free to reach out.

