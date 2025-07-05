Employee Promotion Prediction

This project aims to predict whether an employee will be promoted based on various features such as department, region, education, previous ratings, and more. The dataset is provided as part of a machine learning final assessment.

🧾 Dataset Files

train.csv - Training dataset with features and target (is_promoted).

test_2umaH9m.csv - Test dataset without target column.

sample_submission_M0L0uXE.csv - Sample format for submission file.


📊 Project Workflow

1. Exploratory Data Analysis (EDA)

Displayed dataset info using .head(), .tail(), .info(), .describe().

Identified numeric and categorical features.

Visualized class distribution (is_promoted) and selected categorical variables using seaborn.


2. Handling Missing Values

Imputed missing values in education and previous_year_rating columns using most frequent strategy.


3. Label Encoding

Converted categorical columns (department, region, gender, education, recruitment_channel) into numeric labels using LabelEncoder.


4. Feature Scaling

Standardized selected numerical features using StandardScaler to normalize data:

no_of_trainings, age, previous_year_rating, length_of_service, avg_training_score.



5. Model Training and Evaluation

Split dataset into training and testing sets (80%–20%) using train_test_split.

Trained and evaluated the following models:

✅ Logistic Regression

✅ Random Forest Classifier

✅ k-Nearest Neighbors (k-NN)

✅ Gradient Boosting Classifier

✅ Multi-layer Perceptron (MLP) Classifier


Each model was evaluated using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix


A comparison dataframe was created to rank models based on F1-Score.

6. Hyperparameter Tuning (Optional)

Set up a GridSearchCV configuration (defined but not run) to tune GradientBoostingClassifier parameters:

n_estimators, learning_rate, max_depth, subsample


7. Prediction on Test Data

Applied the same preprocessing steps on the test set.

Used the best-performing model (Gradient Boosting) to generate predictions.

Prepared final predictions in the sample_submission format.

Saved submission file as final_submission.csv.



---

🔧 Technologies Used

Python 3

pandas, numpy

seaborn, matplotlib

scikit-learn



---

🏁 How to Run

1. Clone the repository or open in Google Colab.


2. Upload the dataset files:

train_LZdllcl.csv

test_2umaH9m.csv

sample_submission_M0L0uXE.csv



3. Run the cells in final_assessment.ipynb.


4. The model will generate a final_submission.csv file with predictions.




---

📌 Notes

Remember to match preprocessing steps exactly between train and test datasets.

You can explore additional techniques like feature importance, SMOTE (for imbalance), or ensemble voting for better accuracy.



---

📈 Result Summary
Model	Accuracy	Precision	Recall	F1 Score

Logistic Regression	...	...	...	...
Random Forest	...	...	...	...
k-NN	...	...	...	...
MLP Classifier	...	...	...	...
Gradient Boosting	...	...	...	✅ Best




---

📬 Contact

For any queries or contributions, feel free to reach out!
