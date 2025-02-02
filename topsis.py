#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the decision matrix with rows as models and columns as criteria.
data = {
    'Model': ['BERT', 'RoBERTa', 'XLNet', 'ALBERT', 'DistilBERT'],
    'Accuracy': [0.90, 0.91, 0.88, 0.89, 0.87],
    'F1_Score': [0.89, 0.90, 0.87, 0.88, 0.86],
    'Inference_Time': [50, 55, 60, 40, 30],   # in ms (cost criterion)
    'Model_Size': [110, 125, 110, 12, 66]     # in million parameters (cost criterion)
}

# Create DataFrame and set 'Model' as the index
df = pd.DataFrame(data)
df.set_index('Model', inplace=True)

# Define weights for each criterion (ensure they sum to 1)
weights = np.array([0.4, 0.3, 0.2, 0.1])

# Specify criteria types: True for benefit criteria, False for cost criteria.
criteria_type = [True, True, False, False]

# Step 1: Normalize the decision matrix using vector normalization.
def normalize_decision_matrix(matrix):
    norm_matrix = matrix.copy()
    for column in matrix.columns:
        norm = np.sqrt((matrix[column] ** 2).sum())
        norm_matrix[column] = matrix[column] / norm
    return norm_matrix

normalized_df = normalize_decision_matrix(df)

# Step 2: Multiply the normalized decision matrix by the weights.
weighted_normalized_df = normalized_df * weights

# Step 3: Determine the ideal best and worst values for each criterion.
ideal_best = {}
ideal_worst = {}
for idx, column in enumerate(weighted_normalized_df.columns):
    if criteria_type[idx]:  # Benefit criterion: higher is better
        ideal_best[column] = weighted_normalized_df[column].max()
        ideal_worst[column] = weighted_normalized_df[column].min()
    else:  # Cost criterion: lower is better
        ideal_best[column] = weighted_normalized_df[column].min()
        ideal_worst[column] = weighted_normalized_df[column].max()

# Step 4: Calculate the Euclidean distance from the ideal best and worst for each alternative.
def euclidean_distance(row, ideal):
    return np.sqrt(sum((row - pd.Series(ideal))**2))

distances_positive = weighted_normalized_df.apply(lambda row: euclidean_distance(row, ideal_best), axis=1)
distances_negative = weighted_normalized_df.apply(lambda row: euclidean_distance(row, ideal_worst), axis=1)

# Step 5: Calculate the TOPSIS score (relative closeness to the ideal solution).
topsis_score = distances_negative / (distances_positive + distances_negative)
df['TOPSIS_Score'] = topsis_score

# Step 6: Rank the models based on the TOPSIS score (higher score is better).
df['Rank'] = df['TOPSIS_Score'].rank(ascending=False).astype(int)

# Print the TOPSIS ranking results.
print("TOPSIS Ranking Results:")
print(df[['TOPSIS_Score', 'Rank']].sort_values(by='TOPSIS_Score', ascending=False))

# Step 7: Plot the TOPSIS scores for visualization.
plt.figure(figsize=(10, 6))
bars = plt.bar(df.index, df['TOPSIS_Score'], color='skyblue')
plt.xlabel('Models')
plt.ylabel('TOPSIS Score')
plt.title('TOPSIS Ranking of Pre-trained Models for Text Classification')

# Annotate bar values.
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.3f}', 
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), 
                 textcoords="offset points", 
                 ha='center', va='bottom')

# Create figures directory if it doesn't exist and save the plot.
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/ranking_bar_chart.png', bbox_inches='tight')
plt.show()
