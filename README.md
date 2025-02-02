# TOPSIS for Comparing Pre-trained Models for Text Classification

This project applies the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method to evaluate and rank five pre-trained models for text classification. The models compared are:

- **BERT**
- **RoBERTa**
- **XLNet**
- **ALBERT**
- **DistilBERT**

## Evaluation Criteria

We consider the following criteria:
- **Accuracy** (Benefit criterion)
- **F1 Score** (Benefit criterion)
- **Inference Time (ms)** (Cost criterion)
- **Model Size (M parameters)** (Cost criterion)

## Data

The performance metrics used in this demo are hypothetical and provided in the table below:

| Model       | Accuracy | F1 Score | Inference Time (ms) | Model Size (M) |
|-------------|----------|----------|---------------------|----------------|
| BERT        | 0.90     | 0.89     | 50                  | 110            |
| RoBERTa     | 0.91     | 0.90     | 55                  | 125            |
| XLNet       | 0.88     | 0.87     | 60                  | 110            |
| ALBERT      | 0.89     | 0.88     | 40                  | 12             |
| DistilBERT  | 0.87     | 0.86     | 30                  | 66             |

## Weights

The weights assigned to each criterion are:

- Accuracy: **0.4**
- F1 Score: **0.3**
- Inference Time: **0.2**
- Model Size: **0.1**

These weights reflect the relative importance of each criterion.

## TOPSIS Methodology

The TOPSIS method involves the following steps:
1. **Normalization:** Normalize the decision matrix.
2. **Weighting:** Multiply the normalized matrix by the criterion weights.
3. **Ideal Solutions:** Determine the ideal (best) and negative-ideal (worst) solutions.
4. **Distance Calculation:** Compute the Euclidean distance of each alternative from the ideal and negative-ideal solutions.
5. **Score Calculation:** Calculate the TOPSIS score (relative closeness to the ideal solution).
6. **Ranking:** Rank the models based on the TOPSIS score (higher scores indicate better performance).

## Running the Code

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
2. Run the script:
   ```bash
   python topsis.py
3. The results and a ranking bar chart (saved in the figures/ folder) will be generated.
