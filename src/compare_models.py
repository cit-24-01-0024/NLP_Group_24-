import pandas as pd

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Bi-LSTM"
    ],
    "Accuracy": [
        77.37,
        78.19
    ],
    "Precision": [
        75.09,
        76.81
    ],
    "Recall": [
        77.37,
        78.19
    ],
    "F1 Score": [
        71.95,
        77.43
    ]
})

print(comparison)

comparison.to_csv(
    "results/comparison.csv",
    index=False
)

print("\nComparison table saved successfully!")