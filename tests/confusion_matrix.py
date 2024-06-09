import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def read_json(file_path):
    return pd.read_json(file_path)

def get_polarity(row):
    return row["part2_aggregate"]["polarity"] if row["part2_aggregate"]["polarity"] != "undefined" else row["discussion_polarity"]

def get_llm_polarity(row):
    return row["tools"]["mistral:7b:instruct:int4"]

def main():
    dataset_path = "./data/dataset-mistral-prompt2.json"
    dataset = read_json(dataset_path)
    dataset["polarity"] = dataset.apply(get_polarity, axis=1)
    dataset["llm_polarity"] = dataset.apply(get_llm_polarity, axis=1)
    y_true = dataset["polarity"]
    y_pred = dataset["llm_polarity"] 
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

main()