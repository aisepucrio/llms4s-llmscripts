import pandas as pd
import os
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from run_llm import models

def get_files_names():
    folder_path = 'data'
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
    files_names = [file.split('.')[0] for file in json_files]
    return files_names

def get_polarity(message):
    return message["part2_aggregate"]["polarity"] if message["part2_aggregate"]["polarity"] != "undefined" else message["discussion_polarity"]

def get_predicted_polarity(row, model_name):
    return row['tools'].get(model_name, 'undefined')

def plot_confusion_matrix(cm, model_name, prompt):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {model_name}\nPrompt: {prompt}')
    plt.show()

def confusion_matrix_for_models(file_name, model_names):
    with open(f'data/{file_name}.json', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)

    # Extract the prompt from the filename
    prompt = file_name.split('-')[1] if '-' in file_name else 'N/A'

    # Apply the get_polarity function to each row and print the result
    df['real_polarity'] = df.apply(lambda row: get_polarity(row), axis=1)

    for model_name in model_names:
        df['predicted_polarity'] = df.apply(lambda row: get_predicted_polarity(row, model_name), axis=1)
        cm = confusion_matrix(df['real_polarity'], df['predicted_polarity'], labels=classes)
        print(f"Confusion Matrix for {model_name}:")
        print(cm)
        plot_confusion_matrix(cm, model_name, prompt)

def main(model_names):
    for file_name in get_files_names():
        if file_name == 'dataset':
            continue
        confusion_matrix_for_models(file_name, model_names)

if __name__ == "__main__":
    classes = ['positive', 'neutral', 'negative']
    main(models)
