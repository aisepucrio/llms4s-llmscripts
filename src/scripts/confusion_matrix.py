from run_llm import models
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def get_files_names():
    folder_path = 'data'
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
    files_names = [file.split('.')[0] for file in json_files]
    return files_names

def get_polarity(message):
    try:
        return message["part2_aggregate"]["polarity"] if message["part2_aggregate"]["polarity"] != "undefined" else message["discussion_polarity"]
    except KeyError:
        return "undefined"  # Substitua por um valor padrão conforme necessário

def get_predicted_polarity(row, model_name):
    return row['tools'].get(model_name, 'undefined')

def plot_confusion_matrix(cm, report, model_name, prompt):
    plt.figure(figsize=(17, 7))  # Aumentando a figura para incluir o texto ao lado
    plt.subplot(1, 2, 1)  # Matriz de confusão à esquerda
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {model_name}\nPrompt: {prompt}')
    plt.subplot(1, 2, 2)  # Relatório de classificação à direita
    plt.axis('off')  # Desativando os eixos para o texto
    plt.text(0.5, 0.5, report, ha='center', va='center', fontsize=12)
    plt.show()

def confusion_matrix_for_models(file_name, model_names):
    with open(f'data/{file_name}.json', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df['real_polarity'] = df.apply(lambda row: get_polarity(row), axis=1)
    df = df[df['real_polarity'].isin(classes)]  # Filtrando apenas as polaridades esperadas

    prompt = file_name.split('-')[1] if '-' in file_name else 'N/A'

    for model_name in model_names:
        df['predicted_polarity'] = df.apply(lambda row: get_predicted_polarity(row, model_name), axis=1)
        df = df[df['predicted_polarity'].isin(classes)]  # Filtrando apenas as polaridades esperadas
        cm = confusion_matrix(df['real_polarity'], df['predicted_polarity'], labels=classes)
        report = classification_report(df['real_polarity'], df['predicted_polarity'], target_names=classes)
        print(f"Confusion Matrix for {model_name}:")
        print(cm)
        print(f"Classification Report for {model_name}:\n{report}")
        plot_confusion_matrix(cm, report, model_name, prompt)

def main(model_names):
    for file_name in get_files_names():
        if file_name == 'dataset':
            continue
        confusion_matrix_for_models(file_name, model_names)

if __name__ == "__main__":
    classes = ['positive', 'neutral', 'negative']
    main(models)
