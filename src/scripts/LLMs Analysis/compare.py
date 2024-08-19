from sklearn.metrics import classification_report
import pandas as pd
import os
import json

models = ['llama3.1:8b ', 'llama3:instruct', 'gemma:instruct', 'mistral:instruct', 'SentiStrength', 'SentiStrengthSE', 'SentiCR', 'DEVA', 'Senti4SD']
tools = ['SentiStrength', 'SentiStrengthSE', 'SentiCR', 'DEVA', 'Senti4SD']

def get_files_names():
    folder_path = 'data'
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
    files_names = [file.split('.')[0] for file in json_files]
    return files_names

def get_polarity(message):
    try:
        return message["part2_aggregate"]["polarity"] if message["part2_aggregate"]["polarity"] != "undefined" else message["discussion_polarity"]
    except KeyError:
        return "undefined"

def get_predicted_polarity(row, model_name):
    return row['tools'].get(model_name, 'undefined')

def generate_csv(data):
    df = pd.DataFrame(data, columns=['model', 'prompt', 'precision', 'recall', 'f1-score', 'accuracy'])
    df.to_csv('compare.csv', index=False)

def confusion_matrix_for_models(file_name, model_name):
    with open(f'data/{file_name}.json', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df['real_polarity'] = df.apply(lambda row: get_polarity(row), axis=1)
    df = df[df['real_polarity'].isin(classes)]

    if df.empty:
        print(f"No valid data for model {model_name} on prompt {file_name}. Skipping...")
        return

    prompt = file_name.split('-')[1] if '-' in file_name else 'N/A'

    df['predicted_polarity'] = df.apply(lambda row: get_predicted_polarity(row, model_name), axis=1)
    df = df[df['predicted_polarity'].isin(classes)]

    if df.empty:
        print(f"No valid predictions for model {model_name} on prompt {file_name}. Skipping...")
        return

    report = classification_report(df['real_polarity'], df['predicted_polarity'], target_names=classes, labels=classes, output_dict=True, zero_division=0)

    print(f"Classification Report for {model_name} on prompt {prompt}:\n")
    print(classification_report(df['real_polarity'], df['predicted_polarity'], target_names=classes, labels=classes, zero_division=0))

    metrics = report['weighted avg']
    accuracy = report['accuracy']
    classification_data.append({
        'model': model_name,
        'prompt': prompt,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1-score': metrics['f1-score'],
        'accuracy': accuracy
    })

def confusion_matrix_for_tools(file_name, tool_name):
    with open(f'data/{file_name}.json', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['real_polarity'] = df.apply(lambda row: get_polarity(row), axis=1)
    df = df[df['real_polarity'].isin(classes)]

    if df.empty:
        print(f"No valid data for tool {tool_name} on prompt {file_name}. Skipping...")
        return

    df['predicted_polarity'] = df.apply(lambda row: get_predicted_polarity(row, tool_name), axis=1)
    df = df[df['predicted_polarity'].isin(classes)]

    if df.empty:
        print(f"No valid predictions for tool {tool_name} on prompt {file_name}. Skipping...")
        return

    report = classification_report(df['real_polarity'], df['predicted_polarity'], target_names=classes, labels=classes, output_dict=True, zero_division=0)

    metrics = report['weighted avg']
    accuracy = report['accuracy']
    classification_data_tools[tool_name]['precision'].append(metrics['precision'])
    classification_data_tools[tool_name]['recall'].append(metrics['recall'])
    classification_data_tools[tool_name]['f1-score'].append(metrics['f1-score'])
    classification_data_tools[tool_name]['accuracy'].append(accuracy)

def main(model_names, tool_names):
    for model_name in model_names:
        if model_name in tool_names:
            continue
        print(f"Results for model: {model_name}\n")
        for file_name in get_files_names():
            if file_name == 'dataset':
                continue
            confusion_matrix_for_models(file_name, model_name)
    
    for tool_name in tool_names:
        classification_data_tools[tool_name] = {'precision': [], 'recall': [], 'f1-score': [], 'accuracy': []}
        print(f"Results for tool: {tool_name}\n")
        for file_name in get_files_names():
            if file_name == 'dataset':
                continue
            confusion_matrix_for_tools(file_name, tool_name)
        avg_precision = sum(classification_data_tools[tool_name]['precision']) / len(classification_data_tools[tool_name]['precision'])
        avg_recall = sum(classification_data_tools[tool_name]['recall']) / len(classification_data_tools[tool_name]['recall'])
        avg_f1_score = sum(classification_data_tools[tool_name]['f1-score']) / len(classification_data_tools[tool_name]['f1-score'])
        avg_accuracy = sum(classification_data_tools[tool_name]['accuracy']) / len(classification_data_tools[tool_name]['accuracy'])
        classification_data.append({
            'model': tool_name,
            'prompt': 'N/A',
            'precision': avg_precision,
            'recall': avg_recall,
            'f1-score': avg_f1_score,
            'accuracy': avg_accuracy
        })
    
    generate_csv(classification_data)

if __name__ == "__main__":
    classes = ['positive', 'neutral', 'negative']
    classification_data = []
    classification_data_tools = {}
    main(models, tools)
