from run_llm import models
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import os
import json

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

def confusion_matrix_for_models(file_name, model_names, classes):
    with open(f'data/{file_name}.json', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df['real_polarity'] = df.apply(lambda row: get_polarity(row), axis=1)
    df = df[df['real_polarity'].isin(classes)]  

    results = []

    for model_name in model_names:
        df['predicted_polarity'] = df.apply(lambda row: get_predicted_polarity(row, model_name), axis=1)
        df = df[df['predicted_polarity'].isin(classes)]  
        
        prompt = file_name.split('-')[1]
        precision, recall, f1_score, _ = precision_recall_fscore_support(df['real_polarity'], df['predicted_polarity'], labels=classes, average='weighted')
        results.append((file_name.split('-')[-1], prompt, model_name, precision, recall, f1_score))
    
    return results

def analyze_performance_difference(dataframe):
    raw_data = dataframe[dataframe['message_type'] == 'raw']
    clean_data = dataframe[dataframe['message_type'] == 'clean']

    diff_df = pd.DataFrame(columns=['model', 'prompt', 'metric', 'raw', 'clean', 'difference'])

    for model in dataframe['model'].unique():
        for metric in ['precision', 'recall', 'f1-score']:
            for prompt in dataframe['prompt'].unique():
                raw_value = raw_data[(raw_data['model'] == model) & (raw_data['prompt'] == prompt)][metric].values[0]
                clean_value = clean_data[(clean_data['model'] == model) & (clean_data['prompt'] == prompt)][metric].values[0]
                diff = clean_value - raw_value
                diff_df = pd.concat([diff_df, pd.DataFrame([[model, prompt, metric, raw_value, clean_value, diff]], columns=['model', 'prompt', 'metric', 'raw', 'clean', 'difference'])], ignore_index=True)

    print(diff_df)
    diff_df.to_csv('performance_difference.csv', index=False)
    return diff_df

def main(model_names):
    classes = ['positive', 'neutral', 'negative']
    dataframe = pd.DataFrame(columns=['message_type', 'prompt', 'model', 'precision', 'recall', 'f1-score'])

    for file_name in get_files_names():
        if file_name == 'dataset':
            continue
        
        results = confusion_matrix_for_models(file_name, model_names, classes)
        for result in results:
            message_type, prompt, model_name, precision, recall, f1_score = result
            dataframe = pd.concat([dataframe, pd.DataFrame([[message_type, prompt, model_name, precision, recall, f1_score]], columns=['message_type', 'prompt', 'model', 'precision', 'recall', 'f1-score'])], ignore_index=True)
    
    dataframe['message_type'] = pd.Categorical(dataframe['message_type'], categories=['raw', 'clean'], ordered=True)
    dataframe = dataframe.sort_values(by=['model', 'message_type']).reset_index(drop=True)

    print(dataframe)
    dataframe.to_csv('compare.csv', index=False)

    # Analisando a diferença de performance
    analyze_performance_difference(dataframe)

if __name__ == "__main__":
    main(models)
