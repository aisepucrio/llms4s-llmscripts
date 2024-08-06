import os
import pathlib
import json
import signal
import sys
import ollama
from tqdm.auto import tqdm

client = ollama.Client(host='http://localhost:11434')

models = ['llama3:instruct', 'gemma:instruct', 'mistral:instruct', 'SentiStrength', 'SentiStrengthSE','SentiCR','DEVA','Senti4SD']

# Variável global para controlar a interrupção
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("Process interrupted. Saving progress...")

def classify_pr(pr, model, prompt_content):
    while True:
        try:
            response = ollama.generate(
                model=model,
                format="json",
                options={
                    "temperature": 1,
                    "num_ctx": 8192,
                    "num_predict": -1
                },
                stream=False,
                prompt=f"""\
Read the following message (in triple quotes, formatted as markdown):

\"\"\"
{pr}
\"\"\"

{prompt_content}.
"""
            )
        except:
            continue
        break
    try:
        return json.loads(str(response['response']))['sentiment_polarity']
    except:
        print(response['response'])
        print('Error in response')

def classify_and_save(dataset, model, output_path, prompt, message_type, sample_size=None):
    with open(dataset) as f:
        dataset = json.load(f)
    
    # Carrega o arquivo existente ou inicializa um novo
    if output_path.exists():
        with open(output_path, "r") as f:
            output_data = json.load(f)
    else:
        output_data = dataset

    # Seleciona uma amostra se sample_size for especificado
    if sample_size is not None:
        dataset = dataset[:sample_size]
    
    # Adiciona a classificação ao dataset
    for message in tqdm(dataset):
        if interrupted:
            break
        llm_polarity = classify_pr(message[message_type], model, prompt)
        if 'tools' not in message:
            message['tools'] = {}
        message['tools'][model] = llm_polarity
    
    # Atualiza output_data com as novas classificações
    for idx, message in enumerate(output_data):
        if idx < len(dataset):
            message['tools'][model] = dataset[idx]['tools'][model]

    # Salva o dataset atualizado no output_path
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

def get_prompt(prompt): 
        with open(f"./prompts/{prompt}.txt") as f:
            content = f.read()
            if not content:
                print("File is empjty")
                sys.exit(1)
            else:
                return content

def main(prompt_name, models, message_type, sample_size):
    prompt = get_prompt(prompt_name)
    dataset_path = pathlib.Path("./data/dataset.json")
    output_path = pathlib.Path(fr"./data/analysis-{prompt_name}.json")

    # Configura o manipulador de sinal para interrupções
    signal.signal(signal.SIGINT, signal_handler)

    for model in models:
        if interrupted:
            break
        classify_and_save(dataset_path, model, output_path, prompt, message_type, sample_size)
    print("Classification completed.")

if __name__ == '__main__':
    main('prompt_base', models, 'clean_message', sample_size=None)
