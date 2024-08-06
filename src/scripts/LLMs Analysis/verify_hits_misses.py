import json
import os

def load_data(file_name):
    # Define the base directory
    base_dir = r'C:\Users\breno\OneDrive\Documentos\GitHub\icprj-llms4sarticle\src\data'
    
    # Construct the full file path
    file_path = os.path.join(base_dir, file_name)
    
    # Load the JSON data from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_correct_polarity(message):
    return message["part2_aggregate"]["polarity"] if message["part2_aggregate"]["polarity"] != "undefined" else message.get("discussion_polarity", "undefined")

def evaluate_tool_polarity(data, tool_name):
    correct = 0
    incorrect = 0
    
    for message in data:
        correct_polarity = get_correct_polarity(message)
        tool_polarity = message["tools"].get(tool_name, "undefined")
        
        if tool_polarity == correct_polarity:
            correct += 1
        else:
            incorrect += 1
    
    return correct, incorrect

def main(file_name, tool_name):
    data = load_data(file_name)
    correct, incorrect = evaluate_tool_polarity(data, tool_name)
    
    print(f"Para a ferramenta '{tool_name}':")
    print(f"Acertos: {correct}")
    print(f"Erros: {incorrect}")

if __name__ == "__main__":
    # Passe o nome do arquivo e o nome da ferramenta aqui
    file_name = 'analysis-few_shot-clean_message.json'  # Substitua pelo nome do seu arquivo
    tool_name = 'mistral:instruct'    # Substitua pelo nome da ferramenta
    
    main(file_name, tool_name)
