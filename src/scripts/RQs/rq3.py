import pandas as pd
import matplotlib.pyplot as plt

# Função para comparar desempenho de LLMs e ferramentas específicas
def compare_best_prompt(file_path, metric='f1-score'):
    # Carregar os dados do arquivo CSV
    data = pd.read_csv(file_path)
    
    # Verificar se a métrica fornecida é válida
    if metric not in data.columns:
        raise ValueError(f"Métrica inválida: {metric}. Escolha entre 'precision', 'recall', 'f1-score' e 'accuracy'.")
    
    # Ferramentas específicas para incluir na análise
    include_tools = ['SentiStrength', 'SentiStrengthSE', 'SentiCR', 'DEVA', 'Senti4SD']
    
    # Modelos LLM específicos
    llms = data[~data['model'].isin(include_tools)]['model'].unique()

    # Criar um DataFrame para armazenar os melhores resultados por métrica para os LLMs
    best_results = []

    for llm in llms:
        best_prompt = data[data['model'] == llm].sort_values(by=metric, ascending=False).iloc[0]
        best_results.append({
            'model': llm.replace(':instruct', '').capitalize() + ' (Best Prompt)',
            'prompt': best_prompt['prompt'],
            metric: best_prompt[metric]
        })

    best_results_df = pd.DataFrame(best_results)

    # Adicionar resultados das ferramentas específicas
    for tool in include_tools:
        tool_data = data[data['model'] == tool]
        if not tool_data.empty:
            best_results_df = best_results_df._append({
                'model': tool,
                'prompt': 'N/A',
                metric: tool_data[metric].mean()
            }, ignore_index=True)

    # Criar dicionário para armazenar as posições das barras
    bar_positions = {model: i for i, model in enumerate(best_results_df['model'].unique())}

    # Definir largura das barras
    bar_width = 0.4

    # Definir cor para as barras em tom de cinza escuro
    bar_color = '#696969'

    # Plotar os resultados
    fig, ax = plt.subplots(figsize=(20, 6))  # Aumentar a largura e reduzir a altura
    
    # Adicionar grade
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', axis='y')

    metric_scores = best_results_df[['model', 'prompt', metric]]
    bar_pos = [bar_positions[model] for model in metric_scores['model']]
    bars = plt.bar(bar_pos, metric_scores[metric], bar_width, label=f'Best Prompt ({metric})', color=bar_color, edgecolor='black')

    # Adicionar os valores das barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

    plt.xlabel('Modelo')
    plt.ylabel(metric.capitalize())
    plt.title(f'Comparação de Desempenho dos Melhores Prompts e Ferramentas por Métrica ({metric.capitalize()})')
    plt.xticks([r for r in range(len(bar_positions))], list(bar_positions.keys()), rotation=45)

    plt.tight_layout()
    plt.show()

# Caminho do arquivo CSV
file_path = 'compare.csv'

# Chamar a função de comparação
compare_best_prompt(file_path, metric='f1-score')
