import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o DataFrame
df = pd.read_csv('compare.csv')

# Filtrar apenas os modelos LLMs
llm_models = df[~df['model'].isin(['SentiStrength', 'SentiStrengthSE', 'SentiCR', 'DEVA', 'Senti4SD'])].copy()

# Definir a nova ordem desejada dos prompts e nomes mais bonitos
prompt_order = ['basic_prompt', 'complex_zero_shot', 'cot', 'few_shot', 'one_shot', 'simple_zero_shot']
pretty_prompt_names = ['Basic Prompt', 'Complex Zero-Shot', 'Chain of Thought', 'Few-Shot', 'One-Shot', 'Simple Zero-Shot']

# Mapear os prompts para os nomes mais bonitos
prompt_mapping = dict(zip(prompt_order, pretty_prompt_names))
llm_models['pretty_prompt'] = llm_models['prompt'].map(prompt_mapping)

# Garantir que a coluna 'prompt' siga essa ordem
llm_models['pretty_prompt'] = pd.Categorical(llm_models['pretty_prompt'], categories=pretty_prompt_names, ordered=True)

# Reordenar o DataFrame com base na nova ordem dos prompts
llm_models = llm_models.sort_values('pretty_prompt')

# Definir uma paleta de cores em tons de preto e cinza
color_palette = sns.color_palette("Greys", n_colors=len(llm_models['model'].unique()))

plt.figure(figsize=(12, 6))

# Plotar cada modelo com a cor específica
for i, (model, color) in enumerate(zip(llm_models['model'].unique(), color_palette)):
    subset = llm_models[llm_models['model'] == model]
    plt.plot(subset['pretty_prompt'], subset['f1-score'], marker='o', label=model, color=color)

plt.xlabel('Prompt')
plt.ylabel('F1-score')
plt.title('Variação de Desempenho por Prompt')
plt.legend(title='Modelos', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
