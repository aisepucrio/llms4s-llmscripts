import pandas as pd
import json
from matplotlib import pyplot as plt

def read_json(file):
    with open(file, 'r') as f:
        return json.load(f)
    
def get_polarity_distribuiton(file):
    data = read_json(file)
    emotion_distribution = {}
    for item in data:
        right_polarity = item["part2_aggregate"]["polarity"] if item["part2_aggregate"]["polarity"] != "undefined" else item["discussion_polarity"]
        if right_polarity in emotion_distribution:
            emotion_distribution[right_polarity] += 1
        else:
            emotion_distribution[right_polarity] = 1

    return emotion_distribution

def pizza_plot(emotion_distribution):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    # Define um esquema de cores
    colors = plt.cm.Paired(range(len(emotion_distribution)))
    
    # Plota o gráfico de pizza
    wedges, texts, autotexts = ax.pie(emotion_distribution.values(), labels=emotion_distribution.keys(), 
                                      autopct='%1.1f%%', startangle=140, colors=colors, textprops=dict(color="w"))
    
    # Adiciona um título com espaçamento ajustado
    ax.set_title("Distribuição de Emoções", fontsize=16, weight='bold', pad=20)
    
    # Ajusta a posição das labels e das porcentagens
    for text in texts:
        text.set_fontsize(12)
        text.set_color('black')
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_color('white')
    
    # Adiciona uma legenda
    ax.legend(wedges, emotion_distribution.keys(), title="Emoções", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Garante que o gráfico de pizza seja circular
    ax.axis('equal')
    
    # Exibe o gráfico
    plt.show()

if __name__ == "__main__":
    file = "data/dataset.json"
    emotion_distribution = get_polarity_distribuiton(file)
    pizza_plot(emotion_distribution)
