import os
import pathlib
import json

import ollama
from tqdm.auto import tqdm

client = ollama.Client(host='http://localhost:11434')


def classify_pr(pr):
    while True:
        try:
            response = ollama.generate(
                model="gemma:instruct",
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

As an AI model acting as a meticulous editor tasked with refining GitHub messages, your role is to first purify the content by removing any usernames, links, and non-textual elements such as commands, emojis, line breaks, and special encodings. This transformation aims to create a simple, uniform text format that's ideal for analysis.

After cleaning, your next task is to deeply engage with the sentiment of the cleaned message. As an insightful analyzer, appreciate the tone and language nuances, reflecting on how these elements convey emotions. Then, holistically assess the sentiment, recognizing the human emotions involved without depending solely on emotional keywords.

Classify the sentiment into one of three emotional states: positive, neutral, or negative, based on the overall tone and context. Utilize the Shaver emotion model, where love and joy (and related emotions) are considered positive, anger, sadness, and fear are considered negative, surprise can be positive or negative depending on the context, and neutral is considered the absence of any emotions.

Return the result as a JSON with the following format: {{"sentiment_polarity": "positive OR neutral OR negative"}}.
"""
            )
        except:
            continue
        break
    try:
        return json.loads(str(response['response']))['sentiment_polarity']
    except:
        print(response['response'])


if __name__ == '__main__':
    dataset_path = pathlib.Path("./data/dataset.json")
    with open(dataset_path) as f:
        dataset = json.load(f)
    for message in tqdm(dataset):
        llm_polarity = classify_pr(message["clean_message"])
        message['tools']['gemma:7b:instruct:int4'] = llm_polarity
    output_path = pathlib.Path("./data/dataset-gemma-prompt2.json")
    with open(output_path, "w") as f:
        json.dump(dataset, f)