import json
import pathlib

if __name__ == '__main__':
    dataset_path = pathlib.Path("./data/dataset-mistral-prompt2.json")
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    agreement_counter = 0
    for message in dataset:
        polarity = message["part2_aggregate"]["polarity"] if message["part2_aggregate"]["polarity"] != "undefined" else message["discussion_polarity"]
        llm_polarity = message["tools"]["mistral:7b:instruct:int4"]

        if polarity == llm_polarity:
            agreement_counter += 1

    print(f"{agreement_counter}/{len(dataset)}")

    print(f"{agreement_counter/len(dataset):.2f}")
