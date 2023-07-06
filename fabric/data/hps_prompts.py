import json
import os
import random
import re
from typing import Optional

def sample_prompts(
    max_num_prompts=-1,
    dataset_path="resources/hps_dataset",
    train_file="preference_train.json",
    test_file="preference_test.json",
    split="train",
    seed: Optional[int] = 0,
    fix_spacing: bool = True,
):
    """Sample prompts from the preference dataset.

    Args:
        dataset_path (str): Path to the preference dataset.
        train_file (str): Name of the training file.
        test_file (str): Name of the test file.
        split (str): Either "train" or "test".
        max_num_prompts (int): Number of prompts to sample.

    Returns:
        list: List of sampled prompts.
    """
    assert split in ["train", "test"]
    if split == "train":
        file_path = os.path.join(dataset_path, train_file)
    elif split == "test":
        file_path = os.path.join(dataset_path, test_file)
    else:
        raise ValueError("split must be either train or test")

    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    prompts = []
    for item in data:
        prompts.append(item["prompt"])

    random.Random(seed).shuffle(prompts)

    if max_num_prompts < 0:
        output = prompts
    output = prompts[:max_num_prompts]

    if fix_spacing:
        numbers_regex = r"((?<=[0-9])|(?<=[0-9]\.)|(?<=\b[b-zA-Z])) ((?=[0-9])|(?=[a-zA-Z]\b)|(?=(th|st|nd|rd)\b))"
        output = [re.sub(numbers_regex, "", prompt) for prompt in output]
        output = [re.sub(r" / ", "/", prompt) for prompt in output]
        output = [re.sub(r"\( ", "(", prompt) for prompt in output]
        output = [re.sub(r" \)", ")", prompt) for prompt in output]
        output = [re.sub(r" - ", "-", prompt) for prompt in output]
        output = [re.sub(r" :", ":", prompt) for prompt in output]
        output = [re.sub(r" _ ", "_", prompt) for prompt in output]
    return output


if __name__ == "__main__":
    prompts = sample_prompts(
        "resources/hps_dataset",
        split="test",
        max_num_prompts=10,
        seed=None,
    )
    print(prompts)
