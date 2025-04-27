import torch
from datasets import Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os
from typing import List, Dict, Optional

import pandas as pd
import json

from huggingface_hub import login


def tokenize_and_mask(example, tokenizer, max_length=512):
    full_text = example["text"]

    # Find where the assistant's output starts
    assistant_tag = "<|im_start|>assistant:"
    assistant_start = full_text.find(assistant_tag)

    if assistant_start == -1:
        raise ValueError("No assistant tag found in text.")

    # Get index of start and end of assistant message
    assistant_content_start = assistant_start + len(assistant_tag)
    assistant_end = full_text.find("<|im_end|>", assistant_content_start)

    if assistant_end == -1:
        raise ValueError("No <|im_end|> after assistant found in text.")

    # Create loss mask: -100 outside assistant, real labels inside assistant
    tokenized = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    input_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]
    labels = [-100] * len(input_ids)

    for i, (start, end) in enumerate(offsets):
        if assistant_content_start <= start < assistant_end:
            labels[i] = input_ids[i]

    tokenized["labels"] = labels
    tokenized.pop("offset_mapping")  # Clean up
    return tokenized


def prepare_train_datav1(system_prompt, data, tokenizer, max_length=512):
    """
    Prepares a dataset for fine-tuning an LLM for tool use, formatting responses correctly.

    Args:
        system_prompt (str): The system-level instruction for all examples.
        data (list of dicts): A list of training examples, where each dict contains:
            - "prompt": The user's input.
            - "tool": The name of the tool to be called.
            - "arguments": A dictionary of parameters for the tool.

    Returns:
        Dataset: A Hugging Face Dataset object formatted for fine-tuning.
    """

    # Create a new column called "text" with the formatted prompt-response pairs
    def format_example(row):
        return (
            f"<|im_start|>system: {system_prompt}<|im_end|>\n"
            f"<|im_start|>user: {row['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant:"
            f"{json.dumps(
                {'tool': row['tool'], 'parameters': row['arguments']})}"
            f"<|im_end|>\n"
        )

    # Convert the data to a Pandas DataFrame
    data_df = pd.DataFrame(data)
    data_df["text"] = data_df.apply(format_example, axis=1)
    # print(f"Data df\n{data_df}")

    # Create a new Dataset from the DataFrame
    dataset = Dataset.from_pandas(data_df)

    # Tokenize the dataset
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    # tokenized_dataset = dataset.map(
    #    tokenize_function, remove_columns=dataset.column_names)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_mask(x, tokenizer),
        remove_columns=dataset.column_names
    )

    print(f"Tokenized dataset\n{tokenized_dataset}")
    return tokenized_dataset


def convert_to_chat_format(
    raw_data: List[Dict],
    system_prompt: Optional[str] = None,
    output_path: Optional[str] = None
) -> List[Dict]:
    """
    Converts a list of samples from your tool-calling format into chat-style format for fine-tuning.

    Args:
        raw_data (List[Dict]): List of samples with 'prompt', 'tool', and 'arguments'.
        system_prompt (str, optional): Optional system message.
        output_path (str, optional): If provided, saves the converted dataset as JSONL.

    Returns:
        List[Dict]: Converted dataset in chat format.
    """
    chat_data = []

    for sample in raw_data:
        user_message = {"role": "user", "content": sample["prompt"]}
        assistant_message = {
            "role": "assistant",
            "content": json.dumps({
                "tool": sample["tool"],
                "arguments": sample["arguments"]
            }, ensure_ascii=False, indent=None)
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend([user_message, assistant_message])

        chat_data.append({"messages": messages})

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in chat_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return chat_data


system_prompt = "Generate a JSON response for BittyGPT, a quadruped robot dog simulator with 9 degrees of freedom (DOF). Your job is to generate a motion set for BittyGPT immitating a playful puppy behaviour. Assume any question can answered through calling tool with parameters. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments. You will construct a motion set immitating behaviour of a playful puppy in response to a given command."

more_training_data = [
    # Same 'neutral stand' pose, new prompts
    {
        "prompt": "Hey there, buddy!",
        "tool": "execute_command",
        "arguments": {
            "poses": [{"left_front": {"schoulder": 30, "elbow": 30},
                       "left_back": {"schoulder": 30, "elbow": 35},
                       "right_front": {"schoulder": 30, "elbow": 35},
                       "right_back": {"schoulder": 30, "elbow": 30}}],
            "description": "neutral stand pose, friendly greeting posture"
        }
    },
    {
        "prompt": "Hello pup!",
        "tool": "execute_command",
        "arguments": {
            "poses": [{"left_front": {"schoulder": 30, "elbow": 30},
                       "left_back": {"schoulder": 30, "elbow": 35},
                       "right_front": {"schoulder": 30, "elbow": 35},
                       "right_back": {"schoulder": 30, "elbow": 30}}],
            "description": "neutral stand pose, friendly greeting posture"
        }
    },
    {
        "prompt": "What's up, cutie?",
        "tool": "execute_command",
        "arguments": {
            "poses": [{"left_front": {"schoulder": 30, "elbow": 30},
                       "left_back": {"schoulder": 30, "elbow": 35},
                       "right_front": {"schoulder": 30, "elbow": 35},
                       "right_back": {"schoulder": 30, "elbow": 30}}],
            "description": "neutral stand pose, friendly greeting posture"
        }
    },

    # Sit pose with prompt variety
    {
        "prompt": "Take a seat!",
        "tool": "execute_command",
        "arguments": {
            "poses": [{"left_front": {"schoulder": 45, "elbow": 45},
                       "left_back": {"schoulder": 105, "elbow": -45},
                       "right_front": {"schoulder": 45, "elbow": 45},
                       "right_back": {"schoulder": 105, "elbow": -45}}],
            "description": "sit command with a calm, obedient posture"
        }
    },
    {
        "prompt": "Sit down, puppy!",
        "tool": "execute_command",
        "arguments": {
            "poses": [{"left_front": {"schoulder": 45, "elbow": 45},
                       "left_back": {"schoulder": 105, "elbow": -45},
                       "right_front": {"schoulder": 45, "elbow": 45},
                       "right_back": {"schoulder": 105, "elbow": -45}}],
            "description": "obedient sit pose, training cue"
        }
    },

    # New: "Stretch" pose (bow down)
    {
        "prompt": "Stretch it out!",
        "tool": "execute_command",
        "arguments": {
            "poses": [{"left_front": {"schoulder": 60, "elbow": 80},
                       "left_back": {"schoulder": 20, "elbow": 10},
                       "right_front": {"schoulder": 60, "elbow": 80},
                       "right_back": {"schoulder": 20, "elbow": 10}}],
            "description": "play bow or downward stretch"
        }
    },
    {
        "prompt": "Take a bow!",
        "tool": "execute_command",
        "arguments": {
            "poses": [{"left_front": {"schoulder": 60, "elbow": 80},
                       "left_back": {"schoulder": 20, "elbow": 10},
                       "right_front": {"schoulder": 60, "elbow": 80},
                       "right_back": {"schoulder": 20, "elbow": 10}}],
            "description": "playful bow/stretch"
        }
    },

    # New: "Wiggle" pose (sway one side)
    {
        "prompt": "Wiggle wiggle!",
        "tool": "execute_command",
        "arguments": {
            "poses": [{"left_front": {"schoulder": 20, "elbow": 20},
                       "left_back": {"schoulder": 40, "elbow": 20},
                       "right_front": {"schoulder": 50, "elbow": 20},
                       "right_back": {"schoulder": 30, "elbow": 20}}],
            "description": "hip wiggle to one side, playful gesture"
        }
    },
    {
        "prompt": "Do a happy dance!",
        "tool": "execute_command",
        "arguments": {
            "poses": [{"left_front": {"schoulder": 20, "elbow": 20},
                       "left_back": {"schoulder": 40, "elbow": 20},
                       "right_front": {"schoulder": 50, "elbow": 20},
                       "right_back": {"schoulder": 30, "elbow": 20}}],
            "description": "playful motion with side sway"
        }
    }
]

training_data1 = [
    {
        "prompt": "Roll over!",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {"schoulder": 90, "elbow": 45},
                    "left_back": {"schoulder": 90, "elbow": 45},
                    "right_front": {"schoulder": 90, "elbow": 45},
                    "right_back": {"schoulder": 90, "elbow": 45}
                }
            ],
            "description": "playful roll over on back"
        }
    },
    {
        "prompt": "Jump!",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {"schoulder": 10, "elbow": -10},
                    "left_back": {"schoulder": 100, "elbow": -30},
                    "right_front": {"schoulder": 10, "elbow": -10},
                    "right_back": {"schoulder": 100, "elbow": -30}
                }
            ],
            "description": "leap upward, all legs pushing off"
        }
    },
    {
        "prompt": "Shake a paw!",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {"schoulder": 60, "elbow": 90},
                    "left_back": {"schoulder": 30, "elbow": 30},
                    "right_front": {"schoulder": 30, "elbow": 30},
                    "right_back": {"schoulder": 30, "elbow": 30}
                }
            ],
            "description": "raise left front paw for handshake"
        }
    },
    {
        "prompt": "Beg!",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {"schoulder": 100, "elbow": -10},
                    "left_back": {"schoulder": 80, "elbow": 40},
                    "right_front": {"schoulder": 100, "elbow": -10},
                    "right_back": {"schoulder": 80, "elbow": 40}
                }
            ],
            "description": "upright sitting begging pose"
        }
    },
    {
        "prompt": "Dance!",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {"schoulder": 60, "elbow": 20},
                    "left_back": {"schoulder": 20, "elbow": 40},
                    "right_front": {"schoulder": 30, "elbow": 20},
                    "right_back": {"schoulder": 30, "elbow": 40}
                }
            ],
            "description": "playful twist, mid dance"
        }
    },
    {
        "prompt": "Wave!",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {"schoulder": 100, "elbow": 0},
                    "left_back": {"schoulder": 30, "elbow": 30},
                    "right_front": {"schoulder": 30, "elbow": 30},
                    "right_back": {"schoulder": 30, "elbow": 30}
                }
            ],
            "description": "left front paw raised high for waving"
        }
    },
    {
        "prompt": "Sniff!",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {"schoulder": 60, "elbow": 60},
                    "left_back": {"schoulder": 40, "elbow": 60},
                    "right_front": {"schoulder": 60, "elbow": 60},
                    "right_back": {"schoulder": 40, "elbow": 60}
                }
            ],
            "description": "head down sniffing ground pose"
        }
    },
    {
        "prompt": "Tail wag!",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {"schoulder": 30, "elbow": 30},
                    "left_back": {"schoulder": 30, "elbow": 30},
                    "right_front": {"schoulder": 30, "elbow": 30},
                    "right_back": {"schoulder": 30, "elbow": 30}
                }
            ],
            "description": "neutral pose, tail wag simulated in animation layer"
        }
    },
    {
        "prompt": "Come here!",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {"schoulder": 45, "elbow": 20},
                    "left_back": {"schoulder": 40, "elbow": 20},
                    "right_front": {"schoulder": 45, "elbow": 20},
                    "right_back": {"schoulder": 40, "elbow": 20}
                }
            ],
            "description": "motion cue to approach user"
        }
    },
    {
        "prompt": "Catch!",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {"schoulder": 15, "elbow": -20},
                    "left_back": {"schoulder": 90, "elbow": -20},
                    "right_front": {"schoulder": 15, "elbow": -20},
                    "right_back": {"schoulder": 90, "elbow": -20}
                }
            ],
            "description": "leap forward with front legs stretched to catch object"
        }
    }]

base_train_data = [
    {
        "prompt": "Hi, cute puppy!",
        "tool": "execute_command",
        "arguments": {"poses": [{"left_front": {"schoulder": 30, "elbow": 30}, "left_back": {"schoulder": 30, "elbow": 35}, "right_front": {"schoulder": 30, "elbow": 35}, "right_back": {"schoulder": 30, "elbow": 30}}], "description": "Neutral stand pose, attention to next command"}
    },
    {
        "prompt": "Hi!",
        "tool": "execute_command",
        "arguments": {"poses": [{"left_front": {"schoulder": 30, "elbow": 30}, "left_back": {"schoulder": 30, "elbow": 35}, "right_front": {"schoulder": 30, "elbow": 35}, "right_back": {"schoulder": 30, "elbow": 30}}], "description": "Neutral stand pose, attention to next command"}
    },
    {
        "prompt": "Good Boy",
        "tool": "execute_command",
        "arguments": {"poses": [{"left_front": {"schoulder": 30, "elbow": 30}, "left_back": {"schoulder": 30, "elbow": 35}, "right_front": {"schoulder": 30, "elbow": 35}, "right_back": {"schoulder": 30, "elbow": 30}}], "description": "Neutral stand pose, attention to next command"}
    },
    {
        "prompt": "Sit!",
        "tool": "execute_command",
        "arguments": {"poses": [{"left_front": {"schoulder": 45, "elbow": 45}, "left_back": {"schoulder": 105, "elbow": -45}, "right_front": {"schoulder": 45, "elbow": 45}, "right_back": {"schoulder": 105, "elbow": -45}}], "description": "sit"}
    },
    {
        "prompt": "Stand!",
        "tool": "execute_command",
        "arguments": {"poses": [{"left_front": {"schoulder": 30, "elbow": 30}, "left_back": {"schoulder": 30, "elbow": 30}, "right_front": {"schoulder": 30, "elbow": 30}, "right_back": {"schoulder": 30, "elbow": 30}}], "description": "up"}
    },
    {
        "prompt": "Up!",
        "tool": "execute_command",
        "arguments": {"poses": [{"left_front": {"schoulder": 30, "elbow": 30}, "left_back": {"schoulder": 30, "elbow": 30}, "right_front": {"schoulder": 30, "elbow": 30}, "right_back": {"schoulder": 30, "elbow": 30}}], "description": "up"}
    },
    {
        "prompt": "Stop!",
        "tool": "execute_command",
        "arguments": {"poses": [{"left_front": {"schoulder": 30, "elbow": 30}, "left_back": {"schoulder": 30, "elbow": 30}, "right_front": {"schoulder": 30, "elbow": 30}, "right_back": {"schoulder": 30, "elbow": 30}}], "description": "up"}
    },
    {
        "prompt": "Rest!",
        "tool": "execute_command",
        "arguments": {"poses": [{"left_front": {"schoulder": 0, "elbow": 0}, "left_back": {"schoulder": 0, "elbow": 0}, "right_front": {"schoulder": 0, "elbow": 0}, "right_back": {"schoulder": 0, "elbow": 0}}], "description": "rest, lay"}
    },
    {
        "prompt": "Lay!",
        "tool": "execute_command",
        "arguments": {"poses": [{"left_front": {"schoulder": 0, "elbow": 0}, "left_back": {"schoulder": 0, "elbow": 0}, "right_front": {"schoulder": 0, "elbow": 0}, "right_back": {"scholder": 0, "elbow": 0}}], "description": "rest, lay"}
    },
]

pow_dataset = [
    {
        "prompt": "Shake",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Paw",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Gimme paw",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "High five",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "High paw",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Hello",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Fist bump",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Paw five",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Slap me some paw",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Knucks",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Tag it",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Boom",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Present paw",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Salute",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Offer paw",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    },
    {
        "prompt": "Handshake",
        "tool": "execute_command",
        "arguments": {
            "poses": [
                {
                    "left_front": {
                        "schoulder": 35,
                        "elbow": 75
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 60
                    },
                    "right_front": {
                        "schoulder": 120,
                        "elbow": -40
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -99,
                        "elbow": 40
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": -90,
                        "elbow": 62
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 75
                    },
                    "right_front": {
                        "schoulder": 125,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 95,
                        "elbow": -30
                    }
                },
                {
                    "left_front": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "left_back": {
                        "schoulder": 45,
                        "elbow": 45
                    },
                    "right_front": {
                        "schoulder": 105,
                        "elbow": -45
                    },
                    "right_back": {
                        "schoulder": 105,
                        "elbow": -45
                    }
                },
                {
                    "left_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "left_back": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_front": {
                        "schoulder": 30,
                        "elbow": 30
                    },
                    "right_back": {
                        "schoulder": 30,
                        "elbow": 30
                    }
                }
            ],
            "description": "Hi gesture front leg up"
        }
    }
]


def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16"
        # , bnb_4bit_use_double_quant=True
        # ,llm_int8_enable_fp32_cpu_offload=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


model_id = "meta-llama/Llama-3.2-1B-Instruct"
model, tokenizer = get_model_and_tokenizer(model_id)


data = convert_to_chat_format(base_train_data + more_training_data +
                              training_data1 + pow_dataset, system_prompt, 'dataset.jsonl')

peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"
)

output_model = "Llama-3.2-1B-Instruct-bitty-tunedByAlina"
login(new_session="")


os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(torch.cuda.memory_summary())

training_arguments = TrainingArguments(
    output_dir=output_model,
    report_to=None,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=1e-3,  # for overfitting
    lr_scheduler_type="constant",  # for overfittin
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=50,
    fp16=True,
    push_to_hub=True,
    label_names=["labels"])

torch.cuda.empty_cache()
print(torch.cuda.memory_summary())

trainer = SFTTrainer(
    model=model,
    train_dataset=Dataset.from_list(data),
    # formatting_func=formatted_train,
    peft_config=peft_config,
    args=training_arguments,
    # packing=True,
)
trainer.train()
