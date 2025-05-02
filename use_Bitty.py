from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers import pipeline

from huggingface_hub import login

import json
import re


def execute_command(arguments: dict):
    print("Executing command with arguments:")
    print(json.dumps(arguments, indent=2))
    # Here you'd call your robot interface, e.g. send_serial_command(arguments)
    return "Command executed"


# Tool dispatcher dictionary
TOOL_FUNCTIONS = {
    "execute_command": execute_command,
    # Add more tools here if needed
}


def parse_and_execute(model_output: list):
    if not model_output:
        raise ValueError("Empty model output.")

    # Join list to single string if needed
    output_text = model_output[0] if isinstance(
        model_output, list) else model_output

    # Find JSON substring (naively, but works for well-formed outputs)
    match = re.search(r"\{[\s]*\"tool\".*\}", output_text, re.DOTALL)
    if not match:
        raise ValueError("No tool JSON found in model output.")

    try:
        tool_data = json.loads(match.group())
        tool_name = tool_data["tool"]
        arguments = tool_data["arguments"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to parse tool call: {e}")

    # Call the tool
    if tool_name in TOOL_FUNCTIONS:
        return TOOL_FUNCTIONS[tool_name](arguments)
    else:
        raise ValueError(f"Tool '{tool_name}' not implemented.")


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


def build_prompt(system_prompt, user_input):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]


SYSTEM_PROMPT = "Generate a JSON response for BittyGPT, a quadruped robot dog simulator with 9 degrees of freedom (DOF). Your job is to generate a motion set for BittyGPT immitating a playful puppy behaviour. Assume any question can answered through calling tool with parameters. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments. You will construct a motion set immitating behaviour of a playful puppy in response to a given command."


def main(model_id, hf_key):

    login(new_session="True", token=hf_key)

    model, tokenizer = get_model_and_tokenizer(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Create pipeline
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Simple chat loop

    print("Type 'exit' to quit")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        messages = build_prompt(SYSTEM_PROMPT, user_input)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt",
                           padding=True).to('cuda')

        input_ids = inputs.input_ids.to('cuda')
        attention_mask = inputs["attention_mask"].to('cuda')

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_new_tokens=1024, pad_token_id=tokenizer.pad_token_id)
        model_output = tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        print(f"Responce: {model_output}")
        parse_and_execute(model_output)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run BittyGPT chatbot.")
    parser.add_argument("--model_id", type=str,
                        required=True, help="Model ID to load.")
    parser.add_argument("--hf_key", type=str, required=True,
                        help="Hugging Face API key.")
    args = parser.parse_args()

    main(args.model_id, args.hf_key)
