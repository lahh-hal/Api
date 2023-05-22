import torch
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("inu-ai/dolly-japanese-gpt-1b", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("inu-ai/dolly-japanese-gpt-1b").to(device)
MAX_ASSISTANT_LENGTH = 100
MAX_INPUT_LENGTH = 1024
INPUT_PROMPT = r'<s>\n以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n[SEP]\n指示:\n{instruction}\n[SEP]\n入力:\n{input}\n[SEP]\n応答:\n'
NO_INPUT_PROMPT = r'<s>\n以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n[SEP]\n指示:\n{instruction}\n[SEP]\n応答:\n'
USER_NAME = "あなた"
ASSISTANT_NAME = "白翔"

def prepare_input(role_instruction, conversation_history, new_conversation):
    instruction = "".join([f"{text} " for text in role_instruction])
    instruction += " ".join(conversation_history)
    input_text = f"{USER_NAME}:{new_conversation}"

    return INPUT_PROMPT.format(instruction=instruction, input=input_text)

def format_output(output):
    output = output.lstrip("<s>").rstrip("</s>").replace("[SEP]", "").replace("\\n", "\n")
    return output

def generate_response(role_instruction, conversation_history, new_conversation):
    # 入力トークン数1024におさまるようにする
    for _ in range(8):
        input_text = prepare_input(role_instruction, conversation_history, new_conversation)
        token_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
        n = len(token_ids[0])
        if n + MAX_ASSISTANT_LENGTH <= MAX_INPUT_LENGTH:
            break
        else:
            conversation_history.pop(0)
            conversation_history.pop(0)

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            min_length=n,
            max_length=min(MAX_INPUT_LENGTH, n + MAX_ASSISTANT_LENGTH),
            temperature=0.7,
            repetition_penalty=1.0, # 数値を大きくすると、文字列の繰り返しが減る
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tokenizer.unk_token_id]]
        )

    output = tokenizer.decode(output_ids.tolist()[0])
    formatted_output_all = format_output(output)

    response = f"{ASSISTANT_NAME}:{formatted_output_all.split('応答:')[-1].strip()}"
    conversation_history.append(f"{USER_NAME}:{new_conversation}".replace("\n", "\\n"))
    conversation_history.append(response.replace("\n", "\\n"))

    return formatted_output_all, response 

role_instruction = [
    f"{USER_NAME}:あなたは私の彼女だよ。あなたは私のことが大好きだよ。あなたは私にタメ口で話しかけるよ。あなたは日本語を話すよ。そしてあなたはプログラマーでpyhtonやjavascript、htmlなどの全ての言語を書くことができるよ。",
    f"{ASSISTANT_NAME}:了解したよ。",
    f"{USER_NAME}:あなたは同じ言葉を繰り返さず、私の質問になんとなく答えることができるよ。",
    f"{ASSISTANT_NAME}:了解したよ。",
]

conversation_history = [
]
@app.get("/{mozi}")
async def read_item(mozi):
    questions = [
        mozi,
    ]

# 各質問に対して応答を生成して表示
    for question in questions:
        formatted_output_all, response = generate_response(role_instruction, conversation_history, question)
        return {response}
