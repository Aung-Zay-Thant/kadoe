# Kadoe 4B quick start

Kadoe 4B has no thinking mode. Use these settings everywhere: **temperature 0.3, top-p 0.9, repetition penalty 1.05,
thinking off.** The model’s `generation_config.json` carries the sampling settings, and its chat template already
answers without a thinking block by default.

Example image: any photo with Burmese text in it, such as a receipt, a menu, a sign or a printed page.

## Mac (MLX)

Needs an Apple silicon Mac. The MLX download is about 5 GB.

```bash
pip install -U mlx-vlm "huggingface_hub[cli]"
hf download aungzaythant/Kadoe-4B --include "mlx/*" --local-dir Kadoe-4B
```

Read a photo:

```bash
python -m mlx_vlm.generate --model Kadoe-4B/mlx \
  --image receipt.jpg \
  --prompt "ဒီပုံထဲက စာတွေကို ဖတ်ပေးပါ။" \
  --max-tokens 1024 --temperature 0.3 --top-p 0.9 --repetition-penalty 1.05
```

Ask a question without a photo:

```bash
python -m mlx_vlm.generate --model Kadoe-4B/mlx \
  --prompt "မိုးကြိုး ဘာကြောင့် ပစ်တာလဲ။ ရိုးရိုးရှင်းရှင်း ရှင်းပြပေးပါ။" \
  --max-tokens 1024 --temperature 0.3 --top-p 0.9 --repetition-penalty 1.05
```

## llama.cpp

Tested with llama.cpp b11160 (commit `70c4e15`). The text model and the vision file (`mmproj`) are both needed for photos.

```bash
hf download aungzaythant/Kadoe-4B --include "gguf/*" --local-dir Kadoe-4B
```

One question from the command line:

```bash
llama-mtmd-cli -m Kadoe-4B/gguf/Kadoe-4B-Q8_0.gguf \
  --mmproj Kadoe-4B/gguf/mmproj-Kadoe-4B-Q8_0.gguf \
  --image receipt.jpg -p "ဒီဘောက်ချာမှာ စုစုပေါင်း ဘယ်လောက်ကျလဲ။" \
  --temp 0.3 --top-p 0.9 --repeat-penalty 1.05 --jinja --image-min-tokens 64
```

A local server with an OpenAI-compatible API:

```bash
llama-server -m Kadoe-4B/gguf/Kadoe-4B-Q8_0.gguf \
  --mmproj Kadoe-4B/gguf/mmproj-Kadoe-4B-Q8_0.gguf \
  --temp 0.3 --top-p 0.9 --repeat-penalty 1.05 --reasoning off --port 8080 --image-min-tokens 64 -c 32768
```

What the extra flags do:

- `--image-min-tokens 64` gives every image enough detail. Without it, small images, such as a single line of text
  or a small crop, are read much less well. Keep it in both commands.
- `--jinja` makes `llama-mtmd-cli` use the model's own chat template, which answers without a thinking block.
- `-c 32768` sets a 32k-token context, which keeps memory use modest. Raise it if you need longer conversations.

```python
import base64, requests

image = base64.b64encode(open("receipt.jpg", "rb").read()).decode()
reply = requests.post("http://127.0.0.1:8080/v1/chat/completions", json={
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
        {"type": "text", "text": "ဒီဘောက်ချာမှာ စုစုပေါင်း ဘယ်လောက်ကျလဲ။"},
    ]}],
    "temperature": 0.3, "top_p": 0.9,
})
print(reply.json()["choices"][0]["message"]["content"])
```

## Transformers (Python)

Needs a recent `transformers` with Qwen3.5 support, plus `torch`, `torchvision`, `pillow` and `accelerate`. The BF16
weights are about 9 GB, so the GPU needs more memory than that.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "aungzaythant/Kadoe-4B"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="bfloat16", device_map="auto")

def ask(text, image=None):
    content = ([{"type": "image", "path": image}] if image else []) + [{"type": "text", "text": text}]
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=1024)
    return processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(ask("ဒီပုံထဲက စာတွေကို ဖတ်ပေးပါ။", image="page.jpg"))
print(ask("မိုးကြိုး ဘာကြောင့် ပစ်တာလဲ။ ရိုးရိုးရှင်းရှင်း ရှင်းပြပေးပါ။"))
```

## Tools

Kadoe can call tools that your app gives it: a calculator, a unit converter, the date and time, Burmese number
words, web search, or anything else you can describe with a JSON schema. Pass them to the chat template as
`tools=[...]`, and Kadoe decides when to call one. It writes standard Qwen-style tool calls. This reuses `processor`
and `model` from the Transformers example above.

```python
tokenizer = processor.tokenizer

tools = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Exact arithmetic. Use it for any sum, price or percentage.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "For example 3*1500 + 2*1200"}},
            "required": ["expression"],
        },
    },
}]
messages = [{"role": "user", "content": "လက်ဖက်သုပ် ၃ ပွဲ တစ်ပွဲ ၁၅၀၀ နဲ့ လက်ဖက်ရည် ၂ ခွက် တစ်ခွက် ၁၂၀၀ ဆို ဘယ်လောက်ကျလဲ။"}]

inputs = tokenizer.apply_chat_template(
    messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

When Kadoe wants a tool, its answer is a call block instead of text, like this:

```text
<tool_call>
<function=calculator>
<parameter=expression>
3*1500 + 2*1200
</parameter>
</function>
</tool_call>
```

Run the tool in your own code, add the call and its result to the conversation, and generate again. Kadoe then
answers from the result:

```python
messages += [
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "calculator", "arguments": {"expression": "3*1500 + 2*1200"}}},
    ]},
    {"role": "tool", "content": '{"result": 6900}'},
]
```

Only offer tools you trust: Kadoe chooses the call and its arguments, and your code runs it.

## Tips

- **Repetition.** A 4B model can occasionally repeat a passage. If you build an app, stop the answer when the same
  passage comes back two or three times in a row.
- **Long pages.** Dense pages are read less reliably than short texts.
- **Check what matters.** Prices, dates, names and phone numbers read from a photo can be wrong.
