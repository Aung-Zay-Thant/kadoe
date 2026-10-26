<img src="assets/kadoe-logo-card.png" width="96" height="96" alt="Kadoe logo">

# Kadoe (ကတိုး)

A Burmese chat model that also reads Burmese text in images. My learning project, with much more to come.

မြန်မာလို စကားပြောပြီး ပုံထဲက မြန်မာစာကိုပါ ဖတ်တတ်တဲ့ မော်ဒယ်။ လေ့လာရင်း လုပ်နေတဲ့ ပရောဂျက်ပါ၊ နောက်ထပ် အများကြီး လာဦးမယ်။

[English](#english) · [မြန်မာစာ](#burmese)

<a id="english"></a>

**Speaks Burmese. Now it reads Burmese, too.**

I built Kadoe to learn how language models are made. It’s my learning project: I wanted to see what it takes to teach a model a language, so I picked the one I grew up with. **Kadoe 4B** is the first release, not the last. Much more is coming.

Ask it anything in everyday Burmese. Show it a photo of a sign, a menu or a receipt, and it reads the Burmese and answers your questions. It runs on your own computer: a Mac with MLX, or anything llama.cpp runs on.

Fair warning: Kadoe talks like a native Burmese speaker. Casual and direct, not the stiff, formal Burmese other AI models use. My girlfriend, its very first tester, says that makes it sound a bit rude. I say it just sounds Burmese. We’re still discussing it.

## Models

| Model | What it does | Formats | Download |
|---|---|---|---|
| **Kadoe 4B** | Burmese chat, photos, and reading Burmese text in images | BF16 · MLX 8-bit · GGUF 8-bit | [aungzaythant/Kadoe-4B](https://huggingface.co/aungzaythant/Kadoe-4B) |

Try it in your browser: [Kadoe 4B on Hugging Face Spaces](https://huggingface.co/spaces/aungzaythant/Kadoe-4B).

## What it can do

- **Chat in everyday Burmese** — explanations, advice and help with writing.
- **Read Burmese in photos** — printed pages, signs, documents, receipts and menus. Handwriting is hit-and-miss.
- **Answer questions about a photo’s text** — the total, the date, the price of the tea leaf salad.
- **Describe photos** — in Burmese or English.
- **Use tools** — give it tools in your app (a calculator, a unit converter, the date and time, Burmese number words, web search) and Kadoe decides when to call them. [Example](docs/quickstart.md#tools).
- **Still speak English** — but Burmese is its home ground.

## Quick start

Recommended settings: temperature 0.3, top-p 0.9, repetition penalty 1.05, thinking off. More options, including a local server, are in [docs/quickstart.md](docs/quickstart.md).

### Mac (MLX)

```bash
pip install -U mlx-vlm "huggingface_hub[cli]"
hf download aungzaythant/Kadoe-4B --include "mlx/*" --local-dir Kadoe-4B

python -m mlx_vlm.generate --model Kadoe-4B/mlx \
  --image receipt.jpg \
  --prompt "ဒီဘောက်ချာမှာ စုစုပေါင်း ဘယ်လောက်ကျလဲ။" \
  --max-tokens 512 --temperature 0.3 --top-p 0.9 --repetition-penalty 1.05
```

Leave out `--image` for a plain chat question. The MLX download is about 5 GB.

### llama.cpp

```bash
hf download aungzaythant/Kadoe-4B --include "gguf/*" --local-dir Kadoe-4B

llama-mtmd-cli -m Kadoe-4B/gguf/Kadoe-4B-Q8_0.gguf \
  --mmproj Kadoe-4B/gguf/mmproj-Kadoe-4B-Q8_0.gguf \
  --image receipt.jpg -p "ဒီဘောက်ချာမှာ စုစုပေါင်း ဘယ်လောက်ကျလဲ။" \
  --temp 0.3 --top-p 0.9 --repeat-penalty 1.05 --jinja --image-min-tokens 64
```

Keep `--image-min-tokens 64`: without it, small images such as a single line of text are read much less well. For a local server, see [docs/quickstart.md](docs/quickstart.md).

### Transformers (Python)

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "aungzaythant/Kadoe-4B"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="bfloat16", device_map="auto")

messages = [{"role": "user", "content": [
    {"type": "image", "path": "receipt.jpg"},
    {"type": "text", "text": "ဒီဘောက်ချာမှာ စုစုပေါင်း ဘယ်လောက်ကျလဲ။"},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True,
    return_tensors="pt", enable_thinking=False,
).to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Needs a recent `transformers` with Qwen3.5 support, plus `torch`, `torchvision`, `pillow` and `accelerate`. The sampling settings come from the model’s `generation_config.json`.

## Did I test it?

Yes, a lot: reading, photos and everyday chat. I chose not to post the numbers here. The test that matters is yours: snap a photo, ask something you really want to know, and see what Kadoe says.

## Where it still trips up

Small model, big confidence. It makes mistakes.

- It can be confidently wrong about facts, names and numbers, with a straight face. Check anything important.
- It has an imagination: now and then it describes things that aren’t in the photo.
- Handwriting and dense pages are harder; expect more errors there.

## How Kadoe came together

Every model starts somewhere. Kadoe started as Alibaba’s open [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base), a capable model that wasn’t yet at home in Burmese. Then three steps, each one a lesson for me:

1. **Learning the language — continued pretraining (CPT) on Burmese.** First, Burmese itself, until it stopped feeling foreign.
2. **Learning to chat — supervised fine-tuning (SFT).** Next, to follow what you ask and hold a real conversation, instead of rambling on.
3. **Learning to see.** Last, I gave it eyes: the vision encoder from [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), plus training to read Burmese text in images.

Chat, photos and reading. All in one model.

**Compared with Qwen3.5-4B-Base, the model it started from, Kadoe 4B understands, reads and writes Burmese far better.**

And it writes Burmese the way people actually speak it, not like a translation.

## More to come

Kadoe grows as I learn, so it doesn’t stop here. On the workbench now:

- **Listening** — Kadoe-Hear: Kadoe listening to spoken Burmese and answering, with an ear borrowed from [Raj](https://huggingface.co/aungzaythant/Raj-0.6B). In testing, not in this release.
- **Better reading** — fewer slips on handwriting and dense scanned pages.
- **More faithful photos** — describing only what’s really there.
- **Rewriting and English** — following the brief, and stronger English.

And more after that.

Caught it being wrong (or rude)? [Open an issue](https://github.com/Aung-Zay-Thant/kadoe/issues). That’s how we both learn.

## License

Apache 2.0. Use Kadoe for personal projects, learning, research, and **commercial use**. The weights on Hugging Face come with their [LICENSE](https://huggingface.co/aungzaythant/Kadoe-4B/blob/main/LICENSE) and [NOTICE](https://huggingface.co/aungzaythant/Kadoe-4B/blob/main/NOTICE); the original Qwen materials keep their [upstream license](https://huggingface.co/aungzaythant/Kadoe-4B/blob/main/UPSTREAM-LICENSE.txt). The text in this repository is under the same Apache 2.0 license ([LICENSE](LICENSE)).

## Citation

```bibtex
@misc{kadoe2026,
  author       = {Aung Zay Thant},
  title        = {Kadoe 4B: a Burmese chat model that reads Burmese text in images},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/aungzaythant/Kadoe-4B}}
}
```

---

<a id="burmese"></a>

## မြန်မာစာ

**မြန်မာလို ပြောတတ်တယ်။ အခု မြန်မာစာ ဖတ်တတ်လာပြီ။**

ကတိုးက ဘာသာစကား မော်ဒယ်တွေကို ဘယ်လိုလုပ်ရလဲ ကိုယ်တိုင် သိချင်လို့ လေ့လာရင်း လုပ်နေတဲ့ ပရောဂျက်လေးပါ။ မော်ဒယ်တစ်ခုကို ဘာသာစကားတစ်ခု သင်ပေးရတာ ဘယ်လောက်ခက်လဲ စမ်းကြည့်ချင်တော့ ငယ်ငယ်ကတည်းက ပြောလာတဲ့ မြန်မာစကားနဲ့ပဲ စလိုက်တယ်။ အဲဒီလို စမ်းရင်း ရလာတာက **ကတိုး 4B** ပါ။ ပထမဆုံး ဗားရှင်းပဲ ရှိသေးတယ်၊ နောက်ထပ် အများကြီး လာဦးမယ်။

တစ်ခုခု ရှင်းပြခိုင်းချင်လား၊ အကြံလိုလား၊ စာရေးတာ ကူစေချင်လား — နေ့တိုင်းပြောနေကျ စကားနဲ့ပဲ မေးလိုက်ပါ။ ဆိုင်းဘုတ်၊ မီနူး၊ ဘောက်ချာ ဓာတ်ပုံတစ်ပုံ ပြလိုက်ရင် အထဲက မြန်မာစာကို ဖတ်ပြီး ပြန်ဖြေပေးတယ်။ ကိုယ်ပိုင် ကွန်ပျူတာပေါ်မှာတင် သုံးလို့ရတယ် — Mac မှာ MLX နဲ့ဖြစ်ဖြစ်၊ llama.cpp သုံးလို့ရတဲ့ စက်တွေမှာဖြစ်ဖြစ်။

ကြိုပြောထားမယ်နော် — ကတိုးက မြန်မာတစ်ယောက်လိုပဲ ပြောတယ်။ တခြား AI တွေလို ရုံးစာဖတ်နေသလို တောင့်တောင့်တင်းတင်း မဟုတ်ဘဲ ပေါ့ပေါ့ပါးပါး၊ ဒဲ့ဒိုးပဲ။ ပထမဆုံး စမ်းသုံးပေးတဲ့ ကိုယ့်ရည်းစားကတော့ ကတိုး နည်းနည်း ရိုင်းတယ်တဲ့။ ကိုယ့်ဘက်ကတော့ မြန်မာဆန်တာပါလို့ ထင်တယ်။ ဒီကိစ္စ နှစ်ယောက်သား အခုထိ ငြင်းနေတုန်းပဲ။

### မော်ဒယ်များ

| မော်ဒယ် | ဘာလုပ်ပေးလဲ | ဖော်မတ် | ဒေါင်းလုဒ် |
|---|---|---|---|
| **ကတိုး 4B** | မြန်မာလို စကားပြော၊ ဓာတ်ပုံကြည့်၊ ပုံထဲက မြန်မာစာဖတ် | BF16 · MLX 8-bit · GGUF 8-bit | [aungzaythant/Kadoe-4B](https://huggingface.co/aungzaythant/Kadoe-4B) |

Browser ထဲမှာ စမ်းကြည့်ချင်ရင် — [Hugging Face Spaces မှာ ကတိုး 4B](https://huggingface.co/spaces/aungzaythant/Kadoe-4B)။

### ဘာတွေ လုပ်ပေးနိုင်လဲ

- **နေ့စဉ် မြန်မာလို စကားပြောဖော်** — ရှင်းပြခိုင်းတာ၊ အကြံတောင်းတာ၊ စာရေးကူခိုင်းတာ အကုန်ရတယ်။
- **ဓာတ်ပုံထဲက မြန်မာစာ ဖတ်ပေးတယ်** — ပုံနှိပ်စာမျက်နှာ၊ ဆိုင်းဘုတ်၊ စာရွက်စာတမ်း၊ ဘောက်ချာ၊ မီနူး။ လက်ရေးကတော့ တစ်ခါမှန် တစ်ခါလွဲ။
- **ပုံထဲက စာအကြောင်း မေးလို့ရတယ်** — စုစုပေါင်း ဘယ်လောက်လဲ၊ ရက်စွဲက ဘယ်နေ့လဲ၊ လက်ဖက်သုပ်က ဘယ်လောက်လဲ။
- **ဓာတ်ပုံထဲမှာ ဘာရှိလဲ ပြောပြတယ်** — မြန်မာလိုရော၊ အင်္ဂလိပ်လိုရော။
- **Tool တွေ သုံးတတ်တယ်** — ဂဏန်းတွက်စက်၊ ယူနစ်ပြောင်းတာ၊ ရက်စွဲနဲ့ အချိန်၊ ဂဏန်းကို မြန်မာလို စာလုံးနဲ့ ရေးပေးတာ၊ web search စတဲ့ tool တွေကို app ထဲမှာ ထည့်ပေးထားရင် ဘယ်အချိန် ခေါ်သုံးရမလဲဆိုတာ ကတိုး သူ့ဘာသာ ဆုံးဖြတ်တယ်။ [ဥပမာ](docs/quickstart.md#tools)
- **အင်္ဂလိပ်လည်း ပြောတတ်သေးတယ်** — ဒါပေမဲ့ အတော်ဆုံးကတော့ မြန်မာလိုပဲ။

ဘယ်လို သုံးရလဲဆိုတဲ့ code တွေကို အပေါ်က [Quick start](#quick-start) မှာ ကြည့်ပါ။

### စမ်းသပ်ခဲ့လား

စမ်းတာပေါ့ — စာဖတ်တာ၊ ဓာတ်ပုံကြည့်တာ၊ နေ့စဉ် စကားပြောတာ အကုန် ကိုယ်တိုင် စမ်းခဲ့တယ်။ ဒါပေမဲ့ ရမှတ်တွေကိုတော့ ဒီမှာ မတင်တော့ဘူး။ ဓာတ်ပုံတစ်ပုံယူ၊ တကယ်သိချင်တာ တစ်ခုမေးပြီး ကတိုး ဘာပြောလဲ ကြည့်လိုက်ပါ — အဲဒီလို စမ်းကြည့်တာ အကောင်းဆုံးပဲ။

### သတိထားစရာလေးတွေ

ကတိုးက မော်ဒယ်အသေးလေး ဖြစ်တော့ မှားတာလည်း ရှိတတ်ပါတယ်။

- မသိတာကိုတောင် သိသလို ပြောတတ်တယ်။ နာမည်၊ ဂဏန်း၊ အချက်အလက်တွေ မှားနိုင်လို့ အရေးကြီးတာဆို ပြန်စစ်ကြည့်ပါ။
- တစ်ခါတလေ ပုံထဲမှာ မရှိတာတွေကို ထည့်ပြောတတ်တယ်။
- လက်ရေးစာနဲ့ စာတွေ စိပ်စိပ်ရေးထားတဲ့ စာမျက်နှာတွေမှာ အမှားပိုများတယ်။

### ကတိုး ဖြစ်လာပုံ

ကတိုးကို Alibaba ရဲ့ open မော်ဒယ် [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) ကနေ စလုပ်ထားတာပါ။ တော်တဲ့ မော်ဒယ် ဖြစ်ပေမဲ့ မြန်မာလိုတော့ သိပ်မကျွမ်းသေးဘူး။ အဲဒါကို အဆင့်သုံးဆင့် ခွဲပြီး သင်ပေးခဲ့တယ်၊ အဆင့်တိုင်းမှာ ကိုယ်လည်း တစ်ခုခု ပိုသိလာတယ်။

1. **မြန်မာစကား သင်ပေးတယ် — continued pretraining (CPT)။** ပထမဆုံး မြန်မာစကားနဲ့ ရင်းနှီးကျွမ်းဝင်လာအောင် သင်ပေးတယ်။
2. **စကားပြောတတ်အောင် သင်ပေးတယ် — supervised fine-tuning (SFT)။** ပြီးတော့ ခိုင်းတာကို လိုက်လုပ်တတ်အောင်၊ လျှောက်မပြောဘဲ စကားအပြန်အလှန် ပြောတတ်အောင် သင်ပေးတယ်။
3. **မြင်တတ်အောင် လုပ်ပေးတယ်။** နောက်ဆုံးမှာ [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) ရဲ့ vision encoder ကို ယူပြီး မျက်လုံး တပ်ပေးလိုက်တယ်၊ ပြီးတော့ ပုံထဲက မြန်မာစာကို ဖတ်တတ်အောင် ထပ်သင်ပေးတယ်။

စကားပြောတာ၊ ပုံကြည့်တာ၊ စာဖတ်တာ အကုန်လုံး မော်ဒယ်တစ်ခုတည်းနဲ့ ရတယ်။

**စခဲ့တဲ့ မော်ဒယ် Qwen3.5-4B-Base နဲ့ ယှဉ်ရင် ကတိုး 4B က မြန်မာလို နားလည်တာ၊ ဖတ်တာ၊ ရေးတာ အများကြီး ပိုကောင်းပါတယ်။**

ပြီးတော့ ဘာသာပြန်ထားသလို မဟုတ်ဘဲ မြန်မာတွေ တကယ်ပြောနေကျအတိုင်း ရေးတယ်။

### နောက်ထပ် လာဦးမယ်

ကိုယ် ပိုသိလာလေ ကတိုး ပိုတော်လာလေပဲ၊ ဒါကြောင့် ဒီမှာတင် မရပ်သေးဘူး။ အခု လုပ်နေတာတွေက —

- **နားထောင်တတ်ဖို့** — မြန်မာလို ပြောတာကို နားထောင်ပြီး ပြန်ဖြေပေးမယ့် Kadoe-Hear။ နားကို [Raj](https://huggingface.co/aungzaythant/Raj-0.6B) ဆီက ငှားထားတယ်။ စမ်းသပ်ဆဲမို့ ဒီတစ်ခါ မပါသေးဘူး။
- **စာပိုကောင်းကောင်း ဖတ်ဖို့** — လက်ရေးစာနဲ့ စာစိပ်တဲ့ စာမျက်နှာတွေမှာ အမှားနည်းလာဖို့။
- **ဓာတ်ပုံကို ရှိတဲ့အတိုင်း ပြောဖို့** — မရှိတာ ထည့်မပြောဖို့။
- **စာပြင်ရေးတာနဲ့ အင်္ဂလိပ်လို ဖြေတာ** — ခိုင်းတဲ့အတိုင်း ပြင်ရေးတတ်ဖို့နဲ့ အင်္ဂလိပ်လို ပိုကောင်းကောင်း ဖြေနိုင်ဖို့။

ဒါတွေပြီးရင်လည်း လုပ်ချင်တာတွေ ရှိသေးတယ်။

မှားတာ (ဒါမှမဟုတ် ဒဲ့လွန်းတာ) တွေ့ရင် [issue ဖွင့်ပြီး](https://github.com/Aung-Zay-Thant/kadoe/issues) ပြောပါ — ကတိုးရော ကိုယ်ရော အဲဒီကနေ ဆက်သင်သွားမှာပါ။

### အသုံးပြုခွင့်

Apache 2.0 လိုင်စင်ပါ။ ကိုယ်ပိုင်ပရောဂျက်၊ လေ့လာဖို့၊ သုတေသနအတွက် သုံးလို့ရသလို **စီးပွားရေးအတွက်လည်း သုံးလို့ရတယ်။** အသေးစိတ်ကို Hugging Face ပေါ်က [LICENSE](https://huggingface.co/aungzaythant/Kadoe-4B/blob/main/LICENSE) နဲ့ [NOTICE](https://huggingface.co/aungzaythant/Kadoe-4B/blob/main/NOTICE) မှာ ကြည့်ပါ။ မူရင်း Qwen ဖိုင်တွေကတော့ သူတို့ရဲ့ [မူရင်းလိုင်စင်](https://huggingface.co/aungzaythant/Kadoe-4B/blob/main/UPSTREAM-LICENSE.txt) အတိုင်းပါ။
