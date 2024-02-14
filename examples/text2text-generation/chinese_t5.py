from transformers import T5ForConditionalGeneration, T5Tokenizer
from optimum.ascend import AscendModelForSeq2SeqG


tokenizer = T5Tokenizer.from_pretrained("ClueAI/PromptCLUE-base")
model = T5ForConditionalGeneration.from_pretrained("ClueAI/PromptCLUE-base")


def preprocess(text):
    return text.replace("\n", "_")


def postprocess(text):
    return text.replace("_", "\n")


def answer(text, sample=False, top_p=0.8):
    text = preprocess(text)
    encoding = tokenizer(
        text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt"
    )
    if not sample:
        out = model.generate(
            **encoding,
            return_dict_in_generate=True,
            output_scores=False,
            max_length=128,
            num_beams=4,
            length_penalty=0.6
        )
    else:
        out = model.generate(
            **encoding,
            return_dict_in_generate=True,
            output_scores=False,
            max_length=64,
            do_sample=True,
            top_p=top_p
        )
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])
