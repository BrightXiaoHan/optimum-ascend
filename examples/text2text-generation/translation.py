from optimum.ascend import AscendModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
translator = pipeline(
    "translation",
    model=AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh"),
    tokenizer=tokenizer,
    max_length=128,
)

en_texts = [
    "Hello, my name is Sylvain.",
    "I am a student, and I am learning to program.",
]

hf_translations = translator(en_texts)

model = AscendModelForSeq2SeqLM.from_pretrained(
    "Helsinki-NLP/opus-mt-en-zh",
    export=True,
    task="text2text-generation-with-past",
    soc_version="Ascend910B3",  # replace with your own SoC version
    max_batch_size=16,  # for text2text-generation-with-past, 'batch_size * num_beams' should be less than or equal to 'max_batch_size'
    max_sequence_length=128,
    max_output_sequence_length=128,
)


inputs = tokenizer(
    en_texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
)
outputs = model.generate(**inputs)
ascend_translations = tokenizer.decode(outputs[0], skip_special_tokens=True)

for ht, at, orig in zip(hf_translations, ascend_translations, en_texts):
    print(f"Original: {orig}")
    print(f"Hugging Face: {ht['translation_text']}")
    print(f"Ascend: {at}")
    print()
