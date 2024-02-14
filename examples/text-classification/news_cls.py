import scipy
from optimum.ascend import AscendModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


model = AutoModelForSequenceClassification.from_pretrained(
    "uer/roberta-base-finetuned-chinanews-chinese"
)
tokenizer = AutoTokenizer.from_pretrained(
    "uer/roberta-base-finetuned-chinanews-chinese"
)
text_classification = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
texts = [
    "北京上个月召开了两会",
]
hf_pred_labels = text_classification("北京上个月召开了两会")


model_inputs = tokenizer(
    texts,
    padding="longest",
    truncation=True,
    max_length=512,
    return_tensors="np",
)

model = AscendModelForSequenceClassification.from_pretrained(
    "uer/roberta-base-finetuned-chinanews-chinese",
    export=True,
    task="text-classification",
    max_batch_size=8,
    max_sequence_length=512,
)

outputs = model(**model_inputs)
om_output = outputs["logits"]
predictions = om_output.argmax(-1)

ascend_pred_labels = [
    {"label": model.config.id2label[p], "score": scipy.special.softmax(om_output[i])[p]}
    for i, p in enumerate(predictions)
]

for hf_label, ascend_label in zip(hf_pred_labels, ascend_pred_labels):
    print(f"hf: {hf_label}, ascend: {ascend_label}")


# Save the model if you want to use it later
# model.save_pretrained("../roberta-base-finetuned-chinanews-chinese-ascend")
# model = AscendModelForFeatureExtraction.from_pretrained("../roberta-base-finetuned-chinanews-chinese-ascend")
