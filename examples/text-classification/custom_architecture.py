import scipy
from optimum.ascend import AscendModelForSequenceClassification
from optimum.exporters.onnx.model_configs import BertOnnxConfig
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, pipeline


model = AutoModelForSequenceClassification.from_pretrained("Yuyi-Tech/need_follow_cls")
tokenizer = AutoTokenizer.from_pretrained("Yuyi-Tech/need_follow_cls")
text_classification = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
texts = [
    "您好，我想问一下。我在广东政务服务网网做了公司变更，但是资料写错，请问在哪里可以修改？",
]
hf_pred_labels = text_classification(texts)


model_inputs = tokenizer(
    texts,
    padding="longest",
    truncation=True,
    max_length=512,
    return_tensors="np",
)

config = AutoConfig.from_pretrained("Yuyi-Tech/need_follow_cls")
custom_onnx_config = BertOnnxConfig(
    config=config,
    task="text-classification",
)

custom_onnx_configs = {"model": custom_onnx_config}

model = AscendModelForSequenceClassification.from_pretrained(
    "Yuyi-Tech/need_follow_cls",
    custom_onnx_configs=custom_onnx_configs,
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
# model.save_pretrained("../need_follow_cls-ascend")
# model = AscendModelForFeatureExtraction.from_pretrained("../need_follow_cls-ascend")
