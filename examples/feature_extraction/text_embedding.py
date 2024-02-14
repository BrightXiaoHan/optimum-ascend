import numpy as np
import sentence_transformers
from optimum.ascend import AscendModelForFeatureExtraction
from tqdm import tqdm
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "moka-ai/m3e-base",
)
st_model = sentence_transformers.SentenceTransformer("moka-ai/m3e-base")

texts = [
    "明天不上班",
    "今天天气不错",
    "第二天放假",
    "明天上班",
] * 2
st_output = st_model.encode(texts)

model_inputs = tokenizer(
    texts,
    padding="longest",
    truncation=True,
    max_length=512,
    return_tensors="np",
)

model = AscendModelForFeatureExtraction.from_pretrained(
    "moka-ai/m3e-base",
    export=True,
    task="feature-extraction",
    max_batch_size=8,
    max_sequence_length=512,
)

outputs = model(**model_inputs)
om_output = outputs["sentence_embedding"]
cosine_similarity = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for i in range(len(texts)):
    print(
        f"Similarity between 'ascend' and 'sentence-transformers' for text {texts[i]}:"
    )
    print(cosine_similarity(st_output[i], om_output[i]))

# Benchmarking
print("Benchmarking cpu inference speed")
for i in tqdm(range(10)):
    st_model.encode(texts)

print("Benchmarking ascend inference speed")
for i in tqdm(range(100)):
    model(**model_inputs)

# Save the model if you want to use it later
# model.save_pretrained("../m3e-base-ascend")
# model = AscendModelForFeatureExtraction.from_pretrained("../m3e-base-ascend", task="feature-extraction")
