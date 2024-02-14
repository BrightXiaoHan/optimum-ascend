<p align="center">
    <img src="readme_logo.svg" />
</p>

# optimum-ascend
Optimized inference with Ascend and Hugging Face

Note: This project is in the early development stage. Many features are still not yet refined and lack testing.

## Installation
Install optimum with onnxruntime accelerator
```bash
pip install --upgrade-strategy eager install optimum[onnxruntime]
```
Install this repo
```bash
python -m pip install git+https://github.com/SmallOneHan/optimum-ascend.git
```
Note: It is recommended to install and run this repo in the pre-built Ascend CANN container environment.

## Quick Start
Model conversion can be used through the Optimum command-line interface:
```bash
optimum-cli export ascend -m moka-ai/m3e-base ./m3e-base-ascend --task feature-extraction --soc-version "Ascend310P3"
```

Note that you need to specify the correct soc version. You can check the soc version by running the `npu-smi info` command.

To load a converted model hosted locally or on the 🤗 hub, you can do as follows :

```py
from optimum.ascend import AscendModelForFeatureExtraction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "moka-ai/m3e-base",
)
model = AscendModelForFeatureExtraction.from_pretrained("./m3e-base-ascend")
model_inputs = tokenizer(
    ["你好"],
    padding="longest",
    truncation=True,
    max_length=512,
    return_tensors="np",
)

outputs = model(**model_inputs)
om_output = outputs["sentence_embedding"]
```


## Running the examples
Check out the examples directory to see how 🤗 Optimum Ascend can be used to optimize models and accelerate inference.

Do not forget to install requirements for every example:
```bash
cd <example-folder>
pip install -r requirements.txt
```
