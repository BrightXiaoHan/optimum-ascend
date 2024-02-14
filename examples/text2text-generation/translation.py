# Use a pipeline as a high-level helper
from transformers import pipeline


pipe = pipeline("translation", model="facebook/nllb-200-distilled-600M")


pipe("Hello, my name is Sylvain.", src="en", tgt="fr")
