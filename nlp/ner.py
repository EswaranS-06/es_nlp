from transformers import pipeline

class NERExtractor:
    def __init__(self, model_name):
        print(f"[NLP] Loading NER model: {model_name}")
        self.pipe = pipeline("ner", model=model_name, aggregation_strategy="simple")

    def extract(self, text):
        try:
            ents = self.pipe(text)
        except:
            return []

        return [
            {"entity": e["word"], "type": e["entity_group"], "score": float(e["score"])}
            for e in ents
        ]
