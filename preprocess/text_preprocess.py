import re

class TextPreprocessor:
    def __init__(self):
        pass

    def clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def clean_all(self, texts: list[str]) -> list[str]:
        return [self.clean(text) for text in texts]