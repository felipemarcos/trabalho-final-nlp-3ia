import re

class TreatText:
    def removeNumbers(self, text):
        return re.sub('\s\(\d+\)', '', text)

    def lower(self, text):
        return text.lower()

    def run(self, text):
        t = self.removeNumbers(text)
        t = self.lower(t)
        return t