import torch
import csv
from transformers import pipeline
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer

class arabicTransformers:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def translation(self, text):
        translator = pipeline("translation", model=self.model_name)
        return translator(text)

    def fill_mask(self, text):
        unmasker = pipeline("fill-mask", model=self.model_name)
        return unmasker(text) 
    
    def token_classification(self, text):
        token_classifier = pipeline('ner', model=self.model_name)
        return token_classifier(text)
    
    def question_answering(self, question, context):
        question_answerer = pipeline("question-answering", model=self.model_name)
        return question_answerer(question, context)

    def summarization(self, text):
        summarizer = pipeline("summarization", model=self.model_name)
        return summarizer(text)
    
    def text_generation(self, prompt):
        generator = pipeline("text-generation", model=self.model_name)
        return generator(prompt, max_length=200)
    
    def text_classification(self, text):
        text_classifier = pipeline('text-classification', model=self.model_name) 
        return text_classifier(text)
   
    def text_similarity(self, text1, text2):
        embeddings = self.model.encode([text1, text2])
        tensors = [torch.from_numpy(e) for e in embeddings]
        score = cosine_similarity(tensors[0], tensors[1], dim=0)
        return score
        
    def sentiment_analysis(self, text):
        sentiment_analyzer = pipeline("sentiment-analysis", model=self.model_name)
        return sentiment_analyzer(text)

    def named_entity_recognition(self, text):
        ner = pipeline("ner", model=self.model_name)
        return ner(text)


    @staticmethod
    def create_csv_dataset(file_path, tokenizer, max_length):
        """Creates a dataset from a CSV file."""
        data = []

        # Read the data from the CSV file
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                text = row[0]
                label = row[1]
                data.append((text, label))

        def iterate():
            """Iterates over the data in the dataset."""
            for text, label in data:
                # Tokenize the text
                tokens = tokenizer.tokenize(text)

                # Truncate the tokens if necessary
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]

                # Convert the tokens to a tensor
                input_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

                # Convert the label to a tensor
                label_tensor = torch.tensor([int(label)])

                yield (input_tensor, label_tensor)

        def __len__():
            """Returns the length of the dataset."""
            return len(data)

       
