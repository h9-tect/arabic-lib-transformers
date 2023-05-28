import torch
import csv
from transformers import pipeline
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import Trainer, TrainingArguments

class ArabicTransformers:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def translation(self, text, target_language='en', source_language='auto'):
        translator = pipeline("translation", model=self.model_name)
        return translator(text, target_lang=target_language, source_lang=source_language)
    
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

    def create_csv_dataset(self, file_path, tokenizer, max_length):
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

        return iterate, __len__

    def fine_tune(self, train_dataset, validation_dataset, optimizer="AdamW", learning_rate=2e-5, num_train_epochs=3, warmup_steps=500, weight_decay=0.01, logging_dir=None, logging_steps=500, save_strategy="epoch", save_steps=1000):
        """
        Fine-tunes the model with the provided train_dataset and validates with validation_dataset.
        
        Args:
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            optimizer (str or Optimizer): The optimizer to use during fine-tuning. Default: "AdamW".
            learning_rate (float): The learning rate for the optimizer. Default: 2e-5.
            num_train_epochs (int): The number of training epochs. Default: 3.
            warmup_steps (int): The number of warmup steps. Default: 500.
            weight_decay (float): The weight decay rate for the optimizer. Default: 0.01.
            logging_dir (str): The directory to save training logs. Default: None.
            logging_steps (int): The number of steps between each logging. Default: 500.
            save_strategy (str): The saving strategy. Default: "epoch".
            save_steps (int): The number of steps between each model save. Default: 1000.
        """
        model = SentenceTransformer(self.model_name)
        
        training_args = TrainingArguments(
            output_dir="fine_tuned_model",
            evaluation_strategy="steps",
            eval_steps=logging_steps,
            logging_dir=logging_dir,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            optimizers=optimizer
        )
        
        trainer.train()

    def handle_error(self, error):
        """
        Custom error handling logic.
        
        Args:
            error (Exception): The error or exception to handle.
        """
        # Custom error handling logic
        # You can modify this method to handle specific types of errors or exceptions
        
        # Example: Print the error message and raise the exception again
        print("An error occurred:", str(error))
        raise error

    def optimize_performance(self):
        """
        Optimizes the performance of the model.
        You can include techniques like model quantization, compression, or GPU acceleration.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch library is not installed. Please install it with `pip install torch`.")

        # Enable GPU acceleration if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
