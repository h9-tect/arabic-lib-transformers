import transformers
import nltk
import pyarabic
import torch
import csv
from transformers import pipeline
import sentencepiece as spm
from transformers import BertTokenizer, BertForMaskedLM

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

  return iterate, __len__



class ArabicProcessingLibrary:
    def __init__(self, model_name):
        self.model_name = model_name
        self.translate_pipeline = pipeline("translation_en_to_ar", model=model_name)
        self.fill_mask_pipeline = pipeline("fill-mask", model=model_name)
        self.token_classify_pipeline = pipeline("text-classification", model=model_name)
        self.question_answering_pipeline = pipeline("question-answering", model=model_name)
        self.summarization_pipeline = pipeline("summarization", model=model_name)
        self.chat_pipeline = pipeline("text-generation", model=model_name)
        self.generate_pipeline = pipeline("text-generation", model=model_name)
        self.classify_pipeline = pipeline("text-classification", model=model_name)

    def translate(self, text):
        translation = self.translate_pipeline(text)
        return translation[0]['translation_text']

    def fill_mask(self, text):
        completion = self.fill_mask_pipeline(text, max_length=200, num_beams=1, no_repeat_ngram_size=2, do_sample=False)
        return completion[0]['generated_text']
    
    def token_classify(self, text):
        classification = self.token_classify_pipeline(text, max_length=200)
        return classification[0]['label']
    
    def question_answering(self, question, context):
        answers = self.question_answering_pipeline(question=question, context=context)
        start_index = answers[0]["start"].argmax().item()
        end_index = answers[0]["end"].argmax().item()
        return context[start_index:end_index+1]
    
    def summarize(self, text):
        summary = self.summarization_pipeline(text, max_length=100)
        return summary[0]["summary_text"]
    
    def chat(self, prompt):
        response = self.chat_pipeline(prompt, max_length=200)
        return response[0]['generated_text']
    
    def generate(self, prompt):
        response = self.generate_pipeline(prompt, max_length=200)
        return response[0]['generated_text']
    
    def classify(self, text):
        classification = self.classify_pipeline(text, max_length=200)
        return classification[0]['label']  
    

    def fine_tune(self, train_dataset, val_dataset, epochs, batch_size, learning_rate, model_type):

    # """Fine-tunes a transformer model on the given datasets.
    
    # Args:
    #     train_dataset: A Dataset object for the training data.
    #     val_dataset: A Dataset object for the validation data.
    #     epochs: The number of epochs to train for.
    #     batch_size: The batch size to use for training.
    #     learning_rate: The learning rate to use for training.
    #     model_type: The type of transformer model to use (e.g. "bert", "gpt2").
    # """
    # Load the transformer model
      model = model_type
      if model_type == "bert":

        model = BertForMaskedLM.from_pretrained(self.model_name)
      elif model_type == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(self.model_name)
      else:
        raise ValueError("Unrecognized model type: {}".format(model_type))

    # Set up the optimizer and criterion
      optimizer = Adam(model.parameters(), lr=learning_rate)
      criterion = nn.CrossEntropyLoss()

    # Train the model
      for epoch in range(epochs):

        model.train()
        train_loss = 0
        for input_tensor, label_tensor in train_dataset:
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for input_tensor, label_tensor in val_dataset:
                output = model(input_tensor)
                val_loss += criterion(output, label_tensor).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label_tensor.view_as(pred)).sum().item()
        val_loss /= len(val_dataset)
        accuracy = correct / len(val_dataset)

        print("Epoch {}: train loss = {:.3f}, val loss = {:.3f}, val accuracy = {:.3f}".format(
            epoch, train_loss, val_loss, accuracy))


  
    def evaluate(self, dataset=None):

      if dataset is None:
        return
    # Set the model to evaluation mode
      self.model.eval()

    # Initialize the evaluation metrics
      total_loss = 0
      total_accuracy = 0
      total_examples = 0

    # Loop over the batches in the dataset
      for batch in dataset:

      # Get the input and label tensors
        input_tensor = batch[0]
        label_tensor = batch[1]

      # Forward pass
        output = self.model(input_tensor, labels=label_tensor)
        loss = output[0]

      # Update the evaluation metrics
        total_loss += loss.item()
        total_accuracy += output[1].mean().item()
        total_examples += input_tensor.size(0)

    # Calculate the average loss and accuracy
        avg_loss = total_loss / total_examples
        avg_accuracy = total_accuracy / total_examples

    # Print the evaluation metrics
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {avg_accuracy:.4f}")
        print(f"Examples: {total_examples}")


