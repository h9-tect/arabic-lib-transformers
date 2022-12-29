# arabic-lib-transformers
Introduction
The Sarsor is a Python library that allows users to perform various natural language processing tasks in Arabic, such as translation, text classification, question answering, summarization, text generation, and more. It uses the transformers library to access pre-trained language models and the pyarabic library to process Arabic text.

## Dependencies
In order to use the Sarsor, you will need to install the following dependencies:

* transformers: a library for natural language processing tasks using transformer-based models.
* nltk: a library for natural language processing tasks.
* pyarabic: a library for processing Arabic text.
* torch: a library for deep learning tasks.
* csv: a library for reading and writing CSV files.
* sentencepiece: a library for text tokenization.
Installation
To install the Sarsor, you can use pip:

```sh
pip install transformers nltk pyarabic torch csv sentencepiece
```

## Usage
To use the Sarsor, you will need to import it and create an instance of the Sarsor class. You will also need to specify a model name, which can be any pre-trained language model from the transformers library.
you can use it from 
```sh
pip install -i https://test.pypi.org/simple/ Sarsor==0.1
```

```python
import Sarsor

# Create an instance of the Sarsor
arabic_library = Sarsor("bert-base-arabic")
```
### Here are some examples of how you can use the various methods in the Sarsor:

#### Translation
To translate a piece of text from English to Arabic, you can use the translate method:
```python
# Translate a piece of text from English to Arabic
arabic_text = Sarsor.translate("Hello, how are you?")
print(arabic_text)
```
The question_answering method is a function of the Sarsor class that takes in two arguments: question and context. It uses a pre-trained question answering model to find the answer to the question in the provided context. The method returns the found answer as a string.

Here is an example of how to use the question_answering method:
```python
# Initialize the Sarsor with a specified model name
arabic_processing_library = Sarsor("model_name")

# Provide the question and context
question = "ما هي عاصمة المملكة العربية السعودية؟"
context = "عاصمة المملكة العربية السعودية هي مدينة الرياض."

# Use the question_answering method to find the answer to the question in the context
answer = Sarsor.question_answering(question, context)

print(answer)  # Output: "مدينة الرياض"
```
The text-generation pipeline of the Sarsor is a method that allows you to generate text based on a provided prompt. This can be useful for tasks such as chatbots, language translation, and text summarization.

To use the text-generation pipeline, you can call the generate method of the Sarsor object and pass in the prompt as an argument. For example:
```python
library = Sarsor('bert-base-arabic')
generated_text = library.generate('أهلاً بك في العالم العربي')
print(generated_text)
```
The summarization function of the Sarsor is a method that takes in a piece of text as an input and returns a summary of the text. Here is an example of how you can use this function:

```python
# Initialize the Sarsor with a particular model
arabic_processor = Sarsor(model_name='bert-base-arabic')

# Input text
text = "عندما يتم التعامل مع النص العربي، فإن المعالجة اللغوية هي مهمة هامة في تحليل النص. وتتضمن هذه المهمة العديد من الإجراءات التي تساعد في تحليل العبارات والجمل وتحديد المعاني الخاصة بها. كما يتم استخدام معالجة اللغة العربية للتعرف على النص المراد تحليله وتصنيفه وترجمته إلى لغة أخرى."

# Generate the summary
summary = Sarsor.summarize(text)

print(summary)
```
## fine-tune
To fine-tune a transformer model on an Arabic dataset, you will need to follow the same steps as you would for any other language. Here are the steps you can follow:

Load the transformer model: You can use the BertForMaskedLM or GPT2LMHeadModel classes from the transformers library to load the pretrained BERT or GPT-2 model, respectively.
Set up the optimizer and criterion: You can use the Adam optimizer and the CrossEntropyLoss criterion from the torch library to optimize the model and compute the loss, respectively.
Train the model: In each epoch, you can iterate through the training dataset and use the model, optimizer, and criterion to train the model. You can also compute the training loss for each epoch.
Evaluate the model: In each epoch, you can also iterate through the validation dataset and use the model and criterion to evaluate the model. You can compute the validation loss and accuracy for each epoch.
Print the results: You can print the training loss, validation loss, and validation accuracy for each epoch to monitor the model's progress.
You may also need to handle the input and output text in Arabic. To do this, you will need to use an Arabic tokenizer and convert the input text to a tensor of token IDs, and the output labels to a tensor of integers. You can use the BertTokenizer class from the transformers library to tokenize the Arabic text, and the convert_tokens_to_ids and convert_ids_to_tokens methods to convert the tokens to/from token IDs.
To use the fine-tune function through the Arabic Processing Library, you will first need to create an instance of the library, passing in the name of the transformer model you want to use as an argument:
```python
library = Sarsor("bert-base-arabic")
```
Then, you can call the fine-tune function on the library instance, passing in the training and validation datasets, the number of epochs, the batch size, the learning rate, and the model type as arguments:
```python
library.fine_tune(train_dataset, val_dataset, epochs, batch_size, learning_rate, model_type)
```
For example, if you want to fine-tune a BERT model for 3 epochs with a batch size of 32 and a learning rate of 0.001, you could do the following:

```python
library.fine_tune(train_dataset, val_dataset, 3, 32, 0.001, "bert")
```
### create_csv_dataset
The create_csv_dataset function is used to create a dataset object from a CSV file. The CSV file should contain rows of text and labels, with the text in the first column and the label in the second column.

To use the create_csv_dataset function, you will need to pass in the following arguments:

* file_path: The path to the CSV file.
* tokenizer: A transformer model's tokenizer object, which can be used to tokenize the text in the CSV file.
* max_length: The maximum length of the tokenized text. Any text that exceeds this length will be truncated.
Here is an example of how to use the create_csv_dataset function:
```python
# Load a transformer model and its tokenizer
model = BertModel.from_pretrained('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Create a dataset object from a CSV file
dataset = create_csv_dataset('data.csv', tokenizer, max_length=128)

# Iterate over the dataset and print the text and label for each example
for input_tensor, label_tensor in dataset:
    print(input_tensor)
    print(label_tensor)
```
