# arabic-lib-transformers
Introduction
The arabicTransformers is a Python library that allows users to perform various natural language processing tasks in Arabic, such as translation, text classification, question answering, summarization, text generation, and more. It uses the transformers library to access pre-trained language models and the pyarabic library to process Arabic text.

## Dependencies
In order to use the arabicTransformers, you will need to install the following dependencies:

* transformers: a library for natural language processing tasks using transformer-based models.
* nltk: a library for natural language processing tasks.
* pyarabic: a library for processing Arabic text.
* torch: a library for deep learning tasks.
* csv: a library for reading and writing CSV files.
* sentencepiece: a library for text tokenization.
Installation
To install the arabicTransformers, you can use pip:

```sh
pip install transformers nltk pyarabic torch csv sentencepiece
```

## Usage
To use the Sarsor, you will need to import it and create an instance of the arabicTransformers class. You will also need to specify a model name, which can be any pre-trained language model from the transformers library.
you can use it from 


```python
import arabicTransformers

# Create an instance of the arabicTransformers
arabic_library = arabicTransformers("bert-base-arabic")
```
### Here are some examples of how you can use the various methods in the arabicTransformers:

#### Translation
To translate a piece of text from English to Arabic, you can use the translate method:
```python
# Translate a piece of text from English to Arabic
arabic_text = arabicTransformers.translate("Hello, how are you?")
print(arabic_text)
```
The question_answering method is a function of the arabicTransformers class that takes in two arguments: question and context. It uses a pre-trained question answering model to find the answer to the question in the provided context. The method returns the found answer as a string.

Here is an example of how to use the question_answering method:
```python
# Initialize the arabicTransformers with a specified model name
arabic_processing_library = arabicTransformers("model_name")

# Provide the question and context
question = "ما هي عاصمة المملكة العربية السعودية؟"
context = "عاصمة المملكة العربية السعودية هي مدينة الرياض."

# Use the question_answering method to find the answer to the question in the context
answer = arabicTransformers.question_answering(question, context)

print(answer)  # Output: "مدينة الرياض"
```
The text-generation pipeline of the arabicTransformers is a method that allows you to generate text based on a provided prompt. This can be useful for tasks such as chatbots, language translation, and text summarization.

To use the text-generation pipeline, you can call the generate method of the arabicTransformers object and pass in the prompt as an argument. For example:
```python
library = arabicTransformers('bert-base-arabic')
generated_text = library.generate('أهلاً بك في العالم العربي')
print(generated_text)
```
The summarization function of the arabicTransformers is a method that takes in a piece of text as an input and returns a summary of the text. Here is an example of how you can use this function:

```python
# Initialize the arabicTransformers with a particular model
arabic_processor = arabicTransformers(model_name='bert-base-arabic')

# Input text
text = "عندما يتم التعامل مع النص العربي، فإن المعالجة اللغوية هي مهمة هامة في تحليل النص. وتتضمن هذه المهمة العديد من الإجراءات التي تساعد في تحليل العبارات والجمل وتحديد المعاني الخاصة بها. كما يتم استخدام معالجة اللغة العربية للتعرف على النص المراد تحليله وتصنيفه وترجمته إلى لغة أخرى."

# Generate the summary
summary = arabicTransformers.summarize(text)

print(summary)
```
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
