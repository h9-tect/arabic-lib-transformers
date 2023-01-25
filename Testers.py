from arabicTransformers import *

print("# Translation from English to Arabic:")
model = arabicTransformers("anibahug/marian-finetuned-kde4-en-to-ar") 
translated_text = arabicTransformers.translation(model, "Hello, how are you?") #You need to call the model (the Class Constructor).
print(translated_text, '\n')

print("# Translation from Arabic to English:")
model = arabicTransformers("Patt/fine-tuned_ar-en") 
translated_text = arabicTransformers.translation(model, "أهلا، كيف حالك؟") #You need to call the model (the Class Constructor).
print(translated_text, '\n\n')

print("# English Masked Language Modeling:")
model = arabicTransformers("xlm-roberta-large")
masked_text = arabicTransformers.fill_mask(model, "Hello, I'm a <mask> model.") #You need to call the model (the Class Constructor).
print(masked_text, '\n')

print("# Arabic Masked Language Modeling:")
model = arabicTransformers("CAMeL-Lab/bert-base-arabic-camelbert-mix")
masked_text = arabicTransformers.fill_mask(model, "اللغة العربية هي لغة [MASK].") #You need to call the model (the Class Constructor).
print(masked_text, '\n\n')

print("# English Token Classification (Named Entity Recognition):")
model = arabicTransformers('Jean-Baptiste/roberta-large-ner-english')
ner_text = arabicTransformers.token_classification(model, "My name is Ali and I live in Dubai") #You need to call the model (the Class Constructor).
print(ner_text, '\n')

print("# Arabic Token Classification (Part-of-Speech Tagging):")
model = arabicTransformers('CAMeL-Lab/bert-base-arabic-camelbert-mix-pos-msa')
pos_text = arabicTransformers.token_classification(model, "إمارة أبوظبي هي إحدى إمارات دولة الإمارات العربية المتحدة السبع") #You need to call the model (the Class Constructor).
print(pos_text, '\n\n')

print("# English Question Answering:")
model = arabicTransformers('distilbert-base-cased-distilled-squad')
question="Where do I live?"
context = "My name is Ali and I live in London"
result = arabicTransformers.question_answering(model, question=question, context=context) #You need to call the model (the Class Constructor).
print(result, '\n')

print("# Arabic Question Answering:")
model = arabicTransformers('ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA')
question="ما اسمي؟"
context = "اسمي سعيد وأسكن في الرياض."
result = arabicTransformers.question_answering(model, question=question, context=context) #You need to call the model (the Class Constructor).
print(result, '\n\n')

print("# English Summarization:")
model = arabicTransformers("philschmid/bart-large-cnn-samsum")
text = '''The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, 
and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. 
During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest 
man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York 
City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition 
of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). 
Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.                               
'''
summarized_text = arabicTransformers.summarization(model, text=text) #You need to call the model (the Class Constructor).
print(summarized_text, '\n')

print("# Arabic Summarization:")
model = arabicTransformers("abdalrahmanshahrour/arabartsummarization")
text = '''شهدت مدينة طرابلس، مساء أمس الأربعاء،
احتجاجات شعبية وأعمال شغب لليوم الثالث على التوالي، وذلك بسبب تردي الوضع المعيشي والاقتصادي.
واندلعت مواجهات عنيفة وعمليات كر وفر ما بين الجيش اللبناني والمحتجين استمرت لساعات،
إثر محاولة فتح الطرقات المقطوعة، ما أدى إلى إصابة العشرات من الطرفين.                   
'''
summarized_text = arabicTransformers.summarization(model, text=text) #You need to call the model (the Class Constructor).
print(summarized_text, '\n\n')

print("# English Text Classification (Sentiment Analysis):")
model = arabicTransformers('distilbert-base-uncased-finetuned-sst-2-english') 
text_classified = arabicTransformers.text_classification(model, "I like you. I love you.") #You need to call the model (the Class Constructor).
print(text_classified, '\n')

print("# Arabic Text Classification (Arabic Dialect Identification):")
model = arabicTransformers('Ammar-alhaj-ali/arabic-MARBERT-dialect-identification-city') 
text_classified = arabicTransformers.text_classification(model,"كل عام وانت طيب يا باشا") #You need to call the model (the Class Constructor).
print(text_classified, '\n\n')

print("# English Text Generation:")
model = arabicTransformers('gpt2') 
text_generated = arabicTransformers.text_generation(model, "My name is Julien and I like to") #You need to call the model (the Class Constructor).
print(text_generated, '\n')

print("# Arabic Text Generation:")
model = arabicTransformers('aubmindlab/aragpt2-base') 
text_generated = arabicTransformers.text_generation(model, "القدس مدينة تاريخية، بناها الكنعانيون في") #You need to call the model (the Class Constructor).
print(text_generated, '\n\n')

print("# Arabic Text similarity:")

at = arabicTransformers("symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli")
text1 = "أحب مصر في الصيف"
text2 = "أحب الرياض في  الشتاء"
score = at.text_similarity(text1, text2)
print("Similarity Score:", score)
