# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import arabic_reshaper
import bidi
from bidi.algorithm import get_display


model_name = "abdalrahmanshahrour/auto-arabic-summarization"
abstractive_Tokenizer = AutoTokenizer.from_pretrained(model_name)
abstractive_Model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def abstractiveModelSummarizer(text):
 input_ids = abstractive_Tokenizer(
    [(text)],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=1024)["input_ids"]

 output_ids = abstractive_Model.generate(
      input_ids=input_ids,
      max_length=600,
      no_repeat_ngram_size=8,
      num_beams=4)[0]

 summary = abstractive_Tokenizer.decode(
      output_ids,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=False)
 return summary

textInput ="شهد التاريخ الأوروبي في أواخر القرون الوسطى أحداثا جساما أدَّت إلى تغييرات عميقة في الوعي الثقافي والفكري والسياسي لهذه القارة، وكان الحدث الأشد في إيقاظ هذا الوعي هو حضور الجيران المسلمين سواء من الأندلس أو صقلية أو جنوب المتوسط أو من الأناضول وشرق أوروبا. فقد أحدث هؤلاء المسلمون جميعا من العرب والأتراك والبربر في عصور الحروب والحملات الصليبية، وفي الأزمنة المملوكية والعثمانية، تأثيرا كبيرا في الوعي الأوروبي، لكن العلاقة بين الكنيسة الكاثوليكية ورعاياها الأوروبيين كانت تشهد هي الأخرى تجاذبات هي الأكثر ثورية في تاريخ العلاقة بين الإكليروس وعموم الناس. لقد سيطرت الكنيسة الكاثوليكية في روما على البشر والحجر طوال أكثر من ثلاثة عشر قرنا من الزمان، وفي ذلك يقول المؤرخ القس أندرو ميلر في كتابه مختصر تاريخ الكنيسة إنه حتى القرن السادس عشر الميلادي لم يكن هناك مخلوق مستقل عن الكاهن، بل كان الكاهن هو سيد كل صغير وكبير، وكان له مطلق السلطان على الجسد والنفس، على الدهر الحاضر والأبدية، لم يكن في مقدور أحد التعرض لغضبه أو الوقوف أمام توبيخه، فالحرمان كان يقطع الكل في الحال مهما كانت رتبته أو مقامه، ويطوح به بعيدا عن حظيرة الكنيسة، التي خارج حدودها لا يوجد أقل أمل في الخلاص"
out_put=abstractiveModelSummarizer(textInput)
#reshaped_text_out = arabic_reshaper.reshape(out_put)

#bidi_text_out = get_display(reshaped_text_out)

#print(bidi_text_out)
print(out_put)

     
     


    
    