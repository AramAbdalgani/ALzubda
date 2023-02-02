# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import arabic_reshaper
import bidi
from bidi.algorithm import get_display

model_name = "abdalrahmanshahrour/auto-arabic-summarization"
abstractive_Tokenizer = AutoTokenizer.from_pretrained(model_name)
abstractive_Model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_text(text):
    
    return text.strip()

def reshape_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return reshaped_text

def bidi_text(text):
    bidi_text = get_display(text)
    return bidi_text

def abstractiveModelSummarizer(text):
    text = preprocess_text(text)
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

textInput ="شهدت مدينة طرابلس، مساء أمس الأربعاء، احتجاجات شعبية وأعمال شغب لليوم الثالث على التوالي، وذلك بسبب تردي الوضع المعيشي والاقتصادي. واندلعت مواجهات عنيفة وعمليات كر وفر ما بين الجيش اللبناني والمحتجين استمرت لساعات، إثر محاولة فتح الطرقات المقطوعة، ما أدى إلى إصابة العشرات من الطرفين."
out_put=abstractiveModelSummarizer(textInput)
reshaped_text_out = arabic_reshaper.reshape(out_put)

bidi_text_out = get_display(reshaped_text_out)

print(bidi_text_out)
     