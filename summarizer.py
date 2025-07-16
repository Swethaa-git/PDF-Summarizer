from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def summarize_text(text):
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    max_input_length = 1024
    text = text[:max_input_length]

    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    return summary
