from transformers import pipeline

def summarize_text(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer(text)[0]['summary_text']

input_text = """MS Dhoni, born on July 7, 1981, is a legendary figure in Indian cricket, celebrated for his exceptional leadership and finishing skills. He captained the Indian cricket team to victory in the 2007 T20 World Cup, the 2011 World Cup, and the 2013 ICC Champions Trophy, earning him the nickname "Captain Cool". Dhoni's calm demeanor under pressure and astute tactical decisions made him one of India's most successful captains. Beyond captaincy, he was a dynamic wicketkeeper and a powerful hitter, known for his ability to finish matches. Dhoni's impact on Indian cricket extends beyond statistics, inspiring millions with his humility and dedication. He retired from international cricket in 2020, leaving behind a legacy of remarkable achievements and a lasting influence on the sport. """
print("Original Text:\n", input_text)
print("\nSummary:\n", summarize_text(input_text))
