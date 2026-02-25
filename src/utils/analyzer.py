import re

def simple_bias_check(text):
    # Common clickbait/bias words
    bias_words = ['shocking', 'unbelievable', 'exposed', 'conspiracy', 'secret', 'urgent', 'must see']
    found = [word for word in bias_words if word in text.lower()]
    
    # Logic: Start at 100, lose 20 points per sensationalist word
    score = 100 - (len(found) * 20)
    return max(score, 0), found

if __name__ == "__main__":
    print(simple_bias_check("Shocking secret exposed about the moon!"))
