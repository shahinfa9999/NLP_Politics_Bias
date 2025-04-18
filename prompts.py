def bias_prompt(article_text):
    return f"""
Read the article below and assess its political bias on a scale from -1 (strongly left) to +1 (strongly right), where 0 is neutral. Provide reasoning.
Article:
{article_text}
"""

def summary_prompt(article_text):
    return f"""
Summarize the following news article in 3 sentences or less:
{article_text}
"""

def context_prompt(article_text):
    return f"""
Given the article below, provide a brief historical background relevant to the main issue discussed.
Article:
{article_text}
"""
