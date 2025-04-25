# Political Bias Classification & Summary Toolkit

This project provides tools for detecting political bias in news articles, generating concise summaries, and analyzing historical contextual content using NLP techniques. It uses LDA topic modeling, transformer-based embeddings ( RoBERTa), and machine learning classifiers such as RandomForest, and XGBoost to detect bias, and an openAI API to provide historical context and a summary.

## Project Structure

### Bias_Model_Generation.py
- Trains the bias classification model.
- Uses LDA, RoBERTa embeddings, and various classifiers.
- Performs data balancing, lemmatization, and preprocessing.

### model_Usage.py
- Loads the trained model and applies it to new articles.
- Supports evaluation, summarizes, and provides a historical background.
- Acts as the main function.

### prompts.py
- Contains prompt functions for summarization and context generation.
- Intended for use with OpenAI/Gemini APIs or other LLMs.

## Setup Instructions

### 1. Create and activate a virtual environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 2.  Install dependencies
- Use Python 3.10 to be able to use tesnor flow and sickit-learn
- pip install -r requirements.txt
- Major depedencies:
    -scikit-learn
    -tensor
    -transformers
    -openai 

### 3. Data set downloads (optional to regenerate the models)
- CNN/DailyMail: https://huggingface.co/datasets/cnn_dailymail or CNN/DailyMail: `load_dataset("cnn_dailymail", "3.0.0")`
- MBIC Dataset: https://www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset
- Allsides : https://www.kaggle.com/datasets/supratimhaldar/allsides-ratings-of-bias-in-electronic-media or https://github.com/irgroup/Qbias/blob/main/allsides_balanced_news_headlines-texts.csv

### 4. Output
Bias_Model_Generation.py: Accuracy, precision, recall, and F1 scores for bias classification when training a model, and saves the model.

model_Usage.py generates a summary using predefined prompts (need an openAI API key) and predicts political bias using the model trained in Bias_Model_Generation.py. It outputs the predicted bias label along with a confidence score.

#### Example Output and Model Comparison:

Given the following Fox News article : https://www.foxnews.com/politics/white-house-rips-alleged-pentagon-leakers-shattered-egos-brushes-off-hegseth-second-signal-chat-report, each model provided a bias prediction along with a confidence score:
- Random Forest: LEFT (confidence: 0.54)
- XGBoost: RIGHT (confidence: 0.75)
- Neural Network: RIGHT (confidence: 0.70)

##### Summarized Output:

Amid escalating tensions at the Pentagon, the White House defended Secretary of Defense Pete Hegseth after reports of a second Signal group chat discussing military strikes in Yemen. The administration dismissed the allegations as non-stories propagated by recently fired staffers. Despite substantial firings, including top advisors, the administration insisted no classified information was shared. The reports originated from a group chat involving high-ranking officials, which mistakenly included a journalist. This controversy adds to the ongoing chaos and scrutiny within the Pentagon.

##### Historical Context:

The conflict in Yemen began in earnest in 2014 when Houthi rebels, backed by Iran, seized control of the capital, Sanaa... The United States has been involved in this conflict primarily through security assistance to its allies and counterterrorism operations, amidst concerns over terrorism and regional stability...

#### Interpretation:

This example highlights model behavior on real-world content:

- Neural Network (NNN) correctly identifies the article as right-leaning, consistent with Fox News' well-known bias.

- XGBoost also performs well, matching the label with slightly higher confidence than the NNN.

- Random Forest, in contrast, misclassifies the article as left-leaning with low confidence.

Why the Neural Network Stands Out:
- Accuracy: Its prediction aligns with the known bias of the article's source.
- Confidence: The confidence level (0.70) is strong, especially compared to Random Forest’s 0.54.
- Consistency: Early testing suggests the NNN performs reliably across various sources, making it a strong candidate for primary use or ensemble weighting.


### 5. Example Usage
 
 -  Example Usage: This step has already done but provided for user-specifc fine tuning 
 ``` bash
 python3.10 Bias_Model_Generation.py
 ```
 -Simply copy a news article and paste it into the "trial_article.txt" file and run:
 ``` bash
 python model_Usage.py
 ```

### 6. Future Work
- Improve centrist bias classification.

- Deploy as an API or user-facing app.

## License
MIT License

## Contributions
Feel free to fork and submit pull requests.
