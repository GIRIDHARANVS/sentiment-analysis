import gradio as gr
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})"

# Custom CSS for styling
custom_css = """
#interface-container {
    background-color: #f0f4f8;
    font-family: Arial, sans-serif;
}
#title {
    color: #2c3e50;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}
#description {
    color: #7f8c8d;
    font-size: 16px;
    text-align: center;
    margin-bottom: 40px;
}
#input-box {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 15px;
    font-size: 16px;
    border: 2px solid #ccd1d9;
}
#output-box {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 20px;
    font-size: 16px;
    color: #16a085;
    font-weight: bold;
    border: 2px solid #16a085;
    margin-top: 10px;
}
#submit-button {
    background-color: #16a085;
    color: white;
    border: none;
    padding: 12px 25px;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
#submit-button:hover {
    background-color: #1abc9c;
}
"""

# Create Gradio Interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter Text", placeholder="Type your sentence here...", elem_id="input-box"),
    outputs=gr.Text(label="Sentiment Analysis Result", elem_id="output-box"),
    title="Sentiment Analysis API",
    description="🔍 Enter a sentence, and the model will predict if it's POSITIVE or NEGATIVE.",
    theme="compact",
    css=custom_css
)

# Launch the interface
iface.launch()
