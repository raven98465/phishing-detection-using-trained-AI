import tkinter as tk
from tkinter import messagebox, Label, Frame, Text, Button, OptionMenu, StringVar
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import os

# Retrieve the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define local model paths (Use dynamic path)
model_paths = {
    'ealvaradob/bert-finetuned-phishing': os.path.join(script_dir, 'AI model', 'bert-finetuned-phishing'),
    'dima806/phishing-email-detection': os.path.join(script_dir, 'AI model', 'phishing-email-detection'),
    'kamikaze20/phishing-email-detection_final_2': os.path.join(script_dir, 'AI model', 'phishing-email-detection_final_2'),
    'aibot123/phishing-detection-body-only': os.path.join(script_dir, 'results', 'model_body_only'),
    'huynq3Cyradar/bert-large-finetuned-phishing-url-version': os.path.join(script_dir, 'AI model', 'bert-large-finetuned-phishing-url-version')
}

# Initialize global variables
classifier = None
tokenizer = None
url_classifier = None
url_tokenizer = None
url_model_name = 'huynq3Cyradar/bert-large-finetuned-phishing-url-version'
current_model_name = ''

def load_model(model_name):
    global classifier, tokenizer, url_classifier, url_tokenizer, current_model_name
    model_path = model_paths[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
    current_model_name = model_name

    # Load URL model separately
    url_model_path = model_paths[url_model_name]
    url_tokenizer = AutoTokenizer.from_pretrained(url_model_path)
    url_model = AutoModelForSequenceClassification.from_pretrained(url_model_path)
    url_classifier = pipeline('text-classification', model=url_model, tokenizer=url_tokenizer)

    # Update URL model label
    url_model_label.config(text=f"URL Model: {url_model_name}")

def truncate_text(text, max_length=512):
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    return truncated_text

def extract_urls(text):
    url_pattern = re.compile(r'(https?://\S+)')
    urls = url_pattern.findall(text)
    return urls

def detect_phishing_urls(urls):
    results = []
    for url in urls:
        result = url_classifier(url)
        results.append((url, result[0]['label']))
    return results

def detect_phishing():
    email_content = text_box.get("1.0", tk.END).strip()
    if classifier is None or url_classifier is None:
        messagebox.showerror("Error", "Please select and load the models first")
        return

    truncated_email_content = truncate_text(email_content)
    email_result = classifier(truncated_email_content)

    urls = extract_urls(email_content)
    if urls:
        url_results = detect_phishing_urls(urls)
        url_message = "\n".join([f"URL: {url}, Classification: {label}" for url, label in url_results])
    else:
        url_results = "No URLs found"
        url_message = url_results

    messagebox.showinfo("Result", f"Email classified as: {email_result[0]['label']}\nUsing email model: {current_model_name}\n\n"
                                  f"URL analysis:\n{url_message}\n\nURLs detected using model: {url_model_name}")

# Create main window
root = tk.Tk()
root.title("Phishing Email and URL Detector")
root.geometry("700x700")
root.configure(bg='#add8e6')

# Change taskbar icon
icon_path = os.path.join(script_dir, 'Bensow-Trapez-For-Adobe-Cs6-Ai.ico')
root.iconbitmap(icon_path)

# Create a frame for the model selection and buttons
frame_top = Frame(root, bg='#add8e6')
frame_top.pack(pady=10)

label_model = Label(frame_top, text="Select Model:", bg='#f0f0f0')
label_model.pack(side=tk.LEFT, padx=10)

model_var = StringVar(root)
model_var.set('Select Model')
model_menu = OptionMenu(frame_top, model_var, *(key for key in model_paths.keys() if key != 'huynq3Cyradar/bert-large-finetuned-phishing-url-version'), command=load_model)
model_menu.pack(side=tk.LEFT)

detect_button = Button(frame_top, text="Detect Phishing", command=detect_phishing, bg='#4CAF50', fg='white')
detect_button.pack(side=tk.LEFT, padx=10)

frame_url_model = Frame(root, bg='#add8e6')
frame_url_model.pack(pady=5)

url_model_label = Label(frame_url_model, text=f"URL Model: {url_model_name}", bg='#add8e6')
url_model_label.pack()

frame_text = Frame(root, bg='#add8e6')
frame_text.pack(pady=10)

label_text = Label(frame_text, text="Email Content:", bg='#ffffff')
label_text.pack(anchor="w")

text_box = Text(frame_text, height=40, width=80)
text_box.pack()

root.mainloop()