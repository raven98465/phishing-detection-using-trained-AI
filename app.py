import tkinter as tk
from tkinter import messagebox, Label, Frame, Text, Button, OptionMenu, StringVar
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Define local model path
model_paths = {
    'ealvaradob/bert-finetuned-phishing': 'E:/PG-dissertation/AI model/bert-finetuned-phishing',
    'dima806/phishing-email-detection': 'E:/PG-dissertation/AI model/phishing-email-detection',
    'kamikaze20/phishing-email-detection_final_2': 'E:/PG-dissertation/AI model/phishing-email-detection_final_2',
}

# Initialize global variables
classifier = None
tokenizer = None

def load_model(model_name):
    global classifier, tokenizer
    model_path = model_paths[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

def truncate_text(text, max_length=512):
    """
    Truncate the text to the max length supported by the model.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    return truncated_text

def detect_phishing():
    """
    Detect if the input email content is phishing using the loaded model.
    """
    email_content = text_box.get("1.0", tk.END).strip()
    if classifier is None:
        messagebox.showerror("Error", "Please select and load the model first")
        return
    truncated_email_content = truncate_text(email_content)
    result = classifier(truncated_email_content)
    messagebox.showinfo("Result", f"The email is classified as: {result[0]['label']}")

# Create main window
root = tk.Tk()
root.title("Phishing Email Detector")
root.geometry("600x400")  # Set window size
root.configure(bg='#add8e6')  # Set background color

# Change taskbar icon
icon_path = 'E:/PG-dissertation/phishing_detection/Bensow-Trapez-For-Adobe-Cs6-Ai.ico'
root.iconbitmap(icon_path)

# Create a frame for the model selection and buttons
frame_top = Frame(root, bg='#add8e6')
frame_top.pack(pady=10)

# Add a label for the dropdown menu
label_model = Label(frame_top, text="Select Model:", bg='#f0f0f0')
label_model.pack(side=tk.LEFT, padx=10)

# Create model selection dropdown menu
model_var = StringVar(root)
model_var.set('Select Model')  # Default value
model_menu = OptionMenu(frame_top, model_var, *model_paths.keys(), command=load_model)
model_menu.pack(side=tk.LEFT)

# Create button to start detection
detect_button = Button(frame_top, text="Detect Phishing", command=detect_phishing, bg='#4CAF50', fg='white')
detect_button.pack(side=tk.LEFT, padx=10)

# Create a frame for the text box
frame_text = Frame(root, bg='#add8e6')
frame_text.pack(pady=10)

# Add a label for the text box
label_text = Label(frame_text, text="Email Content:", bg='#ffffff')
label_text.pack(anchor="w")

# Create a text box for users to input email content
text_box = Text(frame_text, height=20, width=80)
text_box.pack()

# Running the application
root.mainloop()