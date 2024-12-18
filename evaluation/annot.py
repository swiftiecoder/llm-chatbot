import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

class AnnotationTool:
    def __init__(self, root, file_path):
        self.root = root
        self.root.title("Model Response Annotation Tool")
        
        # Load CSV file
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        
        # Validation
        required_columns = {'gemini_1.5_flash', 'gemini_2.0_flash', 'learnlm_1.5_flash', 'answer'}
        if not required_columns.issubset(self.df.columns):
            messagebox.showerror("Error", f"CSV file must contain columns: {required_columns}")
            root.destroy()
            return
        
        # Annotation results storage
        self.annotations = {model: [None] * len(self.df) for model in ['gemini_1.5_flash', 'gemini_2.0_flash', 'learnlm_1.5_flash']}
        self.current_index = 0

        # UI setup
        self.setup_ui()
        self.load_sample()
    
    def setup_ui(self):
        # Frame for navigation
        self.nav_frame = tk.Frame(self.root)
        self.nav_frame.pack(side=tk.TOP, pady=10)
        
        self.prev_btn = ttk.Button(self.nav_frame, text="Previous", command=self.previous_sample)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = ttk.Button(self.nav_frame, text="Next", command=self.next_sample)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(self.nav_frame, text="Save Annotations", command=self.save_annotations)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_label = tk.Label(self.root, text="Progress: 0%", font=("Arial", 10, "bold"))
        self.progress_label.pack(pady=5)
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=5)
        
        # Text displays
        self.true_label = tk.Label(self.root, text="True Answer:", font=("Arial", 12, "bold"), fg="green")
        self.true_label.pack(pady=5)
        self.true_text = tk.Text(self.root, height=2, wrap=tk.WORD, state=tk.DISABLED)
        self.true_text.pack(fill=tk.X, padx=10, pady=5)
        
        self.models_frame = tk.Frame(self.root)
        self.models_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        # Text display for the question
        self.question_label = tk.Label(self.root, text="Question:", font=("Arial", 12, "bold"), fg="blue")
        self.question_label.pack(pady=5)
        self.question_text = tk.Text(self.root, height=2, wrap=tk.WORD, state=tk.DISABLED)
        self.question_text.pack(fill=tk.X, padx=10, pady=5)
        
        # Columns for each model response
        self.model_responses = {}
        for model in ['gemini_1.5_flash', 'gemini_2.0_flash', 'learnlm_1.5_flash']:
            label = tk.Label(self.models_frame, text=f"{model} Response:", font=("Arial", 10, "bold"))
            label.pack(pady=2)
            text_box = tk.Text(self.models_frame, height=3, wrap=tk.WORD, state=tk.DISABLED)
            text_box.pack(fill=tk.X, pady=2)
            self.model_responses[model] = text_box
        
        # Annotation buttons
        self.annotation_frame = tk.Frame(self.root)
        self.annotation_frame.pack(pady=5)
        
        self.annotation_vars = {}
        options = ["Correct", "Partially Correct", "Incorrect"]
        for model in ['gemini_1.5_flash', 'gemini_2.0_flash', 'learnlm_1.5_flash']:
            model_frame = tk.Frame(self.annotation_frame)
            model_frame.pack(side=tk.LEFT, padx=10)
            model_label = tk.Label(model_frame, text=f"{model} Annotation:", font=("Arial", 10, "bold"))
            model_label.pack(anchor=tk.W)
            self.annotation_vars[model] = tk.StringVar(value="")
            for option in options:
                btn = ttk.Radiobutton(model_frame, text=option, variable=self.annotation_vars[model], value=option, command=self.save_current_annotation)
                btn.pack(anchor=tk.W)
    
    def load_sample(self):
        # Clear previous state
        self.clear_text_widgets()
        
        # Load data for current index
        row = self.df.iloc[self.current_index]

        # Display question
        self.question_text.config(state=tk.NORMAL)
        self.question_text.delete(1.0, tk.END)
        self.question_text.insert(tk.END, row['question'])
        self.question_text.config(state=tk.DISABLED)

        self.true_text.config(state=tk.NORMAL)
        self.true_text.delete(1.0, tk.END)
        self.true_text.insert(tk.END, row['answer'])
        self.true_text.tag_add("green", "1.0", "end")
        self.true_text.tag_config("green", foreground="green")
        self.true_text.config(state=tk.DISABLED)
        
        for model, text_box in self.model_responses.items():
            text_box.config(state=tk.NORMAL)
            text_box.delete(1.0, tk.END)
            text_box.insert(tk.END, row[model])
            text_box.config(state=tk.DISABLED)
        
        # Load existing annotations if available
        for model in self.annotation_vars:
            if self.annotations[model][self.current_index] is not None:
                self.annotation_vars[model].set(self.annotations[model][self.current_index])
            else:
                self.annotation_vars[model].set("")
        
        # Update progress bar
        self.update_progress_bar()
        
    def clear_text_widgets(self):
        self.true_text.config(state=tk.NORMAL)
        self.true_text.delete(1.0, tk.END)
        self.true_text.config(state=tk.DISABLED)
        
        for text_box in self.model_responses.values():
            text_box.config(state=tk.NORMAL)
            text_box.delete(1.0, tk.END)
            text_box.config(state=tk.DISABLED)
        
    def save_current_annotation(self):
        for model in self.annotation_vars:
            self.annotations[model][self.current_index] = self.annotation_vars[model].get()
        
    def next_sample(self):
        if self.current_index < len(self.df) - 1:
            self.save_current_annotation()
            self.current_index += 1
            self.load_sample()
    
    def previous_sample(self):
        if self.current_index > 0:
            self.save_current_annotation()
            self.current_index -= 1
            self.load_sample()
    
    def save_annotations(self):
        self.save_current_annotation()
        for model in self.annotations:
            self.df[f'{model}_annotation'] = self.annotations[model]
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            self.df.to_csv(save_path, index=False)
            messagebox.showinfo("Saved", f"Annotations saved to {save_path}")
    
    def update_progress_bar(self):
        # Calculate progress percentage
        total = len(self.df)
        progress = (self.current_index + 1) / total * 100
        self.progress_bar["value"] = progress
        self.progress_label.config(text=f"Progress: {int(progress)}%")

if __name__ == "__main__":
    # file_path = 'harry-potter-trivia-ai-100-results.csv'
    file_path = 'harry-potter-trivia-ai-100-results-clustering.csv'
    root = tk.Tk()
    app = AnnotationTool(root, file_path)
    root.mainloop()