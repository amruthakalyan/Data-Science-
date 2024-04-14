import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

class AutoMLApp:
    def __init__(self, master):
        self.master = master
        master.title("AutoML Application")

        self.upload_button = tk.Button(master, text="Upload Dataset", command=self.upload_file)
        self.upload_button.pack()

        self.train_button = tk.Button(master, text="Train Model", command=self.train_model)
        self.train_button.pack()

        self.quit_button = tk.Button(master, text="Quit", command=master.quit)
        self.quit_button.pack()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.target_column = self.select_target_column()
                messagebox.showinfo("Success", "Dataset uploaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to upload dataset: {str(e)}")

    def select_target_column(self):
        target_column = simpledialog.askstring("Target Column", "Enter the name of the target column:")
        return target_column

    def train_model(self):
        if hasattr(self, 'df'):
            try:
                # Handle missing values
                self.df.fillna(self.df.mean(), inplace=True)

                if self.target_column not in self.df.columns:
                    messagebox.showerror("Error", f"Target column '{self.target_column}' not found in the dataset!")
                    return

                X = self.df.drop(columns=[self.target_column])
                y = self.df[self.target_column]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestClassifier()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                messagebox.showinfo("Model Trained", f"Model trained successfully! Accuracy: {accuracy:.2f}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to train model: {str(e)}")
        else:
            messagebox.showerror("Error", "No dataset uploaded yet!")

root = tk.Tk()
app = AutoMLApp(root)
root.mainloop()
