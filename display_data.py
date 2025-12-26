import pandas as pd
import tkinter as tk
from tkinter import ttk

def display_csv(csv_path):
    """
    Displays the content of a CSV file in a resizable graphical window with scrollbars.
    :param csv_path: Path to the CSV file
    """
    # Load the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Indicate in the terminal that the viewer is running
    print("CSV Viewer is running. Close the window to exit.")

    # Create the main Tkinter window
    root = tk.Tk()
    root.title(f"CSV Viewer for {csv_path}")

    # Create a frame for the Treeview and Scrollbars
    frame = tk.Frame(root)
    frame.pack(fill='both', expand=True)

    # Add a vertical scrollbar
    vsb = ttk.Scrollbar(frame, orient="vertical")
    vsb.pack(side="right", fill="y")

    # Add a horizontal scrollbar
    hsb = ttk.Scrollbar(frame, orient="horizontal")
    hsb.pack(side="bottom", fill="x")

    # Create the Treeview widget
    tree = ttk.Treeview(frame, columns=list(df.columns), show='headings',
                        yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    # Configure the scrollbars
    vsb.config(command=tree.yview)
    hsb.config(command=tree.xview)

    # Set the headings for the Treeview
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=120)  # Adjust column width

    # Insert data into the Treeview
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    # Pack the Treeview into the frame
    tree.pack(fill='both', expand=True)

    # Run the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    # Specify the path to your CSV file
    csv_path = input("entrer le chemin vers le csv a display (data/example.csv)")  # Replace with your file path
    display_csv(csv_path)
