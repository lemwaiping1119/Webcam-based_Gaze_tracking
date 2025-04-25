from tkinter import messagebox
import os

def delete_all_files():
    """Delete all files in the calibration results folder."""
    folder_path = r"gaze_estimation/calibration_results"
    
    try:
        # Check if the directory exists
        if os.path.exists(folder_path):
            # Iterate over all files in the directory and delete them
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # Delete file
                except Exception as e:
                    print(f"Error deleting file {file_name}: {e}")
            messagebox.showinfo("Success", "All calibration files have been deleted.")
        else:
            messagebox.showwarning("Warning", "Directory does not exist.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while deleting files: {e}")