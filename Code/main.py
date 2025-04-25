import tkinter as tk
import threading
import warnings
from clear_calibration_results import delete_all_files
from webcam_inference import update_offset


from webcam_inference import start_calibration_thread , start_gaze_estimation , start_webcam_inference


warnings.filterwarnings("ignore")



def create_gui():
    """Create and display the GUI window."""
    
    # Create main window
    window = tk.Tk()
    window.title("Calibration and Webcam Interface")
    window.geometry("400x600")  # Increase window size for better layout
    window.configure(bg='#2E3B4E')  # Set a background color

    # Add a header label
    header_label = tk.Label(window, text="Calibration & Webcam", font=("Helvetica", 16, 'bold'), fg='white', bg='#2E3B4E')
    header_label.pack(pady=20)

    # Style for the buttons
    button_style = {
        'width': 25,
        'height': 2,
        'bg': '#3C91E6',  # Button background color
        'fg': 'white',  # Button text color
        'font': ('Arial', 12, 'bold'),
        'bd': 0,  # Remove border
        'relief': 'flat',
        'activebackground': '#4E9ED7',  # Active button color
        'activeforeground': 'white'
    }

    # Button to start webcam inference in a separate thread
    webcam_button = tk.Button(window, text="Start Webcam Inference", command=lambda: threading.Thread(target=start_webcam_inference, daemon=True).start(), **button_style)
    webcam_button.pack(pady=15)

    # Button to start Calibration
    calibration_button = tk.Button(window, text="Start Calibration", command=start_calibration_thread, **button_style)
    calibration_button.pack(pady=15)

    # Button to start Gaze Estimation model
    gaze_button = tk.Button(window, text="Start Gaze Estimation", command=lambda: threading.Thread(target=start_gaze_estimation, daemon=True).start(), **button_style)
    gaze_button.pack(pady=15)

    # Slider to adjust the offset value
    offset_slider_label = tk.Label(window, text="Adjust Y Offset", font=("Arial", 12, 'bold'), fg='white', bg='#2E3B4E')
    offset_slider_label.pack(pady=10)

    offset_slider = tk.Scale(window, from_=-20, to_=20, orient="horizontal", command=update_offset, sliderlength=20, length=250, bg='#3C91E6', fg='white', font=("Arial", 10))
    offset_slider.set(7)  # Set initial value of the slider to 7
    offset_slider.pack(pady=15)
    
    # Button to delete all files in the calibration results folder
    delete_button = tk.Button(window, text="Delete All Calibration Files", width=25, height=2, bg="#E74C3C", fg="white", font=("Arial", 12, 'bold'), command=delete_all_files)
    delete_button.pack(pady=20)

    # Exit button
    exit_button = tk.Button(window, text="Exit", command=window.quit, width=25, height=2, bg="#E74C3C", fg="white", font=("Arial", 12, 'bold'), bd=0, relief='flat', activebackground="#C0392B", activeforeground="white")
    exit_button.pack(pady=20)

    # Start the Tkinter event loop
    window.mainloop()


    

if __name__ == "__main__":
    create_gui()