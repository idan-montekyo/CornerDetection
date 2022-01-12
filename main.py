from Functions import *
import tkinter as tk
from tkinter import filedialog

# After running the program, a new window will open.
# Select an image (with a valid path!), and the algorithm will mark it's corners.
# Set 'displayProcess' to 'True' in order to see the process in addition.

if __name__ == '__main__':

    # Image path required to be in English ONLY!
    # If the path includes hebrew letters, please move the folder/images to the desktop and continue from there.
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    userImage = cv2.imread(file_path)

    # Set displayProcess value to 'False' in order to print Before-After alone,
    # Set displayProcess value to 'True' in order to print the process in addition (gradients and result matrix).
    displayProcess = False

    myHarris(userImage, displayProcess)
