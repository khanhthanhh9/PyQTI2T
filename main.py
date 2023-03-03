import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QCheckBox, QDialog, QDialogButtonBox, QGridLayout, QPlainTextEdit
from PyQt5.QtGui import QPixmap, QGuiApplication, QClipboard, QImage
from PyQt5.QtCore import Qt, QSize, QSettings
import pytesseract
from pytesseract import Output
from PIL import Image, ImageQt
import pandas as pd
import numpy as np 
import cv2
import os
import time
import re

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # path for tesseract. 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Get the screen resolution and set the window width and height to 2/3 of it
        screen_resolution = QApplication.desktop().screenGeometry()
        width, height = int(screen_resolution.width() * 2 / 3), int(screen_resolution.height() * 2 / 3)
        self.setGeometry(0, 0, width, height)

        self.title = 'Image Viewer'
        self.initUI()

    def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
        # Apply the bilateral filter to the image
        filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        
        return filtered_image
    
    def remove_redundant_spaces(self, string):
    # Remove redundant spaces while keeping newlines normally
        pattern = r"(?m)^(?:\S+\s+)+\S+$|\s+"
        string = re.sub(pattern, lambda match: match.group().replace(" ", "") if len(match.group().split()) > 1 else match.group(), string)
        return string
    
    def enhance_dim_text(self, image):


        # Create a CLAHE object and apply it to the image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)

        # Return the enhanced image
        return enhanced
    
    def adaptive_threshold(self, image):
        # Get the average color of the four border pixels
        top = np.mean(image[0])
        bottom = np.mean(image[-1])
        left = np.mean(image[:, 0])
        right = np.mean(image[:, -1])

        # Determine the type of thresholding based on the average color
        if top + bottom + left + right > 510:
            # Use binary thresholding for light background
            thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        else:
            # Use binary inverse thresholding for dark background
            thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        return thresh
    
    def initUI(self):
        self.setWindowTitle(self.title)

        # Create a status bar
        self.statusBar = self.statusBar()

        # Create a label to display information about the selected image in the status bar
        self.status_image_info = QLabel(self)
        self.statusBar.addPermanentWidget(self.status_image_info)

        # Set initial text for the status image info label
        self.status_image_info.setText('No image selected')

        # Create button to select an image
        self.btn = QPushButton('Select Image', self)
        self.btn.clicked.connect(self.selectImage)


        # Create button to recognize the text 
        self.recognize_btn = QPushButton('Analyze text from image', self)
        # Move the position of the recognize button down
        self.recognize_btn.setGeometry(50, 250, 200, 50)
        self.recognize_btn.clicked.connect(self.image_to_text)

        # Create label to display the selected image
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(False)
        self.label.setMinimumSize(1, 1)

        # Create a vertical layout and add the button and label to it
        vbox = QVBoxLayout()
        vbox.addWidget(self.btn)
        vbox.addWidget(self.recognize_btn)
        vbox.addWidget(self.label)
        

        # Create a widget and set the layout to it
        widget = QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)
        self.show()

    
    def update_image_info(self, filename):
            # Load the selected image and get its size and resolution
            img = Image.open(filename)
            width, height = img.size
            resolution = str(img.info.get('dpi'))

            # Update the status image info label with the file name, size, and resolution
            self.status_image_info.setText('Selected image: ' + os.path.basename(filename) + ' (' + str(width) + ' x ' + str(height) + ', ' + resolution + ')')
    
    def to_grayscale(self, image):
        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_image
    
    def morphological_transformations(self, image, iterations=1):
        # Create a structuring element
        kernel = np.ones((1, 1), np.uint8)
        
        # Dilate the image
        dilated_image = cv2.dilate(image, kernel, iterations=iterations)
        
        # Erode the image
        eroded_image = cv2.erode(dilated_image, kernel, iterations=iterations)
        
        return eroded_image

    def scale_dpi(self, image, target_dpi=300):
        # Get the current DPI of the image
        current_dpi = image.shape[0] / float(image.shape[1])
        
        # Calculate the scaling factor
        scaling_factor = target_dpi / current_dpi
        
        # Resize the image with the scaling factor
        new_size = (int(image.shape[1] * scaling_factor), int(image.shape[0] * scaling_factor))
        resized_image = cv2.resize(image, new_size)
        
        # Save the resized image with the new DPI
        retval, buffer = cv2.imencode('.jpg', resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1])
        output_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        
        return output_image
    
    def add_bilateral_noise(self, img, d=9, sigmaColor=75, sigmaSpace=75):
        """
        Adds bilateral noise to the input image.

        Parameters:
        img (numpy.ndarray): Input image.
        d (int): Diameter of each pixel neighborhood that is used during filtering (default=9).
        sigmaColor (float): Filter sigma in the color space (default=75).
        sigmaSpace (float): Filter sigma in the coordinate space (default=75).

        Returns:
        numpy.ndarray: Noisy image.
        """
        # Convert the image to float32
        img = np.float32(img)

        # Add bilateral noise to the image
        noisy_img = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

        # Clip the pixel values to the valid range
        noisy_img = np.clip(noisy_img, 0, 255)

        # Convert the image back to uint8
        noisy_img = np.uint8(noisy_img)

        return noisy_img

    def selectImage(self):
        
        # Create a QSettings object to remember the last location used by the QFileDialog
        settings = QSettings('MyOrganization', 'MyApp')
        last_dir = settings.value('last_dir', '', type=str)

        # Open a file dialog to select an image
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", last_dir, "Images (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        self.update_image_info(fileName)
        if fileName:
            pixmap = QPixmap(fileName)

            # Scale the image to fit the label if it is larger
            if pixmap.width() > self.label.width() or pixmap.height() > self.label.height():
                scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            else:
                scaled_pixmap = pixmap

            # # Display the image in the label
            self.label.setPixmap(scaled_pixmap)
            

    
    def image_to_text(self):
    # Load input image
        custom_config = r'--oem 3 --psm 6 -l eng'
        
        img = self.label.pixmap().toImage()
        # Convert to grayscale
        # Convert the DPI to 300
        # Apply dilation and erosion
        # Applying Blur with bilateral filter
        

        pil_img = ImageQt.fromqimage(img)
        # Convert the ImageQt object to a numpy array
        np_img = np.array(pil_img)
        np_img = self.to_grayscale(np_img)
        # np_img = self.scale_dpi(np_img)
        # np_img = self.enhance_dim_text(np_img)
        np_img = self.adaptive_threshold(np_img)
        np_img = self.morphological_transformations(np_img,5)
        # np_img = self.add_bilateral_noise(np_img)
        cv2.imshow('Processed image', np_img)
        cv2.waitKey(0)
        d = pytesseract.image_to_data(np_img, config=custom_config, output_type=Output.DICT)

        # Convert OCR result into DataFrame
        df = pd.DataFrame(d)
        # Clean up blanks
        df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]

        # Sort blocks vertically
        sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()

        # Construct text from OCR result
        text = ''
        for block in sorted_blocks:
            curr = df1[df1['block_num'] == block] # CURRENT
            sel = curr[curr.text.str.len() >= 1] # get word 
            char_w = (sel.width / sel.text.str.len()).mean()
            prev_par, prev_line, prev_left = 0, 0, 0
            for ix, ln in curr.iterrows():
                if prev_par != ln['par_num']:
                    text += '\n'
                    prev_par = ln['par_num']
                    prev_line = ln['line_num']
                    prev_left = 0
                elif prev_line != ln['line_num']:
                    text += '\n'
                    prev_line = ln['line_num']
                    prev_left = 0

                added = 0
                if ln['left'] / char_w > prev_left + 1:
                    added = int(ln['left'] / char_w) - prev_left
                    text += ' ' * added
                text += ln['text'] + ' '
                prev_left += len(ln['text']) + added + 1
            text = self.remove_redundant_spaces(text)
            text += '\n'
        with open('output.txt', 'w') as f:
            f.write(text)
        # Create QDialog to display text
        dialog = QDialog(self)
        dialog.setWindowTitle('Extracted Text')
        layout = QVBoxLayout(dialog)
        label = QLabel(text, dialog)
        layout.addWidget(label)

        # Create QDialog to display text
        dialog = QDialog(self)
        dialog.setWindowTitle('Extracted Text')
        layout = QGridLayout(dialog)
        text_edit = QPlainTextEdit(dialog)
        text_edit.setPlainText(text)
        text_edit.setMinimumSize(800, 600) # set the minimum size of the text edit widget

        layout.addWidget(text_edit, 0, 0)

        # Add a "Copy to Clipboard" button
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_copy = button_box.addButton("Copy to Clipboard", QDialogButtonBox.ActionRole)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box, 1, 0)

        dialog.exec_()
        def copy_to_clipboard():
            clipboard = QGuiApplication.clipboard()
            clipboard.setText(text_edit.toPlainText())
            dialog.accept()

        return text

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_V:
            clipboard = QGuiApplication.clipboard()
            mime_data = clipboard.mimeData()
            if mime_data.hasImage():
                # Load the image from the clipboard and convert it to a QPixmap
                image = QImage(mime_data.imageData())
                pixmap = QPixmap.fromImage(image)
                # Scale the image to fit the label if it is larger
                if pixmap.width() > self.label.width() or pixmap.height() > self.label.height():
                    scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
                else:
                    scaled_pixmap = pixmap
                # Display the image in the label
                self.label.setPixmap(scaled_pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
