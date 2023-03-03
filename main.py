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
import imageprocessing as image_processing
import textprocessing as text_processing

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # path for tesseract. 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set the window background color to light gray
        self.setStyleSheet("background-color: #f0f0f0;")
        # Get the screen resolution and set the window width and height to 2/3 of it
        screen_resolution = QApplication.desktop().screenGeometry()
        width, height = int(screen_resolution.width() * 2 / 3), int(screen_resolution.height() * 2 / 3)
        self.setGeometry(0, 0, width, height)

        self.title = 'Image Viewer'
        self.initUI()

    
    
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
        self.btn.setStyleSheet('''
                QPushButton {
                    background-color: #008CBA;
                    color: white;
                    font-weight: bold;
                    font-size: 18px;
                    border-radius: 5px;
                    border: none;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #007A8C;
                }
                QPushButton:pressed {
                    background-color: #006B7B;
                }
            ''')

        # Create button to recognize the text 
        self.recognize_btn = QPushButton('Analyze text from image', self)
        # Move the position of the recognize button down
        self.recognize_btn.setGeometry(50, 250, 200, 50)
        self.recognize_btn.clicked.connect(self.image_to_text)
        self.recognize_btn.setStyleSheet('''
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 18px;
                border-radius: 5px;
                border: none;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #3E8E41;
            }
            QPushButton:pressed {
                background-color: #2E6738;
            }
        ''')
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


    def load_image_from_label(self):
        # Load the selected image from the label and convert it to a numpy array
        img = self.label.pixmap().toImage()
        pil_img = ImageQt.fromqimage(img)
        return np.array(pil_img)

    

    def display_text_in_dialog(self, text):
        # Display extracted text in a dialog
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
        def copy_to_clipboard():
            clipboard = QGuiApplication.clipboard()
            clipboard.setText(text_edit.toPlainText())
            dialog.accept()

        button_copy.clicked.connect(lambda: copy_to_clipboard())

        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box, 1, 0)

        # dialog.exec_()
        return dialog

    def image_to_text(self):
        # Load input image
        img = self.label.pixmap().toImage()
        pil_img = ImageQt.fromqimage(img)
        
        # Preprocess image
        
        np_img = image_processing.preprocess_image(pil_img)
        # Run OCR on preprocessed image
        text = text_processing.run_ocr(np_img)
        
        # Parse OCR results into text
        text = text_processing.parse_ocr_result(text)
        
        # Display text in a dialog
        dialog = self.display_text_in_dialog(text)
        dialog.exec_()
        
        return text

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
