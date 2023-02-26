import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QGuiApplication, QClipboard, QImage
from PyQt5.QtCore import Qt, QSize, QSettings
import pytesseract
from pytesseract import Output
from PIL import Image, ImageQt
import pandas as pd
import numpy as np 
import cv2
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

    def initUI(self):
        self.setWindowTitle(self.title)

        # Create button to select an image
        self.btn = QPushButton('Select Image', self)
        self.btn.clicked.connect(self.selectImage)

        # Create button to recognize the text 
        self.recognize_btn = QPushButton('Analyze text from image', self)
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
        if fileName:
            pixmap = QPixmap(fileName)

            # Scale the image to fit the label if it is larger
            if pixmap.width() > self.label.width() or pixmap.height() > self.label.height():
                scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            else:
                scaled_pixmap = pixmap

            # Display the image in the label
            self.label.setPixmap(scaled_pixmap)

            # Remember the last location used by the QFileDialog
            settings.setValue('last_dir', fileName)
    
    def image_to_text(self):
    # Load input image
        custom_config = r'--oem 3 --psm 6 -l eng'
        
        img = self.label.pixmap().toImage()
    
        pil_img = ImageQt.fromqimage(img)
        # Convert the ImageQt object to a numpy array
        np_img = np.array(pil_img)
        # np_img = self.add_bilateral_noise(np_img)
        # Convert the noisy numpy array back to an ImageQt object
        # pil_img = ImageQt.ImageQt(Image.fromarray(np_img))

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
            curr = df1[df1['block_num'] == block]
            sel = curr[curr.text.str.len() >= 1]
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
            text += '\n'
        # print(text)
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
