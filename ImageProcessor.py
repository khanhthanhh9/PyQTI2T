import cv2
import numpy as np


class ImageProcessor:
    def __init__(self):
        self._pytesseract_path = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        self._pytesseract_config = r'--oem 3 --psm 6 -l eng'

    def to_grayscale(self, image):
        """
        Convert an image to grayscale.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def adaptive_threshold(self, image):
        """
        Apply adaptive thresholding to an image.
        """
        top = np.mean(image[0])
        bottom = np.mean(image[-1])
        left = np.mean(image[:, 0])
        right = np.mean(image[:, -1])

        if top + bottom + left + right > 510:
            thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        else:
            thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        return thresh

    def morphological_transformations(self, image, iterations=1):
        """
        Apply morphological transformations to an image.
        """
        kernel = np.ones((1, 1), np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations=iterations)
        eroded_image = cv2.erode(dilated_image, kernel, iterations=iterations)
        return eroded_image

    def add_bilateral_noise(self, img, d=9, sigmaColor=75, sigmaSpace=75):
        """
        Adds bilateral noise to an image.
        """
        img = np.float32(img)
        noisy_img = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        noisy_img = np.clip(noisy_img, 0, 255)
        noisy_img = np.uint8(noisy_img)
        return noisy_img

    def enhance_dim_text(self, image):
        """
        Enhance the dim text in an image.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        return enhanced

    def image_to_text(self, image):
        """
        Perform OCR on an image and return the recognized text.
        """
        import pytesseract
        from pytesseract import Output

        pytesseract.pytesseract.tesseract_cmd = self._pytesseract_path

        np_img = np.array(image)
        np_img = self.to_grayscale(np_img)
        np_img = self.adaptive_threshold(np_img)
        np_img = self.morphological_transformations(np_img, 5)
        np_img = self.add_bilateral_noise(np_img)
        np_img = self.enhance_dim_text(np_img)

        d = pytesseract.image_to_data(np_img, config=self._pytesseract_config, output_type=Output.DICT)

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
        return text
