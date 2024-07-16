import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
from tkinter import Tk, Canvas, Button, Label, filedialog
from PIL import Image

class HopfieldNetwork:
    def __init__(self, image_size):
        self.image_size = image_size
        self.weights = np.zeros((image_size, image_size))

    def train(self, images):
        num_images = len(images)
        image_size = self.image_size

        for i in range(num_images):
            image = images[i]
            image = np.reshape(image, (image_size, 1))
            self.weights += np.outer(image, image)
            np.fill_diagonal(self.weights, 0)

    def compress(self, image):
        image_size = self.image_size
        image = np.reshape(image, (image_size, 1))
        num_iterations = 100

        for _ in range(num_iterations):
            image = np.sign(np.dot(self.weights, image))

        return image.flatten()

class ImageCompressor:
    def __init__(self):
        self.image = None
        self.compressed_image = None

    def load_image(self, filename):
        self.image = plt.imread(filename)

    def display_image(self):
        plt.imshow(self.image)
        plt.axis('off')
        plt.show()

    def compress_image(self):
        image_size = self.image.size
        flattened_image = self.image.flatten()

        network = HopfieldNetwork(image_size)
        network.train([flattened_image])
        self.compressed_image = network.compress(flattened_image)
        self.compressed_image = self.compressed_image.reshape(self.image.shape)

    def get_image_file_size(self, filename):
        return os.path.getsize(filename)

    def get_compressed_image_size(self):
        return self.compressed_image.nbytes

    def display_compressed_image(self):
        plt.imshow(self.compressed_image)
        plt.axis('off')
        plt.show()

    def load_and_resize_image(self, filename, new_size):
        # Загружаем изображение
        self.image = Image.open(filename)
        # Меняем его размер
        self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)
        # Конвертируем изображение обратно в массив для дальнейшей работы
        self.image = np.array(self.image)
        original_size = self.get_image_file_size(filename)
        self.original_size = original_size

def open_file(compressor, original_size_label, compressed_size_label):
    Tk().withdraw()
    filename = filedialog.askopenfilename()
    if filename:
        # Запрашиваем у пользователя размер нового изображения
        new_width = int(input("Введите ширину нового изображения: "))
        new_height = int(input("Введите высоту нового изображения: "))
        # Загрузка и изменение размера изображения
        compressor.load_and_resize_image(filename, (new_width, new_height))
        compressor.display_image()
        compressor.compress_image()
        compressor.display_compressed_image()
        original_size = compressor.get_image_file_size(filename)
        original_size_label.config(text=f"Original Size: {original_size} bytes")
        compressed_size = compressor.get_compressed_image_size()
        compressed_size_label.config(text=f"Compressed Size: {compressed_size} bytes")

def main():
    compressor = ImageCompressor()

    # Создаем GUI-интерфейс
    root = Tk()
    root.title("Image Compressor")

    original_size_label = Label(root, text="Original Size: 0 bytes")
    original_size_label.pack()

    compressed_size_label = Label(root, text="Compressed Size: 0 bytes")
    compressed_size_label.pack()

    # Создаем кнопку "Загрузить изображение"
    file_button = Button(root, text="Загрузить изображение",
                         command=lambda: open_file(compressor, original_size_label, compressed_size_label))
    file_button.pack()

    # Запускаем цикл обработки событий
    root.mainloop()

if __name__ == "__main__":
    main()