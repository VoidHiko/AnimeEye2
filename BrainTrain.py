import cv2                                          
import os
import numpy as np
from PIL import Image
path = os.path.dirname(os.path.abspath(__file__))
recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
dataPath = path+r'/DataSet'
def get_images_and_labels(datapath):
     # получаем путь к картинкам
     image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
     # списки картинок и подписей на старте пустые
     images = []
     labels = []
     # перебираем все картинки в датасете 
     for image_path in image_paths:
         # читаем картинку и сразу переводим в ч/б
         image_pill = Image.open(image_path).convert('L')
         # переводим картинку в numpy-массив
         image = np.array(image_pill, 'uint8')
         # получаем id пользователя из имени файла
         nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))      #kotone = 1
         # определяем лицо на картинке
         faces = faceCascade.detectMultiScale(image)
         # если лицо найдено
         for (x, y, w, h) in faces:
             # добавляем его к списку картинок 
             images.append(image[y: y + h, x: x + w])
             # добавляем id пользователя в список подписей
             labels.append(nbr)
             # выводим текущую картинку на экран
             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             # делаем паузу
             cv2.waitKey(100)
     # возвращаем список картинок и подписей
     return images, labels
# получаем список картинок и подписей
images, labels = get_images_and_labels(dataPath)
# обучаем модель распознавания на наших картинках и учим сопоставлять её лица и подписи к ним
recognizer.train(images, np.array(labels))
# сохраняем модель
recognizer.save(path+r'/trainer/trainer.yml')
# удаляем из памяти все созданные окнаы
cv2.destroyAllWindows()