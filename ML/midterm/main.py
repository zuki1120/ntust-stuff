# from argparse import ArgumentParser
# from keras.models import load_model
# from time import sleep
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing import image
# import cv2
# import numpy as np

# # 載入分類器
# face_cascade = cv2.CascadeClassifier('face_detect.xml')

# # 載入模型
# classifier = load_model('./output/model.h5')

# video_path = './videos/'

# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# def main():
#     if args.video:
#         filename = args.video
#         cap = cv2.VideoCapture(video_path + filename)

#     while True:
#         ret, frame = cap.read()
#         if ret is not True:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, minNeighbors = 10)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
#             roi_gray = gray[y : y + h, x : x + w]
#             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)

#             if np.sum([roi_gray]) != 0:
#                 roi = roi_gray.astype('float') / 255
#                 roi = img_to_array(roi)
#                 roi = np.expand_dims(roi, axis = 0)

#                 # print(classifier.predict(roi)[0])
#                 label = emotion_labels[np.argmax(classifier.predict(roi)[0])]
#                 label_position = (x, y - 10)
#                 cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             else:
#                 cv2.putText(frame,'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.imshow('Emotion Detector', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--video', type = str,
#                         help = '輸入影片辨識表情',
#                         default = '')
#     args = parser.parse_args()

#     main()

from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from torchvision import transforms
from time import sleep
import cv2
import numpy as np
from model import CNN, VGG16

# 載入分類器
face_cascade = cv2.CascadeClassifier('face_detect.xml')

# 載入模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# model = torch.load('./checkpoint.pt', map_location = device)

model = VGG16().to(device)
model.load_state_dict(torch.load('checkpoint.pt', map_location = device))

model.eval()  # 設定模型為推論模式

video_path = './videos/'

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 定義轉換
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.2,))
])

def main():
    if args.video:
        filename = args.video
        cap = cv2.VideoCapture(video_path + filename)

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, minNeighbors = 10)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y: y + h, x: x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = transform(roi_gray).unsqueeze(0).to(device)  # 加上 batch 維度
                with torch.no_grad():
                    outputs = model(roi)
                    _, predicted = torch.max(outputs, 1)
                    label = emotion_labels[predicted.item()]
                
                label_position = (x, y - 10)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video', type=str,
                        help='輸入影片辨識表情',
                        default='')
    args = parser.parse_args()

    main()