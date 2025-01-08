import cv2
import os

# 加载级联分类器
# 首先确保级联分类器文件存在
cascade_path = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_eye.xml'))

# 检查级联分类器是否成功加载
if face_cascade.empty():
    raise IOError('无法加载人脸分类器')
if eye_cascade.empty():
    raise IOError('无法加载眼睛分类器')

try:
    # 读取图像 - 使用正确的路径格式
    img = cv2.imread(r'C:\Users\Administrator\Desktop\1\3.jpg')
    
    # 检查图像是否成功加载
    if img is None:
        raise IOError('无法加载图像')

    # 将图像转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # 在检测到的人脸上画框
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # 在人脸上检测眼睛
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f'发生错误: {str(e)}')
