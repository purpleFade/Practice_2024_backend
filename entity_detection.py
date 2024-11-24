from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import json
import os
import uuid
from datetime import datetime
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# ============================================
# 1. Функції попередньої обробки
# ============================================

def load_image(image_path):
    """
    Завантаження зображення з файлу.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f'Error: Image {image_path} not found.')
        exit()
    return image

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Корекція яскравості та контрасту зображення.
    """
    img = np.int16(image)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img

def scale_and_align_image(image, target_size=(1280, 720)):
    """
    Масштабування та вирівнювання зображення до стандартного розміру.
    """
    # Масштабування зображення
    scaled_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    # Вирівнювання зображення (якщо потрібно)
    # Тут можна додати код для вирівнювання, якщо це необхідно
    aligned_image = scaled_image  # Поки що залишимо без змін
    return aligned_image

def preprocess_image(image, brightness=0, contrast=0, target_size=(800, 600)):
    """
    Попередня обробка: корекція яскравості та контрасту, масштабування, згладжування та перетворення в відтінки сірого.
    """
    # Корекція яскравості та контрасту
    adjusted_image = adjust_brightness_contrast(image, brightness, contrast)
    # Масштабування та вирівнювання
    processed_image = scale_and_align_image(adjusted_image, target_size)
    # Гауссове згладжування для зменшення шуму
    blurred_image = cv2.GaussianBlur(processed_image, (5, 5), 1.5)
    # Перетворення в відтінки сірого
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    return gray_image, processed_image  # Повертаємо також оброблене зображення в кольорі

# ============================================
# 2. Виявлення країв
# ============================================

def edge_detection(image):
    """
    Виконання виявлення країв за допомогою алгоритму Canny.
    """
    edges = cv2.Canny(image, 50, 150)
    return edges

# ============================================
# 3. Пошук контурів
# ============================================

def contour_detection(original_image, edges):
    """
    Пошук та малювання контурів на оригінальному зображенні.
    """
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = original_image.copy()
    object_info = []

    # Фільтрація контурів за площею
    min_area = 100  # Порогова площа
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Якщо контури знайдені
    if large_contours:
        # Знайти найбільший контур (головний об'єкт)
        main_contour = max(large_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        cv2.rectangle(contour_image, (x, y), (x+w, y+h), (0, 0, 255), 3)
        # Додати назву об'єкта
        object_name = "Main Object"
        cv2.putText(contour_image, object_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # Зберегти інформацію про головний об'єкт
        obj = {
            'id': 0,
            'type': object_name,
            'coordinates': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        }
        object_info.append(obj)
    else:
        print("Головний об'єкт не знайдено.")

    return contour_image, object_info

# ============================================
# 4. Метод Watershed
# ============================================

def watershed_segmentation(original_image):
    """
    Застосування методу Watershed для сегментації зображення.
    """
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Порогове перетворення
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Усунення шуму
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Визначення фону та переднього плану
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Невідома область
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Маркування
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(original_image, markers)
    # Малювання меж
    watershed_image = original_image.copy()
    watershed_image[markers == -1] = [255, 0, 0]
    return watershed_image

# ============================================
# 5. Метод K-means Clustering
# ============================================

def kmeans_segmentation(original_image, K=3):
    """
    Сегментація зображення за допомогою кластеризації K-means.
    """
    Z = original_image.reshape((-1, 3))
    Z = np.float32(Z)
    # Критерії та застосування k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Перетворення назад в зображення
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(original_image.shape)
    return segmented_image

# ============================================
# 6. Виявлення ознак за допомогою SIFT
# ============================================

def sift_detection(original_image, gray_image):
    """
    Виявлення ключових точок та дескрипторів за допомогою SIFT.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    sift_image = cv2.drawKeypoints(original_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Збір інформації про об'єкти
    object_info = []
    for idx, keypoint in enumerate(keypoints):
        obj = {
            'id': idx,
            'type': 'SIFT Keypoint',
            'coordinates': {'x': int(keypoint.pt[0]), 'y': int(keypoint.pt[1])}
        }
        object_info.append(obj)
    return sift_image, object_info

# ============================================
# 7. Haar-класифікатор
# ============================================

def haar_cascade_detection(original_image, gray_image, cascade_path):
    """
    Виявлення об'єктів за допомогою Haar-класифікатора.
    """
    # Завантаження класифікатора
    cascade = cv2.CascadeClassifier(cascade_path)
    # Виявлення об'єктів
    objects = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    # Малювання рамок навколо об'єктів
    haar_image = original_image.copy()
    object_info = []
    for idx, (x, y, w, h) in enumerate(objects):
        cv2.rectangle(haar_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        obj = {
            'id': idx,
            'type': 'Haar Detected Object',
            'coordinates': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        }
        object_info.append(obj)
    return haar_image, object_info

# ============================================
# 8. Виявлення об'єктів за допомогою YOLOv3
# ============================================

def yolo_detection(original_image, config_path, weights_path, names_path, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Виявлення об'єктів за допомогою YOLOv3.
    """
    # Завантаження назв класів
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    # Завантаження нейронної мережі
    net = cv2.dnn.readNet(weights_path, config_path)
    # Підготовка зображення
    height, width = original_image.shape[:2]
    blob = cv2.dnn.blobFromImage(original_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    # Отримання назв вихідних шарів
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)
    # Ініціалізація списків
    class_ids = []
    confidences = []
    boxes = []
    # Обробка вихідних даних
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > confidence_threshold:
                # Виявлено об'єкт
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                class_ids.append(class_id)
    # Придушення некоректних спрацьовувань
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    # Малювання рамок
    yolo_image = original_image.copy()
    object_info = []
    if len(indices) > 0:
        for idx, i in enumerate(indices.flatten()):
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            conf = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(yolo_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(yolo_image, f"{label} {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Збір інформації про об'єкт
            obj = {
                'id': idx,
                'type': label,
                'confidence': float(conf),
                'coordinates': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            }
            object_info.append(obj)
    return yolo_image, object_info

# ============================================
# 9. Головна функція
# ============================================

def main():
    image_path = 'object.jpg'
    image = load_image(image_path)
    # Попередня обробка
    gray_image, processed_image = preprocess_image(image, brightness=30, contrast=30, target_size=(800, 600))

    # Створення унікальної теки для збереження результатів
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:6]
    output_dir = f'results_{timestamp}_{unique_id}'
    os.makedirs(output_dir, exist_ok=True)


    print(f'Виявлення об’єктів завершено. Результати збережено в папку "{output_dir}".')
# ============================================
# Шлях для завантаження та обробки зображення
# ============================================

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    # Створення унікальної теки для збереження результатів
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:6]
    output_dir = f'results_{timestamp}_{unique_id}'
    os.makedirs(output_dir, exist_ok=True)

    # Збереження отриманого зображення
    image_path = os.path.join(output_dir, 'uploaded_image.jpg')
    file.save(image_path)

    # Завантаження та обробка зображення
    image = load_image(image_path)
    gray_image, processed_image = preprocess_image(image, brightness=30, contrast=30, target_size=(800, 600))

    all_object_info = []

    # Виявлення країв
    edges = edge_detection(gray_image)
    cv2.imwrite(os.path.join(output_dir, 'edges.jpg'), edges)

    # Пошук контурів
    contour_img, contour_info = contour_detection(processed_image, edges)
    cv2.imwrite(os.path.join(output_dir, 'contours.jpg'), contour_img)
    all_object_info.extend(contour_info)

    # Метод Watershed
    watershed_img = watershed_segmentation(processed_image)
    cv2.imwrite(os.path.join(output_dir, 'watershed.jpg'), watershed_img)

    # K-means Clustering
    kmeans_img = kmeans_segmentation(processed_image, K=3)
    cv2.imwrite(os.path.join(output_dir, 'kmeans.jpg'), kmeans_img)

    # SIFT Detection
    sift_img, sift_info = sift_detection(processed_image, gray_image)
    cv2.imwrite(os.path.join(output_dir, 'sift.jpg'), sift_img)
    all_object_info.extend(sift_info)

    # Haar Cascade Detection
    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    haar_img, haar_info = haar_cascade_detection(processed_image, gray_image, haar_cascade_path)
    cv2.imwrite(os.path.join(output_dir, 'haar.jpg'), haar_img)
    all_object_info.extend(haar_info)

    # YOLOv3 Detection
    yolo_config_path = 'yolov3.cfg'
    yolo_weights_path = 'yolov3.weights'
    yolo_names_path = 'coco.names'
    yolo_img, yolo_info = yolo_detection(processed_image, yolo_config_path, yolo_weights_path, yolo_names_path)
    cv2.imwrite(os.path.join(output_dir, 'yolo.jpg'), yolo_img)
    all_object_info.extend(yolo_info)

    # Генерація JSON-файлу з інформацією про об'єкти
    json_path = os.path.join(output_dir, 'object_info.json')
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_object_info, json_file, indent=4, ensure_ascii=False)

    # Підготовка відповіді
    response = {
        'message': 'Image processed successfully.',
        'results_folder': output_dir,
        'object_info': all_object_info
    }

    return jsonify(response), 200

# Шлях для отримання згенерованих зображень
@app.route('/results/<path:filename>', methods=['GET'])
def get_result_file(filename):
    return send_from_directory('.', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
