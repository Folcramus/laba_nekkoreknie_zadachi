import numpy as np
import cv2
import os
import sys

"""
Программа выполняет деконволюцию (устранение размытия) изображений с возможностью 
интерактивной настройки параметров. Поддерживает два типа размытия:
1. Движение (motion blur) - линейное размытие по заданному углу
2. Расфокусировка (defocus blur) - круговое размытие
"""


def estimate_blur_parameters(img):
    """Автоматическая оценка параметров размытия на изображении"""
    try:
        # Конвертация в 8-битный формат для операторов OpenCV
        img_uint8 = np.uint8(img * 255)

        # 1. Оценка угла размытия через градиенты Собеля
        sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        # Вычисление среднего угла градиентов
        angle_deg = np.rad2deg(np.mean(np.arctan2(sobel_y, sobel_x))) % 180

        # 2. Оценка степени размытия через анализ спектра Фурье
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)  # Центрирование спектра
        magnitude = np.log(np.abs(fft_shift) + 1)  # Логарифмический масштаб

        # Игнорируем центральную область (низкие частоты)
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        magnitude[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0

        # Эмпирическая оценка радиуса размытия
        d_estimate = int(np.clip(rows / (2 * np.max(magnitude)), 5, 50))

        # 3. Оценка SNR через лапласиан
        laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
        snr_value = 10 * np.log10(np.mean(img) ** 2 / np.var(laplacian))
        snr_estimate = int(np.clip(snr_value, 10, 50))

        return angle_deg, d_estimate, snr_estimate
    except Exception as e:
        print(f"Ошибка оценки параметров: {e}")
        return 135, 22, 25  # Значения по умолчанию


def blur_edge(img, d=31):
    """
    Размытие краев изображения для уменьшения артефактов при деконволюции.
    Параметр d определяет ширину области размытия.
    """
    h, w = img.shape[:2]

    # Добавляем зеркальные границы для уменьшения краевых эффектов
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)

    # Применяем гауссово размытие
    img_blur = cv2.GaussianBlur(img_pad, (2 * d + 1, 2 * d + 1), -1)[d:-d, d:-d]

    # Создаем весовую маску для плавного перехода
    y, x = np.indices((h, w))
    dist = np.dstack([x, w - x - 1, y, h - y - 1]).min(-1)
    w = np.minimum(np.float32(dist) / d, 1.0)

    # Смешиваем исходное и размытое изображение
    return img * w + img_blur * (1 - w)


def create_motion_kernel(angle_deg, d, sz=65):
    """
    Создает ядро размытия движения (линейное).
    angle_deg - угол направления размытия в градусах
    d - длина размытия
    sz - размер ядра
    """
    angle_rad = np.deg2rad(angle_deg)
    kern = np.ones((1, int(d)), np.float32)  # Линейное ядро

    # Матрица аффинного преобразования для поворота
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    A = np.float32([[c, -s, 0], [s, c, 0]])

    # Центрирование ядра
    sz2 = sz // 2
    A[:, 2] = (sz2, sz2) - np.dot(A[:, :2], ((d - 1) * 0.5, 0))

    # Применяем поворот с интерполяцией
    return cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)


def create_defocus_kernel(d, sz=65):
    """
    Создает круговое ядро размытия (расфокусировка).
    d - диаметр размытия
    sz - размер ядра
    """
    kern = np.zeros((sz, sz), np.uint8)
    # Рисуем белый круг на черном фоне
    cv2.circle(kern, (sz // 2, sz // 2), int(d) // 2, 255, -1, cv2.LINE_AA)
    return np.float32(kern) / 255.0  # Нормализация


def deconvolve_image(img, angle, d, snr, defocus=True):
    """
    Основная функция деконволюции по методу Винера.
    Параметры:
    img - входное изображение
    angle - угол размытия (для motion blur)
    d - степень размытия
    snr - оценка отношения сигнал/шум
    defocus - флаг типа размытия
    """
    # 1. Предварительная обработка изображения
    img_blurred = blur_edge(img)

    # 2. Преобразование Фурье изображения
    IMG = cv2.dft(img_blurred, flags=cv2.DFT_COMPLEX_OUTPUT)

    # 3. Создание PSF (Point Spread Function)
    if defocus:
        psf = create_defocus_kernel(d)
    else:
        psf = create_motion_kernel(angle, d)
    psf /= psf.sum()  # Нормализация

    # 4. Подготовка PSF для частотной области
    psf_pad = np.zeros_like(img)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf  # Центрирование

    # 5. Преобразование Фурье PSF
    PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)

    # 6. Вычисление фильтра Винера
    PSF2 = (PSF ** 2).sum(-1)
    iPSF = PSF / (PSF2 + 10 ** (-0.1 * snr))[..., np.newaxis]

    # 7. Применение фильтра
    RES = cv2.mulSpectrums(IMG, iPSF, 0)

    # 8. Обратное преобразование Фурье
    res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # 9. Коррекция смещения
    res = np.roll(res, -kh // 2, 0)
    res = np.roll(res, -kw // 2, 1)

    # 10. Нормализация результата
    return cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX), psf


def main():
    """Основная функция с интерфейсом пользователя"""
    # Загрузка изображения
    filename = "blurred.jpg"
    filepath = os.path.join(os.path.dirname(__file__), filename)

    img = cv2.imread(filepath, 0)  # Загрузка в градациях серого
    if img is None:
        print(f"Ошибка загрузки изображения {filepath}")
        return

    # Подготовка изображения
    img = np.float32(img) / 255.0  # Нормализация в [0, 1]
    cv2.imshow('Исходное изображение', img)

    # Автоматическая оценка параметров
    angle, d, snr = estimate_blur_parameters(img)
    print(f"Автооценка: Угол={angle}°, D={d}, SNR={snr} дБ")

    # Инициализация интерфейса
    cv2.namedWindow('Деконволюция')
    cv2.namedWindow('PSF')

    # Создание трекбаров
    cv2.createTrackbar('Угол', 'Деконволюция', int(angle), 180, lambda x: None)
    cv2.createTrackbar('D', 'Деконволюция', int(d), 50, lambda x: None)
    cv2.createTrackbar('SNR', 'Деконволюция', int(snr), 50, lambda x: None)

    defocus = True  # Флаг типа размытия
    print("Управление:\nПробел - переключение типа размытия\nS - сохранение\nESC - выход")

    while True:
        # Получение текущих параметров
        angle = cv2.getTrackbarPos('Угол', 'Деконволюция')
        d = cv2.getTrackbarPos('D', 'Деконволюция')
        snr = cv2.getTrackbarPos('SNR', 'Деконволюция')

        # Выполнение деконволюции
        result, psf = deconvolve_image(img, angle, d, snr, defocus)

        # Отображение результатов
        cv2.imshow('Деконволюция', result)
        cv2.imshow('PSF', psf)

        # Обработка клавиш
        key = cv2.waitKey(30)
        if key == 27:  # ESC - выход
            break
        elif key == ord(' '):  # Пробел - смена типа размытия
            defocus = not defocus
            print(f"Тип PSF: {'defocus' if defocus else 'motion'}")
        elif key == ord('s'):  # Сохранение результата
            cv2.imwrite('deblurred_result.jpg', result * 255)
            print("Результат сохранен как 'deblurred_result.jpg'")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()