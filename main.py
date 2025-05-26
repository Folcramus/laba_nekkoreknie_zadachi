import numpy as np
import cv2
import os
import sys

"""
Программа выполняет устранение размытия изображений с возможностью 
интерактивной настройки параметров. Поддерживает два типа размытия:
1. Линейное размытие (движение камеры)
2. Круговое размытие (расфокусировка)
"""


def estimate_blur_params(image):
    """Автоматическая оценка параметров размытия на изображении"""
    try:
        # Конвертация в 8-битный формат
        image_uint8 = np.uint8(image * 255)

        # 1. Оценка угла размытия через градиенты
        grad_x = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
        angle_deg = np.rad2deg(np.mean(np.arctan2(grad_y, grad_x))) % 180

        # 2. Оценка степени размытия через Фурье-анализ
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shifted) + 1)

        # Игнорируем центральную область
        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2
        magnitude[center_row-10:center_row+10, center_col-10:center_col+10] = 0

        # Эмпирическая оценка радиуса размытия
        blur_radius = int(np.clip(rows / (2 * np.max(magnitude)), 5, 50))

        # 3. Оценка SNR через лапласиан
        laplacian = cv2.Laplacian(image_uint8, cv2.CV_64F)
        snr_value = 10 * np.log10(np.mean(image) ** 2 / np.var(laplacian))
        snr_estimate = int(np.clip(snr_value, 10, 50))

        return angle_deg, blur_radius, snr_estimate
    except Exception as e:
        print(f"Ошибка оценки параметров: {e}")
        return 135, 22, 25  # Значения по умолчанию


def smooth_edges(image, border_size=31):
    """Размытие краев изображения для уменьшения артефактов"""
    height, width = image.shape[:2]

    # Добавляем зеркальные границы
    padded_image = cv2.copyMakeBorder(image, border_size, border_size,
                                    border_size, border_size, cv2.BORDER_WRAP)

    # Применяем гауссово размытие
    blurred = cv2.GaussianBlur(padded_image, (2*border_size+1, 2*border_size+1), -1)
    blurred = blurred[border_size:-border_size, border_size:-border_size]

    # Создаем весовую маску для плавного перехода
    y_coords, x_coords = np.indices((height, width))
    distances = np.dstack([x_coords, width - x_coords - 1,
                          y_coords, height - y_coords - 1]).min(-1)
    weights = np.minimum(np.float32(distances) / border_size, 1.0)

    # Смешиваем изображения
    return image * weights + blurred * (1 - weights)


def create_motion_blur_kernel(angle_degrees, length, size=65):
    """Создает ядро линейного размытия (движение)"""
    angle_radians = np.deg2rad(angle_degrees)
    kernel = np.ones((1, int(length)), np.float32)

    # Матрица поворота
    cos_val, sin_val = np.cos(angle_radians), np.sin(angle_radians)
    transform_matrix = np.float32([[cos_val, -sin_val, 0], [sin_val, cos_val, 0]])

    # Центрирование ядра
    center = size // 2
    transform_matrix[:, 2] = (center, center) - np.dot(transform_matrix[:, :2], ((length-1)*0.5, 0))

    # Применяем аффинное преобразование
    return cv2.warpAffine(kernel, transform_matrix, (size, size), flags=cv2.INTER_CUBIC)


def create_defocus_blur_kernel(diameter, size=65):
    """Создает ядро кругового размытия (расфокусировка)"""
    kernel = np.zeros((size, size), np.uint8)
    center = size // 2
    cv2.circle(kernel, (center, center), int(diameter)//2, 255, -1, cv2.LINE_AA)
    return np.float32(kernel) / 255.0


def apply_deconvolution(input_image, blur_angle, blur_radius, snr_ratio, is_defocus=True):
    """
    Применяет деконволюцию методом Винера
    Параметры:
    input_image - входное изображение
    blur_angle - угол размытия (для линейного)
    blur_radius - степень размытия
    snr_ratio - отношение сигнал/шум
    is_defocus - флаг типа размытия
    """
    # Предварительная обработка
    processed_image = smooth_edges(input_image)

    # Преобразование Фурье
    image_fft = cv2.dft(processed_image, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Создание ядра размытия
    if is_defocus:
        psf_kernel = create_defocus_blur_kernel(blur_radius)
    else:
        psf_kernel = create_motion_blur_kernel(blur_angle, blur_radius)
    psf_kernel /= psf_kernel.sum()

    # Подготовка PSF
    psf_padded = np.zeros_like(input_image)
    kernel_height, kernel_width = psf_kernel.shape
    psf_padded[:kernel_height, :kernel_width] = psf_kernel

    # Фурье-образ PSF
    psf_fft = cv2.dft(psf_padded, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kernel_height)

    # Расчет фильтра Винера
    psf_power = (psf_fft ** 2).sum(-1)
    inverse_psf = psf_fft / (psf_power + 10 ** (-0.1 * snr_ratio))[..., np.newaxis]

    # Применение фильтра
    result_fft = cv2.mulSpectrums(image_fft, inverse_psf, 0)

    # Обратное преобразование
    result = cv2.idft(result_fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # Коррекция положения
    result = np.roll(result, -kernel_height//2, 0)
    result = np.roll(result, -kernel_width//2, 1)

    # Нормализация
    return cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX), psf_kernel


def run_deblurring_interface():
    """Основная функция с интерфейсом пользователя"""
    # Загрузка изображения
    input_file = "blurred.jpg"
    file_path = os.path.join(os.path.dirname(__file__), input_file)

    original_image = cv2.imread(file_path, 0)  # Загрузка в градациях серого
    if original_image is None:
        print(f"Ошибка загрузки изображения {file_path}")
        return

    # Нормализация изображения
    original_image = np.float32(original_image) / 255.0
    cv2.imshow('Исходное изображение', original_image)

    # Автоматическая оценка параметров
    estimated_angle, estimated_radius, estimated_snr = estimate_blur_params(original_image)
    print(f"Автооценка: Угол={estimated_angle}°, Радиус={estimated_radius}, SNR={estimated_snr} дБ")

    # Инициализация интерфейса
    cv2.namedWindow('Результат деконволюции')
    cv2.namedWindow('Ядро размытия')

    # Создание элементов управления
    cv2.createTrackbar('Угол', 'Результат деконволюции', int(estimated_angle), 180, lambda x: None)
    cv2.createTrackbar('Радиус', 'Результат деконволюции', int(estimated_radius), 50, lambda x: None)
    cv2.createTrackbar('SNR', 'Результат деконволюции', int(estimated_snr), 50, lambda x: None)

    use_defocus = True  # Флаг типа размытия
    print("Управление:\nПробел - переключение типа размытия\nS - сохранение\nESC - выход")

    while True:
        # Получение текущих параметров
        current_angle = cv2.getTrackbarPos('Угол', 'Результат деконволюции')
        current_radius = cv2.getTrackbarPos('Радиус', 'Результат деконволюции')
        current_snr = cv2.getTrackbarPos('SNR', 'Результат деконволюции')

        # Выполнение деконволюции
        deblurred_result, blur_kernel = apply_deconvolution(
            original_image, current_angle, current_radius, current_snr, use_defocus)

        # Отображение результатов
        cv2.imshow('Результат деконволюции', deblurred_result)
        cv2.imshow('Ядро размытия', blur_kernel)

        # Обработка клавиш
        key = cv2.waitKey(30)
        if key == 27:  # ESC - выход
            break
        elif key == ord(' '):  # Пробел - смена типа размытия
            use_defocus = not use_defocus
            print(f"Тип размытия: {'расфокусировка' if use_defocus else 'движение'}")
        elif key == ord('s'):  # Сохранение результата
            cv2.imwrite('deblurred_result.jpg', deblurred_result * 255)
            print("Результат сохранен как 'deblurred_result.jpg'")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_deblurring_interface()