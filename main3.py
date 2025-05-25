import numpy as np


def solve_tikhonov(A, f, alpha=0.1):
    """
    Решает систему Au = f методом регуляризации Тихонова.

    Параметры:
        A : numpy.ndarray
            Матрица коэффициентов системы (m x n).
        f : numpy.ndarray
            Вектор правой части системы (m,).
        alpha : float, optional
            Параметр регуляризации (по умолчанию 0.1).

    Возвращает:
        u : numpy.ndarray
            Регуляризованное решение системы (n,).
    """
    # Шаг 1: Проверка размеров
    m, n = A.shape
    if len(f) != m:
        raise ValueError("Размеры A и f не согласованы!")

    # Шаг 2: Вычисляем A^T A и A^T f
    ATA = A.T @ A
    ATf = A.T @ f

    # Шаг 3: Добавляем регуляризацию (A^T A + αI)
    regularized_matrix = ATA + alpha * np.eye(n)

    # Шаг 4: Решаем систему (A^T A + αI)u = A^T f
    try:
        u = np.linalg.solve(regularized_matrix, ATf)
    except np.linalg.LinAlgError:
        # Если система вырождена, используем псевдообратную матрицу
        u = np.linalg.pinv(regularized_matrix) @ ATf

    return u


# Пример использования
if __name__ == "__main__":
    # Пример 1: Хорошо обусловленная система
    A1 = np.array([
        [2, -1],
        [1, 3]
    ])
    f1 = np.array([1, 2])

    # Пример 2: Плохо обусловленная система
    A2 = np.array([
        [1, 1],
        [1, 1.0001]
    ])
    f2 = np.array([2, 2.0001])

    # Пример 3: Недоопределенная система (2 уравнения, 3 неизвестных)
    A3 = np.array([
        [1, 1, 1],
        [1, -1, 1]
    ])
    f3 = np.array([1, 0])

    # Решаем системы
    alpha = 0.1  # Параметр регуляризации

    for i, (A, f) in enumerate(zip([A1, A2, A3], [f1, f2, f3]), 1):
        print(f"\nПример {i}:")
        print("Матрица A:\n", A)
        print("Вектор f:", f)

        # Решение без регуляризации (если возможно)
        try:
            u_exact = np.linalg.solve(A, f)
            print("\nТочное решение (без регуляризации):", u_exact)
        except np.linalg.LinAlgError:
            print("\nТочное решение невозможно (система вырождена или недоопределена)")

        # Решение с регуляризацией Тихонова
        u_tikhonov = solve_tikhonov(A, f, alpha)
        print(f"Решение Тихонова (α={alpha}):", u_tikhonov)

        # Вычисляем невязку
        residual = np.linalg.norm(A @ u_tikhonov - f)
        print(f"Невязка: {residual:.6f}")
