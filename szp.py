import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import correlate2d
from matplotlib.widgets import Button

# Параметры моделирования
wavelength = 266e-9  # Длина волны (м)
f = 2.5  # Фокусное расстояние (м)
propagation_distance = 2.5  # Длина распространения (м)
l = 3  # Топологический заряд
w0 = 5e-3  # Радиус пучка (м)
N = 2048  # Размер сетки
size = 25e-3  # Размер области моделирования (м)
f_lens = 0.1  # Фокусное расстояние линзы (м)
propagation_distance_after_lens = 2 * f_lens  # Расстояние распространения после линзы (м)

# Создание координатной сетки
x = np.linspace(-size/2, size/2, N)
y = np.linspace(-size/2, size/2, N)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)

# Создание гауссова пучка
gaussian = np.exp(-(r**2)/(w0**2))

# Генерация масок SZP
arg = l * phi - (np.pi * r**2) / (wavelength * f)
phase_mask = np.exp(1j * arg)  # Фаза от 0 до 2π
amplitude_mask = np.where(np.sin(arg) >= 0, 1, 0)

# Функция распространения Френеля
def fresnel_propagate(field, dx, wavelength, z):
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(N, dx)
    fy = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    field_fft = np.fft.fft2(field)
    field_focal = np.fft.ifft2(field_fft * H)
    x_focal = fx * wavelength * z
    y_focal = fy * wavelength * z
    return x_focal, y_focal, field_focal

# Дифракция на SZP
field_amp = gaussian * amplitude_mask
field_phase = gaussian * phase_mask

x_focal, y_focal, diffracted_amp = fresnel_propagate(field_amp, size/N, wavelength, propagation_distance)
_, _, diffracted_phase = fresnel_propagate(field_phase, size/N, wavelength, propagation_distance)

# Дифракция на цилиндрической линзе
def cylindrical_lens_fresnel(field, dx, wavelength, f_lens, z):
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(N, dx)
    fy = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fy)

    # Цилиндрическая фаза (только по одной координате)
    H_lens = np.exp(-1j * k * (Y**2 / (2 * f_lens)))  # Ось X или Y

    # Френелевская передача в свободном пространстве
    H_prop = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))

    field_lens = field * H_lens  # Применяем линзу
    field_fft = np.fft.fft2(field_lens)  # Фурье-преобразование
    field_focal = np.fft.ifft2(field_fft * H_prop)  # Обратное Фурье с фазовым множителем

    return field_focal

# Дифракция после первой линзы
field_after_lens_amp = cylindrical_lens_fresnel(diffracted_amp, size/N, wavelength, f_lens, propagation_distance_after_lens)
field_after_lens_phase = cylindrical_lens_fresnel(diffracted_phase, size/N, wavelength, f_lens, propagation_distance_after_lens)

# Дифракция после второй линзы (конвертера)
field_after_converter_amp = cylindrical_lens_fresnel(field_after_lens_amp, size/N, wavelength, f_lens, propagation_distance_after_lens)
field_after_converter_phase = cylindrical_lens_fresnel(field_after_lens_phase, size/N, wavelength, f_lens, propagation_distance_after_lens)

# Интенсивности с логарифмической нормализацией
intensity_amp = np.log(1 + np.abs(diffracted_amp)**2 / np.max(np.abs(diffracted_amp)**2))
intensity_phase = np.log(1 + np.abs(diffracted_phase)**2 / np.max(np.abs(diffracted_phase)**2))

intensity_after_lens_amp = np.log(1 + np.abs(field_after_lens_amp)**2 / np.max(np.abs(field_after_lens_amp)**2))
intensity_after_lens_phase = np.log(1 + np.abs(field_after_lens_phase)**2 / np.max(np.abs(field_after_lens_phase)**2))

intensity_after_converter_amp = np.log(1 + np.abs(field_after_converter_amp)**2 / np.max(np.abs(field_after_converter_amp)**2))
intensity_after_converter_phase = np.log(1 + np.abs(field_after_converter_phase)**2 / np.max(np.abs(field_after_converter_phase)**2))

# Фаза после дифракции на SZP
phase_after_diffracted_amp = np.angle(diffracted_amp)
phase_after_diffracted_phase = np.angle(diffracted_phase)

# Визуализация
fig, ax = plt.subplots(4, 3, figsize=(10, 10))
fig.suptitle('Визуализация Гауссова пучка и дифракции', fontsize=16)

# Верхний ряд: Гауссов пучок и две маски
im0 = ax[0, 0].imshow(gaussian, cmap='plasma', extent=[-size/2e-3, size/2e-3, -size/2e-3, size/2e-3], origin='lower')
ax[0, 0].set_title('Исходный гауссов пучок')
plt.colorbar(im0, ax=ax[0, 0])

im1 = ax[0, 1].imshow(amplitude_mask, cmap='gray', extent=[-size/2e-3, size/2e-3, -size/2e-3, size/2e-3], origin='lower')
ax[0, 1].set_title('Амплитудная SZP')
plt.colorbar(im1, ax=ax[0, 1])
plt.imsave("Маска_амплитудная.png", amplitude_mask, cmap='gray', origin='lower', dpi=1200)

im2 = ax[0, 2].imshow(np.angle(phase_mask), cmap='twilight', extent=[-size/2e-3, size/2e-3, -size/2e-3, size/2e-3], origin='lower')
ax[0, 2].set_title('Фазовая SZP')
plt.colorbar(im2, ax=ax[0, 2])

# Второй ряд: Дифракция на SZP
im3 = ax[1, 1].imshow(intensity_amp, cmap='plasma', extent=[-size/2e-3, size/2e-3, -size/2e-3, size/2e-3], origin='lower')
ax[1, 1].set_title('Дифракция (амплитудная) на SZP')
plt.colorbar(im3, ax=ax[1, 1])

im4 = ax[1, 2].imshow(intensity_phase, cmap='plasma', extent=[-size/2e-3, size/2e-3, -size/2e-3, size/2e-3], origin='lower')
ax[1, 2].set_title('Дифракция (фазовая) на SZP')
plt.colorbar(im4, ax=ax[1, 2])

ax[1,0].axis('off')

# Третий ряд: Дифракция после конвертера
im5 = ax[2, 1].imshow(intensity_after_converter_amp, cmap='plasma', extent=[-size/2e-3, size/2e-3, -size/2e-3, size/2e-3], origin='lower')
ax[2, 1].set_title('Дифракция (амплитудная) после конвертера')
plt.colorbar(im5, ax=ax[2, 1])

im6 = ax[2, 2].imshow(intensity_after_converter_phase, cmap='plasma', extent=[-size/2e-3, size/2e-3, -size/2e-3, size/2e-3], origin='lower')
ax[2, 2].set_title('Дифракция (фазовая) после конвертера')
plt.colorbar(im6, ax=ax[2, 2])

# Графики для фазы
im7 = ax[3, 1].imshow(phase_after_diffracted_amp, cmap='twilight', extent=[-size/2e-3, size/2e-3, -size/2e-3, size/2e-3], origin='lower')
ax[3, 1].set_title('Фаза после дифракции (амплитудная) на SZP')
plt.colorbar(im7, ax=ax[3, 1])

im8 = ax[3, 2].imshow(phase_after_diffracted_phase, cmap='twilight', extent=[-size/2e-3, size/2e-3, -size/2e-3, size/2e-3], origin='lower')
ax[3, 2].set_title('Фаза после дифракции (фазовая) на SZP')
plt.colorbar(im8, ax=ax[3, 2])


# Пустая ось, нужно убрать (костыль)
ax[2, 0].axis('off')
ax[3,0].axis('off')

# Настройка ползунка зума
ax_zoom = plt.axes([0.05, 0.05, 0.2, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_zoom, 'Zoom', 1, 20, valinit=1, valstep=0.1)

def update(val):
    zoom_factor = slider.val
    new_size = size / zoom_factor
    ax[1, 1].set_xlim([-new_size/2e-3, new_size/2e-3])
    ax[1, 1].set_ylim([-new_size/2e-3, new_size/2e-3])
    ax[1, 2].set_xlim([-new_size/2e-3, new_size/2e-3])
    ax[1, 2].set_ylim([-new_size/2e-3, new_size/2e-3])
    ax[2, 1].set_xlim([-new_size/2e-3, new_size/2e-3])
    ax[2, 1].set_ylim([-new_size/2e-3, new_size/2e-3])
    ax[2, 2].set_xlim([-new_size/2e-3, new_size/2e-3])
    ax[2, 2].set_ylim([-new_size/2e-3, new_size/2e-3])
    ax[3, 1].set_xlim([-new_size/2e-3, new_size/2e-3])
    ax[3, 1].set_ylim([-new_size/2e-3, new_size/2e-3])
    ax[3, 2].set_xlim([-new_size/2e-3, new_size/2e-3])
    ax[3, 2].set_ylim([-new_size/2e-3, new_size/2e-3])
    fig.canvas.draw_idle()

def save_zoomed_images(event):
    zoom_factor = slider.val
    new_size = size / zoom_factor
    xlim = [-new_size/2e-3, new_size/2e-3]
    ylim = [-new_size/2e-3, new_size/2e-3]

    # Создаем временную фигуру для каждого графика
    def save_individual_plot(data, cmap, filename, extent, xlim, ylim):
        fig_save, ax_save = plt.subplots(figsize=(8, 8))
        im_save = ax_save.imshow(
            data,
            cmap=cmap,
            extent=extent,
            origin='lower',
            interpolation='bicubic'  # Интерполяция для сглаживания
        )
        ax_save.set_xlim(xlim)
        ax_save.set_ylim(ylim)
        fig_save.tight_layout()
        fig_save.savefig(filename, dpi=600)
        plt.close(fig_save)

    # Общие параметры extent
    extent = [-size/2e-3, size/2e-3, -size/2e-3, size/2e-3]

    try:
        # Дифракция на SZP (амплитудная)
        save_individual_plot(intensity_amp, "plasma", "diffracted_amp.png", extent, xlim, ylim)

        # Дифракция на SZP (фазовая)
        save_individual_plot(intensity_phase, "plasma", "diffracted_phase.png", extent, xlim, ylim)

        # Дифракция после конвертера (амплитудная)
        save_individual_plot(intensity_after_converter_amp, "plasma", "after_converter_amp.png", extent, xlim, ylim)

        # Дифракция после конвертера (фазовая)
        save_individual_plot(intensity_after_converter_phase, "plasma", "after_converter_phase.png", extent, xlim, ylim)

        # Фаза после дифракции на SZP (амплитудная)
        save_individual_plot(phase_after_diffracted_amp, "twilight", "phase_after_diffracted_amp.png", extent, xlim, ylim)

        # Фаза после дифракции на SZP (фазовая)
        save_individual_plot(phase_after_diffracted_phase, "twilight", "phase_after_diffracted_phase.png", extent, xlim, ylim)

        print("Изображения успешно сохранены!")
    except Exception as e:
        print(f"Ошибка при сохранении изображений: {e}")

slider.on_changed(update)

ax_button = plt.axes([0.05, 0.15, 0.2, 0.03])
button = Button(ax_button, 'Save Zoomed')
button.on_clicked(save_zoomed_images)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()