import numpy as np
from scipy.integrate import solve_ivp as scipy_solve_ivp
import matplotlib.pyplot as plt
import sklearn.manifold as skl_manifold
import time
from math import pi


## Дифференциальные уравнения ##
def chua_circuit(t, state, alpha, beta, a, b):
    x, y, z = state

    f = lambda x: b * x + 0.5 * (a - b) * (abs(x + 1) - abs(x - 1))
        
    dx = alpha * (y - x - f(x))
    dy = x - y + z
    dz = -beta * y

    return [dx, dy, dz]


def lorenz_84(t, state, G, F, a, b):
    x, y, z = state

    dx = -a * x - y ** 2 - z ** 2 + a * F
    dy = -y + x * y - b * x * z + G
    dz = -z + b * x * y + x * z

    return [dx, dy, dz]


def modified_aa(t, state, g, m, d, gamma):
    x, y, z, phi = state

    f = lambda x: x ** 2 * int(x > 0)

    dx = m * x + y - x * phi - d * x ** 3
    dy = -x
    dz = phi
    d_phi = -gamma * phi + gamma * f(x) - g * z

    return [dx, dy, dz, d_phi]


def kks_generator_mod(t, state, lmbd, beta, omega, b, eps, k):
    x, y, z = state

    dx = y
    dy = (lmbd + z + x ** 2 - beta * x ** 4) * y - omega ** 2 * x
    dz = b * (eps - z) - k * y ** 2

    return [dx, dy, dz]
## -------------- ##


## Методы понижения размерности ##
DRM_NUM = 4
dim_red_methods = {
    'Isomap': skl_manifold.Isomap(),
    'Laplacian Eigenmap': skl_manifold.SpectralEmbedding(),
    'LLE': skl_manifold.LocallyLinearEmbedding(),
    't-SNE': skl_manifold.TSNE(),
    }
## -------------- ##


## Системы, их параметры и начальные условия ##
P_NUM = 6
systems = {
    'chua': {
        'name': 'Система Чуа',
        'filename': 'chua_circuit',
        'func': chua_circuit,
        'param_names': ['$\\alpha$', '$\\beta$', 'a', 'b'],
        'params': [
            [9.4, 14 + 2/7, -8/7, -5/7],
            [9.85, 14.3, -8/7, -5/7],
            [8.5, 14.3, -8/7, -5/7],
            [9.11, 16, -8/7, -5/7],
            [10, 11, -8/7, -5/7],
            [4, 14.3, -8/7, -5/7],
        ],
        'init_values': [-1.5, 0, 0],
        'time': [0, 60, 0.01, 10],
        '3d': True,
    },
           
    'lorenz': {
        'name': 'Система Лоренц-84',
        'filename': 'lorenz-84',
        'func': lorenz_84,
        'param_names': ['G', 'F', 'a', 'b'],
        'params': [
            [1, 8, 0.14, 4],
            [1, 5, 0.25, 4],
            [1, 8, 0.25, 2],
            [1, 8, 0.25, 1.5],
            [0.3, 8, 0.25, 4],
            [0.799, 8, 0.25, 4],
        ],
        'init_values': [1, 1, 1],
        'time': [0, 125, 0.01, 75],
        '3d': True,
    },
           
    'modified_aa': {
        'name': 'Модифицированный аттрактор Анищенко-Астахова',
        'filename': 'modified-aa',
        'func': modified_aa,
        'param_names': ['g', 'm', 'd', '$\\gamma$'],
        'params': [
            [0.25, 0.129, 0.001, 0.2],
            [0.036, 0.027, 0.001, 0.2],
            [0.25, 0.115, 0.001, 0.2],
            [0.5, 0.065, 0.001, 0.2],
            [0.5, 0.067, 0.001, 0.2],
            [0.5, 0.075, 0.001, 0.2],
        ],
        'init_values': [-1.5, 0, 0, 0],
        'time': [0, 620, 0.01, 500],
        '3d': False,
    },

    'kks_generator': {
        'name': 'Модификация генератора Кузнецова-Кузнецова-Станкевич',
        'filename': 'kks-generator_mod',
        'func': kks_generator_mod,
        'param_names': ['$\\lambda$', '$\\beta$', '$\\omega$', 'b', '$\\epsilon$', 'k'],
        'params': [
            [2, 1/18, 2 * pi, 1, 4, 0.02],
            [8, 1/18, 3.1 * pi, 1, 4, 0.02],
            [2, 1/18, 3 * pi, 1, 4, 0.02],
            [8, 1/18, 4 * pi, 1, 4, 0.02],
            [2, 1/18, 2.5 * pi, 1, 4, 0.02],
            [2, 1/18, 3.1 * pi, 1, 4, 0.02],
        ],
        'init_values': [-1.5, 0, 0],
        'time': [0, 40, 0.01, 20],
        '3d': True,
    },
}
## -------------- ##


## Выбор анализируемой системы и промежутка времени ##
curr_system = systems['chua']

params = curr_system['params']
initial_values = curr_system['init_values']
t_start, t_end, dt, t_cut = curr_system['time']
t_array = np.arange(t_cut, t_end, dt)
## -------------- ##


## Расчёт параметров для расположения графиков ##
plot_side_pix = 200
space = 0.2
left_pix = top_pix = 180
right_pix = bottom_pix = 60
dpi = 100

width_inch = (left_pix / dpi) +  \
             (right_pix / dpi) +  \
             (plot_side_pix / dpi) * (DRM_NUM + 1) +  \
             (plot_side_pix * space / dpi) * (DRM_NUM + 1 - 1)

height_inch = (top_pix / dpi) +  \
              (bottom_pix / dpi) +  \
              (plot_side_pix / dpi) * P_NUM +  \
              (plot_side_pix * space / dpi) * (P_NUM - 1)

left = (left_pix / dpi) / width_inch
right = 1 - (right_pix / dpi) / width_inch
bottom = (bottom_pix / dpi) / height_inch
top = 1 - (top_pix / dpi) / height_inch
## -------------- ##


## Создание матрицы для графиков ##
fig, axes = plt.subplots(P_NUM, DRM_NUM + 1, subplot_kw={'xticks': [], 'yticks': []}, figsize=(width_inch, height_inch))
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=space, hspace=space)
plt.suptitle(f"{curr_system['name']}, начальные значения: {tuple(initial_values)}, промежуток времени и шаг: ({t_cut}, {t_end}, {dt})", fontsize='xx-large', wrap=True)
for num, method in enumerate(list(dim_red_methods.keys()) + ['3D-график' if curr_system['3d'] else 'График (x, y, z)']):
    fig.text(left + ((1 + space) * plot_side_pix * num) / dpi / width_inch,  \
             top + (1 - top) * 0.25,  \
             method,  \
             fontsize='x-large')
## -------------- ##


print(f"{curr_system['name']}: начальные значения: {initial_values}, промежуток времени и шаг: {[t_cut, t_end, dt]}")
tmp_num = str(time.time_ns())

for params_num in range(P_NUM):
    print(f'    Параметры: {params[params_num]}')
    fig.text(left * 0.20,  \
             top - (0.75 * plot_side_pix + (1 + space) * plot_side_pix * params_num) / dpi / height_inch,  \
             '\n'.join(map(lambda param: f'{param[0]}: {param[1]:0.3f}...', zip(curr_system['param_names'], params[params_num]))),  \
             fontsize='x-large')
    
    t1 = time.time_ns()
    solution = scipy_solve_ivp(curr_system['func'], (t_start, t_end), initial_values, args=params[params_num], t_eval=t_array)
    t2 = time.time_ns()
    print(f'        Вычисления завершены, потребовалось {(t2 - t1) / 1000000000 :.3f} секунд')

    # Создание трёхмерного графика и сохранение временного изображения
    axes[params_num][-1] = fig.add_subplot(P_NUM, DRM_NUM + 1, (DRM_NUM + 1) * (params_num + 1), projection = '3d', xticks=[], yticks=[], zticks=[])
    axes[params_num][-1].plot(solution.y[0, :], solution.y[1, :], solution.y[2, :], color=(0.14, 0, 0.5, 1))
    plt.savefig(f"tmp_{curr_system['filename']}.png")
    print(f'        3D-график построен')
    
    for method_num, method_name in enumerate(dim_red_methods):
        try:
            t1 = time.time_ns()
            after_dim_red = dim_red_methods[method_name].fit_transform(solution.y.T)  # применение метода понижения размерности
            t2 = time.time_ns()
        except Exception:
            axes[params_num][method_num].text(0.5, 0.5, 'Error')
            plt.savefig(f"tmp_{curr_system['filename']}.png")
            print(f'        {method_name} не применён, произошла ошибка')
        else:
            axes[params_num][method_num].scatter(after_dim_red[:, 0], after_dim_red[:, 1], color=(0.14, 0, 0.5, 1), s=0.2)  # вставка графика в ячейку матрицы графиков
            axes[params_num][method_num].set_title(f'{(t2 - t1) / 1000000000 :.3f} c')
            plt.savefig(f"tmp_{curr_system['filename']}.png")
            print(f'        {method_name} применён, потребовалось {(t2 - t1) / 1000000000 :.3f} секунд')
            

plt.savefig(f"{curr_system['filename']}_{tmp_num}.png")
print(f"{curr_system['name']}: расчёты завершены")
