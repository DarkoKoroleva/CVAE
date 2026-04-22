"""
Генерация входного датасета из набора сырых данных
"""


import os
import json
import glob
from typing import Union, List
import pandas as pd


def extract_screw_data(root_dirs: Union[str, List[str]]) -> pd.DataFrame:
    """
    Обходит папки screw_* внутри каждой из указанных корневых директорий,
    извлекает геометрические и гидродинамические параметры.

    Parameters:
    root_dirs (str or list): путь к одной папке или список путей

    Returns:
    pd.DataFrame: объединённые данные из всех найденных screw_* папок
    """
    # Если передан один путь, превращаем его в список
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    data_rows = []

    for root_dir in root_dirs:
        screw_dirs = glob.glob(os.path.join(root_dir, 'screw_*'))

        for screw_path in screw_dirs:
            context_files = glob.glob(os.path.join(screw_path, 'context_*.json'))
            geom_files = glob.glob(os.path.join(screw_path, 'geom_context_*.json'))

            if not context_files or not geom_files:
                print(f"Пропущено: {screw_path} – отсутствуют нужные JSON файлы")
                continue

            context_file = context_files[0]
            geom_file = geom_files[0]

            try:
                with open(context_file, 'r') as f:
                    context = json.load(f)
                with open(geom_file, 'r') as f:
                    geom_list = json.load(f)
                    geom = geom_list[0] if geom_list else {}
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Ошибка чтения {screw_path}: {e}")
                continue

            row = {
                'z1': geom.get('z1'),
                'z2': geom.get('z2'),
                'A': geom.get('A'),
                'r1': geom.get('r1'),
                'r2': geom.get('r2'),
                'r': geom.get('r'),
                'r0': geom.get('r0'),
                'h': geom.get('h'),
                'L': geom.get('L'),
                'Q_theor': context.get('Q_theor'),
                'eps_theor': context.get('eps_theor'),
                'etha_theor': context.get('etha_theor')
            }

            if None in row.values():
                print(f"Предупреждение: в {screw_path} не хватает некоторых полей")

            data_rows.append(row)

    df = pd.DataFrame(data_rows)
    return df


if __name__ == "__main__":
    dirs = ['/Users/macbook/Desktop/screws2/20260420_172448', '/Users/macbook/Desktop/screws']
    df = extract_screw_data(dirs)
    if not df.empty:
        df.to_csv('screws_dataset.csv', index=True, index_label='', encoding='utf-8')
        print(f"Датасет сохранён в 'screws_dataset.csv'. Всего записей: {len(df)}")
    else:
        print("Не найдено ни одного корректного винта.")