"""
Генерация входного датасета из набора сырых данных
"""


import os
import json
import glob
import pandas as pd


def extract_screw_data(root_dir='/Users/macbook/Desktop/screws'):
    """
    Обходит папки screw_* в root_dir, извлекает геометрические и гидродинамические параметры.
    Возвращает pandas DataFrame.
    """
    data_rows = []
    # Ищем все подпапки, начинающиеся с screw_
    screw_dirs = glob.glob(os.path.join(root_dir, 'screw_*'))

    for screw_path in screw_dirs:
        # Ищем файлы context_*.json и geom_context_*.json в папке
        context_files = glob.glob(os.path.join(screw_path, 'context_*.json'))
        geom_files = glob.glob(os.path.join(screw_path, 'geom_context_*.json'))

        if not context_files or not geom_files:
            print(f"Пропущено: {screw_path} – отсутствуют нужные JSON файлы")
            continue

        # Берём первый найденный (предполагается, что он один)
        context_file = context_files[0]
        geom_file = geom_files[0]

        try:
            with open(context_file, 'r') as f:
                context = json.load(f)
            with open(geom_file, 'r') as f:
                geom_list = json.load(f)
                # geom_list — это список, обычно из одного элемента
                geom = geom_list[0] if geom_list else {}
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Ошибка чтения {screw_path}: {e}")
            continue

        # Извлекаем нужные поля
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
        # Проверяем, что все поля есть (не None)
        if None in row.values():
            print(f"Предупреждение: в {screw_path} не хватает некоторых полей")
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    return df


if __name__ == "__main__":
    df = extract_screw_data()
    if not df.empty:
        # Сохраняем CSV с индексом (первая пустая колонка, как в примере)
        df.to_csv('screws_dataset.csv', index=True, index_label='', encoding='utf-8')
        print(f"Датасет сохранён в 'screws_dataset.csv'. Всего записей: {len(df)}")
    else:
        print("Не найдено ни одного корректного винта.")