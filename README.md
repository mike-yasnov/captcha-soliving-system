# Решение каптчи

## Запуск
Создайте виртуальное окружение и установите зависимости 
```bash 
python -m venv venv 
pip install -r requirements
```
Для запуска кода
```
python main.py --data_dir path-to-dir
```

## Описание работы решения
В данном решении используется две модели: YoloV8m для детекции иконок и цифр, Vit-base-patch16-224 для экстракции эмбедингов 

1. Изначально в yolo модель передается изображение 400х200 (в него входит темплейт и первая каптча)
2. Полученные bbox'ы сопоставляются орбиталям и достаются иконки с каждого изображения нужной орбитали 
3. Данные иконки сравниваются при помощи image similarity system (сравнение косинуснового расстояния эмбеддингов)