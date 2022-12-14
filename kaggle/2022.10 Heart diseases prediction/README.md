# Чемпионат "Heart diseases prediction"

Ссылка: https://www.kaggle.com/competitions/yap10-heart-diseases-predictions

## Результат участия

2 место из 10

## Общее впечатление

Второе соревнование. Понял в чем была моя главная ошибка в первом соревновании - отсутствие базовой модели и обработка данных без оценки влияния обработки на базовую модель.

Данное соревнование учебное, данных мало и они практически не нуждаются в обработке. Простым удалением выбросов и несложной генерацией новых признаков я существенно ухудшил результаты модели (catboost). На почти сырых данных катбуст показал себя лучше всего (как и в первом соревновании, совпадение ли?).


## Выводы и мысли

ВАЖНО: соревнование учебное, данные чистые.

1. Необходимо делать базовую модель.
1. Катбуст слишком крут. Его лучше не портить своими потугами обработать данные, он сам выжмет из них максимум. Возможно, стоит относиться к нему, как к AutoML.
1. В следующий раз нужно посмотреть побольше моделей. Победитель использовал ансамбль из бустингов. Мои ансамбли из разных типов моделей лишь портили катбуст, а не диверсифицировали предсказание.
1. Необходимо рассматривать поведение модели с различными random seed.
1. В следующий раз нужно попробовать перенести побольше кода в файлы, а блокноты использовать только для анализа результатов. Я и так работаю через буферы в виде .pickle файлов, никаких проблем переехать на новый подход быть не должно. Плюсом, в таком случае удобно пускать расчеты в консоли/докере, а значит можно будет работать не только на своем компе, но и на любом, который смогу найти)) Если подготовлю флешку с линуксом и нужными образами. В целом, это хорошее направление для саморазвития.
1. Тесты нужно проводить с помощью бутстрепа. Это позволяет неплохо побороть шум и оценить эффективность модели.
1. Нужно проводить анализ достаточности размера обучающей выборки.
1. Хорошо бы сделать некоторое универсальное решение, в котором можно было бы просто менять базовые модели. Само решение можно организовать так, как и решение для этого соренования, но сделать несколько папок для каждой модели.
1. Не стоит бездумно брать лучшие модели с теста (при оценке средней метрики при вариации размеров выборок / случайных зерен). Вполне вероятно, что они лучшие только на тестовой выборке.

Старые, подтвержденные, выводы и советы себе:
1. Хватит биться в сотые доли метрики. Лучше делать упор на идеи.
1. Лучшие решения, выглядят так, как будто были написаны и посчитаны за день. Без скидки на катбуст и его AutoML характер. Скорее всего для более серьезных соревнований картина совершенно другая.

## Описание

Нужно предсказать наличие болезней сердца по нескольким признакам. Задача бинарной классификации.

**Метрика**: ROC AUC
