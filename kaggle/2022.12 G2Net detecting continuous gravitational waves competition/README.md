# <a href="[https://codenrock.com/contests/turbohackaton#/info](https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves)">G2Net Detecting Continuous Gravitational Waves</a>

## Результат участия

Не загружал решение.

## Общее впечатление

Очень интересное соревнование. Задача весьма необычная и для своего решения требует не только разносторонних знаний, но и дает огромнейший простор для фантазии.

Очень порадовало, что для того, чтобы просто вникнуть в описание и условия задачи пришлось целый вечер читать обсуждения на каггле. Это была не та задачка, когда на сырых данных пускаешь катбуст и получаешь место в первой тройке.

Основной особенностью соревнования было то, что данные для обучения моделей нужно было сгенерировать самим. Я начал участвовать в соревновании за месяц до конца, времени на ошибки не было.

Данные для обучения можно было сгенерировать только на macos или на linux - для меня это большая проблема. Единственное место где я мог их получить становился колаб, так как его можно подключить к гугл диску и туда писать эти данные. Также проблемой была их неоптимальная генерация - они сразу писались на диск. Чтобы места на гугл диске хватало приходилось поступать следующим образом: писать на гугл диск из колаба специальной функцией для генерации данных (предоставлена организаторами), загружать данные в колаб, обрабатывать, удалять из данных ненужное, считать одну полезную фичу и только потом сохранять в нужном формате на гугл диск. Это приводило к тому, что за сутки генерировалось несколько сотен спектрограмм и это без учета того, что колаб частенько останавливал блокнот. Минимальное число спектрограмм для обучения я оценивал в 5-10 тысяч. Требовались меры.

Я принял решение генерировать спектрограммы другого размера, но с теми же параметрами. После оптимизации процесса, за несколько дней я сгенерировал 60 000 спектрограмм. Параллельно подготовил модели, которые прекрасно работали на этих спектрограммах.

Однако, когда я перешел к предсказанию на тестовых данных, обнаружил, что сгенерированные данные отличаются от тестовых спектрограмм по статистикам (при этом по словам организаторов, тестовые данные также были сгенерированы в большинстве своем теми же инструментами). Я потратил несколько дней в попытках приспособить модели к обучению на одних данных, а работе на других, однако должного эффекта это не возымело. Времени оставалось мало, я решил отказаться от участия.

## Выводы и мысли

В целом нечего сказать. Было интересно, но я не дошел до конца, чтобы можно было как-нибудь оценить тот опыт, что получил.

## Описание

По спектрограммам сигналов с гравитационных детекторов нужно определить есть ли в них гравитационные волны.