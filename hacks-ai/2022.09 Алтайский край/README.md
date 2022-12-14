# Чемпионат "Прогнозирование статуса студента"

Ссылка: https://hacks-ai.ru/championships/758263

## Результат участия

60 место из 128

## Общее впечатление

Первое соревнование и полный провал. После первой-второй недели работы возникло ощущение, что работаю не в том направлении. Понять в чем проблема и куда двигаться не смог до самого конца.

## Анализ победителей

<details>
  <summary><a href="https://github.com/IGragon/RSV_Altai/blob/main/RSV_Altai_Abramov(1).ipynb">1 место</a></summary>

  #### Общее впечатление
  
  Работа выглядит так, будто кто-то за пару минут накидал блокнот для оценки задачи, не для ее решения. Не спорю, что могла быть проведена огромная работа где-то за пределами этого блокнота, но результатов этой работы в этом блокноте не видно. Возможно автору повезло настроить катбуст так, что он сходу на почти сырых данных дал отличную метрику.

  #### +
  
  1. Более подробный анализ учебных заведений (через список сокращений). Я такой список делал вручную и, скорее всего, он вышел неточным.
  1. Генерация аббревиатур из длинных текстов для дальнейшего анализа по сокращениям. Гениально!
  1. Самостоятельный расчет коэффициентов для балансировки классов.
  1. Затюнены некоторые интересные параметры катбуста.
  
  #### +-
  
  1. Использование кириллицы в названиях столбцов. Читать удобно, а печатать код неудобно, отсюда вывод, что для коротких презентационных отчетов нужно использовать кириллицу и понятные названия, для подкапотной работы - латиницу.
  1. Разбиение кодов факультетов на отдельные числа.
  1. Замена отсутствующих значений медианой. Насколько это эффективно? Я пытался восстановить значения.

  #### -
  
  1. Отсутствие комментариев и неструктурированный отчет. Сам этим грешу, теперь будет мотивация следить за этим и в своей работе.
  1. Неподавление вывода технической информации.
  1. Обработка данных для теста простой копипастой кода.
  
</details>

<details>
  <summary><a href="https://github.com/nikita-p/hacksai/blob/master/notebook.ipynb">2 место</a></summary>

  #### Общее впечатление
  
  Работа сделана хорошо, однако рецепт как и у первого места: взять сырые данные, слегка обработать, запустить на них катбуст с подобранными параметрами. Не понимаю, как такой подход может быть настолько результативен.
  

  #### +
  
  1. Порядок. Насколько же проще и приятнее анализировать. Надо взять на заметку.
  1. Анализ изменения метрики в зависимости от количества итераций катбуста.
  1. Pool из библиотеки катбуста.
  
  #### +-
  
  1. Код факультета без изменений.
  1. Ручные веса классов.
  1. Мало параметров катбуста.

  #### -
  
  1. Хардкодинг параметров. Может быть это говорит о том, что код не рефакторился и хороший результат был получен быстро без затрачивания огромных сил и средств на доведение модели?
  1. Вывод информации о параметрах катбуста в строку.
  
</details>

## Выводы и мысли

1. Работу лучше начать с поиска самых важных признаков и попытаться выжать из них по максимуму. В данном случае предсказание на **нескольких** признаках давало 95% результата. Возможно именно поэтому первые места заняли обычные катбусты на почти сырых данных.
2. Если среди признаков есть золотые, то не имеет смысла ковыряться в данных. Нужна специальная стратегия работы с золотыми признаками.
3. Очень важно провести анализ данных для предсказания на закрытой выборке. Если там есть смещение, то это нужно учесть.
4. Катбуст - серебрянная пуля.
5. Нужно пробовать как можно больше различных моделей.
6. Нужно отводить время проверке идей, а не биться лбом в сотые доли метрики.
7. Нужно не забывать подбирать гиперпараметры. Во время этой работы потратил несколько часов за пару дней до конца. Правды ради, толку это не дало: на обучающей выборке метрика улетела в небеса, на моей тестовой выросла на пару десятых, а на закрытой тестовой упала! Хотя у победителей результат получен именно за счет подбора гиперпараметров.
8. Скорее всего я сильно переобучил модели.
9. Люди, активно участвующие в соревнованиях, имеют кучу свободного времени. Победители же фигачят решения, которые можно сгенерировать за пару часов.

## Описание

По данным Минобрнауки РФ, в этом году более 6,5 млн. человек подали заявление на зачисление. В 2022 г. количество бюджетных мест увеличилось до 588 044, большинство из них — в региональных университетах.

Высшее образование имеет большое значение для формирования карьерной траектории. На сегодняшний день в России около 700 вузов предоставляют возможность обучения по образовательным программам в различных направлениях. После школы или колледжа молодые люди выбирают, в какой сфере хотят развиваться. И, конечно, каждый будущий студент хочет получить действительно качественное и разностороннее образование.

Ежегодно тысячи абитуриентов в нашей стране участвуют в приемной кампании, выбирая себе будущую профессию или дополнительные знания и навыки, чтобы повысить свои профессиональные компетенции и стать востребованным специалистом. Но все ли студенты заканчивают университет? Все ли остаются в том направлении, которое выбрали изначально? Сколько студентов берет академический отпуск, и какая часть обучающихся решает поступать в магистратуру и аспирантуру?

Перед участниками чемпионата стоит задача предсказать будущий потенциальный статус студента на основе данных нескольких тысяч студентов. Такая прогнозная модель может стать хорошим инструментом при планировании учебной работы в вузах, взаимодействии с работодателями или реализации научных проектов.

## Условие задачи

На основе данных об абитуриенте участникам чемпионата необходимо разработать модель, которая будет предсказывать текущий статус студента, а именно:

- продолжит ли студент обучение
- отчислится
- возьмет академический отпус

**Метрика**: F1
