from sklearn.metrics import f1_score, roc_auc_score, make_scorer


THREADS = 2

PATH_TO_DATA = ''
TEST_DATA_FILE = r'test_dataset_test.csv'
TRAIN_DATA_FILE = r'train_dataset_train.csv'

COLS_RENAMING_DICT = {'ID': 'id', 'Код_группы': 'group_code', 'Год_Поступления': 'start_year',
                      'Пол': 'gender', 'Основания': 'condition', 'Изучаемый_Язык': 'language',
                      'Дата_Рождения': 'birthday', 'Уч_Заведение': 'school',
                      'Где_Находится_УЗ': 'school_location', 'Год_Окончания_УЗ': 'school_finish_year',
                      'Пособие': 'pension', 'Страна_ПП': 'country', 'Регион_ПП': 'region',
                      'Город_ПП': 'city', 'Общежитие': 'community', 'Наличие_Матери': 'has_mother',
                      'Наличие_Отца': 'has_father', 'Страна_Родители': 'relativies_country',
                      'Опекунство': 'guardianship', 'Село': 'countryside', 'Иностранец': 'foreign',
                      'КодФакультета': 'faculty', 'СрБаллАттестата': 'mean_mark', 'Статус': 'status'}
TARGET_FEATURE = COLS_RENAMING_DICT['Статус']

TARGET_REPLACER = {4: 0, 3: 1, -1: 2}
INVERSE_TARGET_REPLACER = {y: x for x, y in TARGET_REPLACER.items()}

TEST_SIZE = [0.01 for _ in range(40)]
SCORERS = [make_scorer(f1_score, average='macro'), make_scorer(roc_auc_score, needs_proba=True, average='micro', multi_class='ovr')]

RANDOM_SEED = 0