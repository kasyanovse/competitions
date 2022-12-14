from math import nan

countries = {'рос': 'россия', 'казах': 'казахстан', 'кыргыз': 'ср азия', 'киргиз': 'ср азия',
             'рус': 'россия', 'таджик': 'ср азия', 'китай': 'китай',
             'армен': 'россия', 'туркм': 'ср азия', 'нигер': 'другая', 'узбек': 'ср азия', 'франц': 'другая',
             'монгол': 'другая', 'украин': 'другая', 'афган': 'ср азия', 'герман': 'другая',
             'молдов': 'другая', 'кнр': 'китай', 'чеченск': 'россия'}

countries_large = {'другая': nan, 'казахстан': 'казахстан', 'китай': 'китай', 'россия': 'россия', 'ср азия': 'ср азия',}

regions = {'акмолинская обл': 'с казахстан', 'акмолинская обл.': 'с казахстан', 'алайский край': 'алтайский к',
           'алматинская': 'юв казахстан', 'алматинская обл': 'юв казахстан', 'алматинская область': 'юв казахстан',
           'алтай респ': 'алтай', 'алтайски край': 'алтайский к', 'алтайский  край': 'алтайский к', 'алтайский карй': 'алтайский к',
           'алтайский кр': 'алтайский к', 'алтайский кра': 'алтайский к', 'алтайский край': 'алтайский к',
           'алтайский край ': 'алтайский к', 'алтайский край,': 'алтайский к', 'алтайский край, барнаул г': 'алтайский к',
           'алтайский крайай': 'алтайский к', 'алтайский края': 'алтайский к', 'алтайсктй край': 'алтайский к',
           'алтаский край': 'алтайский к', 'амурская область': 'амурская обл', 'андижанская обл': 'узбек',
           'аньхой провинция': 'китай', 'аньхуй провинция': 'китай', 'ао чукотский': 'чукотка', 'араратская обл.': 'армения',
           'баткенская обл': 'кыргыз', 'баткенская область': 'кыргыз', 'башкортостан респ': 'башкирия',
           'белгородская область': 'белгород', 'бурятия респ': 'бурятия', 'в-казахстанская': 'в казахстан',
           'в-казахстанская обл': 'в казахстан', 'в-казахстанская обл.': 'в казахстан', 'в-казахстанская область': 'в казахстан',
           'вко': 'в казахстан', 'волгоградская обл': 'волгоград', 'волгоградская область': 'волгоград',
           'воронежская обл': 'воронеж', 'восточно - казахстан обл.': 'в казахстан', 'восточно-казастанская обл.': 'в казахстан',
           'восточно-казахстанская': 'в казахстан', 'восточно-казахстанская обл': 'в казахстан',
           'восточно-казахстанская обл.': 'в казахстан', 'восточно-казахстанская область': 'в казахстан',
           'восточно-кзахстанская обл': 'в казахстан', 'г москва': 'москва', 'г санкт-петербург': 'спб', 'г. барнаул': 'алтайский к',
           'ганьсу провинция': 'китай', 'гбао': 'таджик', 'гиссарский р-н': 'таджик', 'горно бадахшанская ао': 'таджик',
           'горно-бадахшанская ао': 'таджик', 'горно-бадахшанская аобл': 'таджик', 'горно-бадахшанская обл': 'таджик',
           'гуандун': 'китай', 'гуандун пров.': 'китай', 'дагестан респ': 'дагестан', 'джалал-абадская': 'кыргыз',
           'джалал-абадская область': 'кыргыз', 'донецкая обл': 'донецк', 'жалал-абад обл': 'кыргыз',
           'жалал-абадская': 'кыргыз', 'жалал-абадская обл': 'кыргыз', 'жалал-абадская обл.': 'кыргыз',
           'жалал-абадская область': 'кыргыз', 'желал-абадская обл': 'кыргыз', 'забайкальский край': 'забайкальский к',
           'западно-казахстанская обл': 'з казахстан', 'иркутская обл': 'иркутск', 'иркутская область': 'иркутск',
           'иссык-кульская': 'кыргыз', 'иссык-кульская обл': 'кыргыз', 'иссык-кульская обл.': 'кыргыз',
           'казахстан респ': 'казахстан', 'казыбекбийский р-н': 'караганда', 'камчатский край': 'камачтка', 'карагандинская': 'караганда',
           'карагандинская обл': 'караганда', 'карагандинская обл.': 'караганда', 'кемеровская обл': 'кемерово',
           'кемеровская область': 'кемерово', 'коми респ': 'коми', 'костанайская обл': 'костанай', 'краснодарский край': 'краснодар',
           'красноярский край': 'красноярск', 'курганская область': 'курган', 'курская обл': 'курск', 'кызылординская область': 'кызылорда',
           'лебапская обл': 'туркмен', 'лебапская обл.': 'туркмен', 'ленинабадская обл': 'таджик', 'ленинградская обл': 'спб',
           'ляонин': 'китай', 'ляонин провинция': 'китай', 'магаданская обл': 'магадан', 'магаданская область': 'магадан',
           'мангистауская обл': 'юз казахстан', 'москва г': 'москва', 'московкая обл': 'мособл', 'московская обл': 'мособл',
           'московская область': 'мособл', 'мурманская область': 'мурманск', 'наронская': 'кыргыз', 'нарын': 'кыргыз',
           'нарынская': 'кыргыз', 'новосибирская обл': 'новосиб', 'новосибирская обл.': 'новосиб', 'новосибирская область': 'новосиб',
           'омская обл': 'омск', 'омская область': 'омск', 'оренбургская область': 'оренбург', 'ошская обл': 'кыргыз',
           'ошская область': 'кыргыз', 'павлодарская обл': 'павлодар', 'павлодарская обл.': 'павлодар', 'павлодарская область': 'павлодар',
           'пр.хэйлунцзян': 'китай', 'приморский край': 'приморский к', 'пров хэйлунцзян': 'китай', 'пров. ганьсу': 'китай',
           'пров. гуандун': 'китай', 'пров. ляолин': 'китай', 'пров. ляонин': 'китай', 'пров. синьцзян': 'китай', 'пров. хубэй': 'китай',
           'пров. хэбэй': 'китай', 'пров. хэйлунцзян': 'китай', 'пров. хэнань': 'китай', 'пров. цзилинь': 'китай', 'пров. цзянси': 'китай',
           'пров. шаньдун': 'китай', 'провинция ганьсу': 'китай', 'провинция ляонин': 'китай', 'провинция синьцзян': 'китай',
           'провинция сычуань': 'китай', 'провинция хайнань': 'китай', 'провинция хунань': 'китай', 'провинция хэйлунцзян': 'китай',
           'провинция цзилинь сыпин': 'китай', 'провинция цзунци': 'китай', 'провинция цзянсу': 'китай', 'респ алтай': 'алтай',
           'респ кыргыстан': 'кыргыз', 'респ саха (якутия)': 'якутия', 'республика алтай': 'алтай', 'республика башкортостан': 'башкирия',
           'республика бурятия': 'бурятия', 'республика крым': 'крым', 'республика саха /якутия/': 'якутия', 'республика тыва': 'тува',
           'республика хакасия': 'хакасия', 'рязанская': 'рязань', 'самарская область': 'самара', 'санкт-петербург г': 'спб',
           'саха (якутия) респ': 'якутия', 'саха /якутия/ респ': 'якутия', 'саха респ (якутия)': 'якутия', 'сахалинская обл': 'сахалин',
           'сахалинская область': 'сахалин', 'свердловская обл': 'екб', 'свердловская область': 'екб', 'северо-казахстанская обл': 'с казахстан',
           'синьцзян провинция': 'китай', 'сицуань провинция, цзань вэй уезд': 'китай', 'согдийская обл': 'таджик', 'согдийская обл.': 'таджик',
           'согдийская область': 'таджик', 'ставропольский край': 'ставрополь', 'сычуань': 'китай', 'сычуань провинция': 'китай',
           'талас': 'кыргыз', 'таласская обл': 'кыргыз', 'таласская обл.': 'кыргыз', 'ташкентская обл': 'узбек', 'тверская обл': 'тверь',
           'томская обл': 'томск', 'томская область': 'томск', 'тульская обл': 'тула', 'тыва респ': 'тува', 'тюменская обл': 'тюмень',
           'тюменская область': 'тюмень', 'удмуртская респ': 'удмуртия', 'уезд чжунсянь': 'китай', 'уйгурская ао': 'китай',
           'ферганская обл': 'узбек', 'хабаровский край': 'хабаровск', 'хайнань пров': 'китай', 'хайнань пров.': 'китай',
           'хакасия': 'хакасия', 'хакасия респ': 'хакасия', 'ханты-мансийский автономный округ - югра ао': 'хмао', 'ханты-мансийский ао': 'хмао',
           'ханты-мансийский ао - югра': 'хмао', 'хатлонская': 'таджик', 'хатлонская обл': 'таджик', 'хатлонская обл.': 'таджик',
           'хубэй провинция': 'китай', 'хукумата обл': 'таджик', 'хунань провинция': 'китай', 'хэ нань провинция': 'китай', 'хэбэй': 'китай',
           'хэбэй провинция': 'китай', 'хэйбэй провинция': 'китай', 'хэйлунцзян': 'китай', 'хэйлунцзян провинция': 'китай', 'хэнань провинция': 'китай',
           'цзилинь провинция': 'китай', 'цзянси пров.': 'китай', 'цзянсу': 'китай', 'цзянсу провинция': 'китай', 'чарей марастун': 'китай',
           'челябинская обл': 'челяб', 'челябинская область': 'челяб', 'чеченская респ': 'чечня', 'чеченская республика': 'чечня',
           'чжэцзян провинция': 'китай', 'чорякарон': 'таджик', 'чувашская респ': 'чувашия', 'чуй обл.': 'кыргыз', 'чуйская обл': 'кыргыз',
           'чуйская обл.': 'кыргыз', 'чуйская область': 'кыргыз', 'чукотский ао': 'чукотка', 'шаньдун': 'китай', 'шаньдун провинция': 'китай',
           'шохмансур': 'таджик', 'шэньси провинция': 'китай', 'ысык-кульская': 'кыргыз', 'южно-казахстанская область': 'ю казахстан',
           'ямало-ненецкий ао': 'янао', 'ярославская обл': 'ярославль',}

regions_large = {'амурская обл': 'россия', 'армения': 'россия', 'башкирия': 'россия', 'белгород': 'россия', 'бурятия': 'россия', 'волгоград': 'россия',
                 'воронеж': 'россия', 'дагестан': 'россия', 'донецк': 'россия', 'екб': 'россия', 'з казахстан': 'казахстан', 'забайкальский к': 'россия',
                 'иркутск': 'россия', 'казахстан': 'казахстан', 'камачтка': 'россия', 'караганда': 'казахстан', 'коми': 'россия', 'костанай': 'казахстан',
                 'краснодар': 'россия', 'красноярск': 'россия', 'крым': 'россия', 'курган': 'россия', 'курск': 'россия', 'кызылорда': 'казахстан',
                 'кыргыз': 'ср азия', 'магадан': 'россия', 'москва': 'россия', 'мособл': 'россия', 'мурманск': 'россия', 'омск': 'россия', 'оренбург': 'россия',
                 'павлодар': 'казахстан', 'приморский к': 'россия', 'рязань': 'россия', 'с казахстан': 'казахстан', 'самара': 'россия', 'сахалин': 'россия',
                 'спб': 'россия', 'ставрополь': 'россия', 'таджик': 'ср азия', 'тверь': 'россия', 'томск': 'россия', 'тува': 'россия', 'тула': 'россия',
                 'туркмен': 'ср азия', 'тюмень': 'россия', 'удмуртия': 'россия', 'узбек': 'ср азия', 'хабаровск': 'россия', 'хакасия': 'россия', 
                 'хмао': 'россия', 'челяб': 'россия', 'чечня': 'россия', 'чувашия': 'россия', 'чукотка': 'россия', 'ю казахстан': 'казахстан',
                 'юв казахстан': 'казахстан', 'юз казахстан': 'казахстан', 'якутия': 'россия', 'янао': 'россия', 'ярославль': 'россия', 'в казахстан': 'казахстан'}

regions_to_country = {'алтай': 'россия', 'алтайский к': 'россия', 'казахстан': 'казахстан', 'кемерово': 'россия',
                      'китай': 'китай', 'новосиб': 'россия', 'россия': 'россия', 'ср азия': 'ср азия',}

city_small_list_to_country = {' ': nan, 'барнаул': 'россия', 'барнаул г': 'россия', 'белокуриха': 'россия', 'бийск': 'россия',
                              'благовещенка рп': 'россия', 'г. барнаул': 'россия', 'г. донецк': 'россия', 'г. новоалтайск': 'россия',
                              'г.барнаул': 'россия', 'доль': 'россия', 'новоалтайск': 'россия', 'порт-хархорт': nan, 'рубцовск': 'россия',
                              'с. повалиха': 'россия', 'с. подсосново': 'россия', 'славгород': 'россия', 'среднекрасилово с': 'россия',
                              'туркменабат': 'ср азия', 'ховд': nan}
country_to_regions = {'другая': nan, 'казахстан': 'казахстан', 'китай': 'китай', 'россия': nan, 'ср азия': 'ср азия'}

city_to_countryside = {'Барнаул': 0, 'Барнаул г': 0, 'Благовещеский р-н': 1, 'Борисово': 0, 'Заринск': 0, 'Камень-на-Оби': 0, 'Кулунда': 0,
                       'Луговское с': 1, 'Новоалтайск': 0, 'Поспелихинский': 1, 'Рубцовск г': 0, 'Рубцовский': 0, 'Тальменский район': 1, 'Чепош с': 1,
                       'Шипуново с': 1, 'г.Москва': 0, 'р-н Волчихинский': 1, 'р-н Локтевский': 1, 'р.п.Малиновое озеро': 1, 'р.п.Тальменка': 1,
                       'с. Черемное': 1, 'с.Балыктуюль': 1, 'с.Волчиха': 1, 'с.Гилево': 1, 'с.Енисейское': 1, 'с.Красногорское': 1,}

city_replace_vals = [('барнаул', None), ('баранул', 'барнаул'), ('банаул', 'барнаул'), ('бареаул', 'барнаул'), ('барнааул', 'барнаул'), ('барнаул', 'барнаул'),
                     ('бийск', None), ('бийка', 'бийск'), ('новоалтайск', None), ('новалтайск', 'новоалтайск'), ('алейск', None), ('горно алтайск', None),
                     ('рубцовск', None), ('славгород', None), ('камень на оби', None), ('заринск', None), ('павловск', None), ('семей', None), ('усть каменогорск', None),
                     ('яровое', None), ('душанбе', None), ('кемерово', None), ('павлодар', None), ('шипуново', None), ('волчиха', None), ('сибирский', None),
                     ('алтайское', None), ('белокуриха', None), ('егорьевск', None), ('троицкое', None), ('поспелиха', None), ('первомайское', None), ('мамонтово', None),
                     ('горняк', None), ('шемонаиха', None), ('зудилово', None), ('тальменка', None), ('риддер', None), ('чэньчжоу', 'китай'), ('красноярск', None),
                     ('новосибирск', None), ('омск', None), ('караганда', None), ('абакан', None), ('экибастуз', None), ('новый уренгой', None), ('чита', None), ('екатеринбург', None),
                     ('сургут', None), ('семипалатинск', None), ('ангарск', None), ('москва', None), ]
city_replace_vals += [('с', 'село'), ('c', 'село'), ('село', 'село'), ('поселок', 'село'), ('района', 'село'), ('р н', 'село'), ('селение', 'село'), ('деревня', 'село'),
                      ('д', 'село'), ('п', 'село'), ('пос', 'село'), ('рп', 'село'), ('ст', 'село'), ('дачный', 'село'), ('пгт', 'село'), ('г', 'город'),]

school_common_replace_vals = [('алтайский государственный университет', 'агу'), ('алтайский государственный универстет', 'агу'),
                    ('барнаульский государственный педагогический университет', 'агпу'), ('барнаульский государственный педагогический универститет', 'агпу'), ('алтайский государственный педагогический университет', 'агпу'), ('алтайская государственная педагогическая академия', 'агпу'), ('барнаульский ордена трудового красного знамени государственный педагогический институт', 'агпу'), ('барнаульский ордена трудового красного знамени государственной педагогическийинститут', 'агпу'), ('барнаульский государственный педагогический институт', 'агпу'), ('алтайская гоударственная педагогическая академия', 'агпу'), ('государственный педагогический институт', 'агпу'), 
                    ('алтайская академия экономики и права', 'ааэп'),
                    ('алтайский государственный аграрный университет', 'агау'),
                    ('барнаульский государственный педагогический колледж', 'бгпк'), ('кгб поу барнульский государственный педагогический колледж', 'бгпк'), ('гоу спо барнаульский государственный профессионально педагогический колледж', 'бгпк'), ('гоу спо барнаульский государсвенный профессионально педагогический колледж', 'бгпк'), ('кгоу спо барнаульский государтвенный педагогический колледж', 'бгпк'), ('педагогический коллежд', 'бгпк'), ('барнаульское педагогическое училище', 'бгпк'), ('алтайский государственный педагогичсекий колледж', 'бгпк'), ('кгбоу спо барнаульский государственный педагогичский колледж', 'бгпк'), 
                    ('российская академия народного хозяйства и государственной службы при президенте', 'ранхигс'),
                    ('алтайский промышленно экономический колледж', 'апэк'),
                    ('барнаульский кадетский корпус', 'бкк'),
                    ('алтайский государственный технический университет', 'алгту'), ('алтгу', 'алгту'), ('алтайский политехнический институт им и и ползунова', 'алгту'), ('фгбоу впо алтайский государственный техническип университет им и и ползунова', 'алгту'), ('фгбоу во алтайский государственный политехнический университет', 'алгту'), ('алтайский государственный технический институт им и и ползунова', 'алгту'),  ('алтаский государственный технический университет им и и ползунова', 'алгту'), ('гоу впо алтайский гоударственный технический университет им и и ползунова', 'алгту'), ('гоу впо алтайский государсвтенный технический университет им и и ползунова', 'алгту'), 
                    ('сигма', 'лсигма'),
                    ('лицей эрудит', 'лэрудит'),
                    ('алтайский краевой педагогический лицей', 'апл'),
                    ('алтайский архитектурно строительный колледж', 'ааск'),
                    ('барнаульский строительный колледж', 'ааск'),
                    ('алтайский государственный институт культуры', 'агик'),
                    ('алтайская академия гостеприимства', 'ааг'),
                    ('республиканский классический лицей', 'ркл'),
                    ('алтайский государственный медицинский университет', 'агму'),
                    ('горно алтайский государственный политехнический колледж', 'гатт'), ('респ алтай горно алтайский государственный политехнический колледж', 'гатт'), 
                    ('алтайский политехнический техникум', 'атт'), ('алтайский политехнический техникум', 'атт'), 
                    ('рубцовский индустриальный институт алтайского государственного технического университета им ползунова', 'рии'),
                     ('кгбо школа интернат бийский лицей интернат алтайского края', 'би'), ('кгбоу бийский лицей интернат', 'би'), ('кгбо школа интернат алтайского края', 'би'), ('кгбоу бийский лицей интернат алтайского края', 'би'), ('кгоу бийский лицей интернат', 'би'),
                    ('кгб поу бийский педагогический колледж', 'бпк'), ('кгбоу бийский педагогический колледж', 'бпк'), ('кгбпоу бийский педагогический колледж', 'бпк'), 
                    ('кгбпоу бийский государственный колледж', 'бгк'), ('кгб поу бийский государственный колледж', 'бгк'), ('кгбоу спо бийский государственный колледж', 'бгк'), ('фгоу впо бийский государственный колледж', 'бгк'), ('кгбоу бийский государственный колледж', 'бгк'), ('фгоу спо бийский государственный колледж', 'бгк'), 
                    ('шукшина', 'бийскга'), ('бийский государственный педагогический институт', 'бийскга'),
                     ]
school_ordinal_replace_vals = [('кгбоу спо алтайский государственный музыкальный колледж', 'колледж'), ('кгбпоу алтайский государственный колледж', 'колледж'), ('фгоу спо барнаульсий строительный колледж', 'колледж'), ('частная школа колледж', 'колледж'), ('кгбоу спо барнаульский базовый медицинский колледж', 'колледж'), ('кгбпоу международный колледж сыроделия и профессиональных технологий', 'колледж'), ('алтайский государственный промышленно экономический колледж', 'колледж'), ('фгоу спо алтайский государственный колледж', 'колледж'), ('барнаульский машиностроительный колледж', 'колледж'), ('колледж кайнар', 'колледж'), ('ано спо барнаульский гуманитарный колледж', 'колледж'), ('кгбоу спо барнаульский торгово экономический колледж', 'колледж'), ('кгб поу алтайский краевой колледж культуры и искусств', 'колледж'), ('кгб поу алтайский государственный колледж', 'колледж'), ('кгбпоу алтайский краевой колледж культуры и искусств', 'колледж'), ('кгбпоу алтайский государственный музыкальный колледж', 'колледж'), ('кгбоу спо алтайский краевой колледж культуры', 'колледж'), ('кгбоу спо алтайский государственный колледж', 'колледж'), ('гоу спо барнаульский торгово экономический колледж', 'колледж'), ('алтайский краевой колледж культуры', 'колледж'), ('кгкп риддерский аграрно технический колледж', 'колледж'), ('тоо колледж менеджмента и бизнеса города астаны', 'колледж'), ('фгооу спо павловский сельскохозяйственный колледж', 'колледж'), ('алтайский экономический колледж', 'колледж'), ('кгу шахтинский горно индустриальный колледж', 'колледж'), ('барнаульский торгово экономический колледж', 'колледж'), ('кгкп павлодарский химико механический колледж', 'колледж'), ('кгбпоу международный колледж сыроделия', 'колледж'), ('кгбоу спо бийский государственный музыкальный колледж', 'колледж'), ('алтайский государственный музыкальный колледж', 'колледж'), ('кгб поу барнаульский базовый медицинский колледж', 'колледж'), ('фгоу спо барнаульский торгово экономический колледж', 'колледж'), ('кгбпоу барнаульский базовый медицинский колледж', 'колледж'), ('комсомольский филиал гоу хабаровский государственный медицинский колледж', 'колледж'), ('фгбоу спо алтайский государственный промышленно экономический колледж', 'колледж'), ('кгб поу родинский медицинский колледж', 'колледж'), ('благовещенский финансово экономический колледж', 'колледж'), ('фгоу спо бийский государсвтенный колледж', 'колледж'), ('кгбпоу каменский медицинский колледж', 'колледж'), ('кгб поу международный колледж сыроделия и профессиональных технологий', 'колледж'), ('кгбоу спо павловский сельскохозяйственный колледж', 'колледж'), ('колледж им кумаша нургалиева', 'колледж'), ('кгбоу спо алтайский государственный промышленно экономический колледж', 'колледж'), ('гоу спо алтайский государственный колледж культуры', 'колледж'), ('кгбоу алтайский краевой колледж культуры', 'колледж'), ('гоу спо барнаульский базовый медицинский колледж', 'колледж'), ('фгоу спо бийский государственный колледж', 'колледж'), ('алтайскирй промышленно экономический колледж г барнаул', 'колледж'), ('барпнаульский машиностроительный колледж', 'колледж'), ('университетский колледж лондона лондонский международный университет', 'колледж'), ('кгбпоу славгородский педагогический колледж', 'колледж'), ('бпоу медицинский колледж', 'колледж'), ('колледж иновационного евразийского университета', 'колледж'), ('кгбпоу волчихинский политехнический колледж', 'колледж'), ('технологический колледж', 'колледж'), ('пахтаабадский медицинский колледж', 'колледж'), ('кгкп геологоразведочный колледж уо вко акимата', 'колледж'), ('колледж семей', 'колледж'), ('гуманитарно технический колледж', 'колледж'), ('кгп на пхв павлодарский медицинский высший колледж', 'колледж'), ('семиплатинский юридический колледж мвд республики казахстан', 'колледж'), ('фгоу спо барнаульский промышленно экономический колледж', 'колледж'), ('гбпоу новосибирской области новосибирский технологический колледж питания', 'колледж'), ('кгоу спо каменский педагогический колледж', 'колледж'), ('высший колледж инновационного евразийского университета', 'колледж'), ('алтайский государственный колледж', 'колледж'), ('огоу спо новосибирский музыкальный колледж имени а ф мурова', 'колледж'), ('колледж информатики', 'колледж'), ('барнаульский колледж сервиса и туризма', 'колледж'), ('кгпоу алтайский государственный колледж', 'колледж'), ('каданский сельскохозяйственный профессиональный колледж', 'колледж'), ('кгоу рубцовский педагогический колледж', 'колледж'), ('шафриканский профессионально технический колледж', 'колледж'),
                    ('фгбоу впо оренбургский государственный педагогический университет', 'пед'), ('академический лицей при ташкентском государственном педагогическом университете им низами', 'пед'), ('гоу впо новосибирский государственный педагогический университет', 'пед'), ('гоу впо бийский педагогический государственный университет им в м шукшина', 'пед'), ('уманский государственный педагогический университет им павла тычины', 'пед'), ('фгбоу впо томский государственный педагогический университет', 'пед'), ('гоу впо бийский педагогический государственный университет имени в м шукшина', 'пед'), ('таджикский государственный педагогический университет им садриддина айни', 'пед'), ('кгбпоу славгородский педагогический колледж', 'пед'), ('фгбоу впо новосибирский государственный педагогический университет', 'пед'), ('кгоу спо каменский педагогический колледж', 'пед'), ('кгоу рубцовский педагогический колледж', 'пед'), 
                    ('таджикский технический университет имени академика м с осими', 'техн'), ('казахстанский государственный технический университет им д серикбаева', 'техн'), ('фгбоу во новосибирский государственный технический университет', 'техн'), ('фгбоу впо кузбасский государственный технический университет имени т ф горбачева', 'техн'), ('восточно казахстанский технический университет', 'техн'), ('кгкп риддерский аграрно технический колледж', 'техн'), ('фгбоу во новосибирский государственный технический университет им и и ползунова', 'техн'), ('карагандинский государственный технический университет', 'техн'), ('профессиональное техническое училище 13', 'техн'), ('фгбоу впо омский государственный технический университет', 'техн'), ('казахский политехнический институт им в и ленина', 'техн'), ('кгбпоу волчихинский политехнический колледж', 'техн'), ('гуманитарно технический колледж', 'техн'), ('шафриканский профессионально технический колледж', 'техн'),
                    ('школа', 'школа'), ('сош', 'школа'), ('мсош', 'школа'), ('мбоу', 'школа'), ('лицей', 'школа'), ('гимназия', 'школа'), ('сопш', 'школа'), ('осш', 'школа'), ('кадетский корпус', 'школа'), ('сш', 'школа'), ('ош', 'школа'), 
                    ('спо', 'колледж'), ('техникум', 'колледж'), ('училище', 'колледж'), ('колледж', 'колледж'), ('спн', 'колледж'), ('нпо', 'колледж'), ('нпо', 'колледж'), ('пу', 'колледж'), ('спу', 'колледж'),
                    ('университет', 'во'), ('институт', 'во'),  ('впо', 'во'),  ('фгбоу', 'во'), ('академия', 'во'),]
    
school_barn_replace_vals = [('гимназия 40', 'барнг40'), ('гимназия 5', 'барнг5'), ('гимназия 85', 'барнг85'), ('гимназия 80', 'барнг80'), ('гимназия 79', 'барнг79'), ('гимназия 1', 'барнг1'), ('гимназия 123', 'барнг123'), ('гимназия 22', 'барнг22'), ('гимназия 27', 'барнг27'), ('гимназия 74', 'барнг74'), ('гимназия 3', 'барнг3'), ('гимназия 69', 'барнг69'), ('гимназия 42', 'барнг42'), ('гимназия 131', 'барнг131'), ('гимназия 11', 'барнг11'), ('гимназия 45', 'барнг45'), ('гимназия 2', 'барнг2'), ('гимназия 330', 'барнг330'), ('гимназия 10', 'барнг10'), ('гимназия 44', 'барнг44'), ('гимназия 4', 'барнг4'), ('гимназия 8', 'барнг8'), ('гимназия 6', 'барнг6'), ('гимназия 24', 'барнг24'), ('гимназия 23', 'барнг23'), ('гимназия 25', 'барнг25'), ('гимназия 54', 'барнг54'), ('гимназия 1505', 'барнг1505'), ('гимназия 35', 'барнг35'), ('гимназия 59', 'барнг59'), ('гимназия 147', 'барнг147'), ('сош 59', 'барнс59'), ('сош 110', 'барнс110'), ('сош 53', 'барнс53'), ('сош 1', 'барнс1'), ('сош 4', 'барнс4'), ('сош 75', 'барнс75'), ('сош 45', 'барнс45'), ('сош 19', 'барнс19'), ('сош 126', 'барнс126'), ('сош 107', 'барнс107'), ('сош 96', 'барнс96'), ('сош 117', 'барнс117'), ('сош 128', 'барнс128'), ('сош 37', 'барнс37'), ('сош 76', 'барнс76'), ('сош 78', 'барнс78'), ('сош 127', 'барнс127'), ('сош 114', 'барнс114'), ('сош 91', 'барнс91'), ('сош 102', 'барнс102'), ('сош 34', 'барнс34'), ('сош 72', 'барнс72'), ('сош 125', 'барнс125'), ('сош 99', 'барнс99'), ('сош 60', 'барнс60'), ('сош 120', 'барнс120'), ('сош 113', 'барнс113'), ('сош 5', 'барнс5'), ('сош 84', 'барнс84'), ('сош 89', 'барнс89'), ('сош 111', 'барнс111'), ('сош 106', 'барнс106'), ('сош 10', 'барнс10'), ('сош 94', 'барнс94'), ('сош 2', 'барнс2'), ('сош 13', 'барнс13'), ('сош 48', 'барнс48'), ('сош 68', 'барнс68'), ('сош 103', 'барнс103'), ('сош 54', 'барнс54'), ('сош 9', 'барнс9'), ('сош 39', 'барнс39'), ('сош 3', 'барнс3'), ('сош 64', 'барнс64'), ('сош 55', 'барнс55'), ('сош 81', 'барнс81'), ('сош 132', 'барнс132'), ('сош 118', 'барнс118'), ('сош 25', 'барнс25'), ('сош 62', 'барнс62'), ('сош 51', 'барнс51'), ('сош 15', 'барнс15'), ('сош 8', 'барнс8'), ('сош 12', 'барнс12'), ('сош 88', 'барнс88'), ('сош 83', 'барнс83'), ('сош 63', 'барнс63'), ('сош 31', 'барнс31'), ('сош 70', 'барнс70'), ('сош 98', 'барнс98'), ('сош 131', 'барнс131'), ('сош 41', 'барнс41'), ('сош 6', 'барнс6'), ('сош 17', 'барнс17'), ('сош 50', 'барнс50'), ('сош 79', 'барнс79'), ('сош 24', 'барнс24'), ('сош 38', 'барнс38'), ('сош 23', 'барнс23'), ('сош 16', 'барнс16'), ('сош 52', 'барнс52'), ('сош 14', 'барнс14'), ('сош 20', 'барнс20'), ('сош 123', 'барнс123'), ('сош 43', 'барнс43'), ('сош 56', 'барнс56'), ('сош 7', 'барнс7'), ('сош 49', 'барнс49'), ('сош 40', 'барнс40'), ('сош 149', 'барнс149'), ('сош 21', 'барнс21'), ('сош 97', 'барнс97'), ('сош 93', 'барнс93'), ('сош 35', 'барнс35'), ('сош 28', 'барнс28'), ('сош 47', 'барнс47'), ('сош 206', 'барнс206'), ('сош 11', 'барнс11'), ('сош 101', 'барнс101'), ('сош 108', 'барнс108'), ('сош 100', 'барнс100'), ('сош 29', 'барнс29'), ('сош 32', 'барнс32'), ('сош 90', 'барнс90'), ('сош 82', 'барнс82'), ('сош 27', 'барнс27'), ('сош 30', 'барнс30'), ('сош 134', 'барнс134'), ('сош 26', 'барнс26'), ('сош 18', 'барнс18'), ('сош 69', 'барнс69'), ('сош 299', 'барнс299'), ('сош 133', 'барнс133'), ('сош 33', 'барнс33'), ('сош 42', 'барнс42'), ('сш 17', 'барнс17'), ('сш 16', 'барнс16'), ('сш 1', 'барнс1'), ('сш 3', 'барнс3'), ('сш 30', 'барнс30'), ('сш 10', 'барнс10'), ('сш 34', 'барнс34'), ('сш 100', 'барнс100'), ('сш 6', 'барнс6'), ('сш 7', 'барнс7'), ('сш 33', 'барнс33'), ('сш 37', 'барнс37'), ('сш 9', 'барнс9'), ('сш 11', 'барнс11'), ('сш 14', 'барнс14'), ('осш 3', 'барно3'), ('осш 30', 'барно30'), ('осш 33', 'барно33'), ('осшг 1', 'барно1'), ('лицей 101', 'барнл101'), ('лицей 124', 'барнл124'), ('лицей 129', 'барнл129'), ('лицей 86', 'барнл86'), ('лицей 112', 'барнл112'), ('лицей 130', 'барнл130'), ('лицей 60', 'барнл60'), ('лицей 6', 'барнл6'), ('лицей 45', 'барнл45'), ('лицей 122', 'барнл122'), ('лицей 73', 'барнл73'), ('лицей 38', 'барнл38'), ('лицей 3', 'барнл3'), ('лицей 121', 'барнл121'), ('лицей 2', 'барнл2'), ('лицей 7', 'барнл7'), ('лицей 19', 'барнл19'), ('лицей 17', 'барнл17'), ('лицей 110', 'барнл110'), ('лицей 1', 'барнл1'), ('лицей 4', 'барнл4'), ('лицей 126', 'барнл126'), ('лицей 16', 'барнл16'), ('лицей 8', 'барнл8'), ('лицей 12', 'барнл12'), ('лицей 11', 'барнл11'), ('лицей 80', 'барнл80'), ('лицей 24', 'барнл24'), ('лицей 35', 'барнл35'), ('лицей 66', 'барнл66'),]

school_biysk_replace_vals = [('гимназия 11', 'бийскг11'), ('гимназия 1', 'бийскг1'), ('гимназия 2', 'бийскг2'), ('сош 40', 'бийскс40'), ('сош 8', 'бийскс8'), ('сош 18', 'бийскс18'), ('сош 34', 'бийскс34'), ('сош 3', 'бийскс3'), ('сош 41', 'бийскс41'), ('сош 1', 'бийскс1'), ('сош 9', 'бийскс9'), ('сош 12', 'бийскс12'), ('сош 20', 'бийскс20'), ('сош 6', 'бийскс6'), ('сош 4', 'бийскс4'), ('сош 17', 'бийскс17'), ('сош 7', 'бийскс7'), ('сош 5', 'бийскс5'), ('сош 25', 'бийскс25'), ('сош 117', 'бийскс117'), ('сош 31', 'бийскс31'), ('сош 39', 'бийскс39'), ('сош 81', 'бийскс81'), ('сош 15', 'бийскс15'), ('сош 33', 'бийскс33'), ('сош 75', 'бийскс75'), ('сш 18', 'бийскс18'), ('осш 18', 'бийско18'), ('лицей 22', 'бийскл22'),
                ('кгбо школа интернат бийский лицей интернат алтайского края', 'би'), ('кгбоу бийский лицей интернат', 'би'), ('кгбо школа интернат алтайского края', 'би'), ('кгбоу бийский лицей интернат алтайского края', 'би'), ('кгоу бийский лицей интернат', 'би'),
                ('кгб поу бийский педагогический колледж', 'бпк'), ('кгбоу бийский педагогический колледж', 'бпк'), ('кгбпоу бийский педагогический колледж', 'бпк'), 
                ('кгбпоу бийский государственный колледж', 'бгк'), ('кгб поу бийский государственный колледж', 'бгк'), ('кгбоу спо бийский государственный колледж', 'бгк'), ('фгоу впо бийский государственный колледж', 'бгк'), ('кгбоу бийский государственный колледж', 'бгк'), ('фгоу спо бийский государственный колледж', 'бгк'), 
                ('шукшина', 'бийскга'), ('бийский государственный педагогический институт', 'бийскга'),
               ]

school_novoaltaysk_replace_vals = [('гимназия 166', 'новоалтайскг166'), ('сош 1', 'новоалтайскс1'), ('сош 19', 'новоалтайскс19'), ('сош 9', 'новоалтайскс9'), ('сош 15', 'новоалтайскс15'), ('сош 30', 'новоалтайскс30'), ('сош 17', 'новоалтайскс17'), ('сош 3', 'новоалтайскс3'), ('сош 10', 'новоалтайскс10'), ('сош 12', 'новоалтайскс12'), ('сош 175', 'новоалтайскс175'), ('лицей 8', 'новоалтайскл8'), ('лицей 46', 'новоалтайскл46'),
                ('общеобразовательная школа 10', 'новоалтайскс10'), ('общеобразовательная школа 12', 'новоалтайскс12'), 
                ('новоалтайское государственное художественное', 'наут'), ('государственное училище техникум', 'наут')]

school_filter = {'ordinal_school': ['мбоу', 'соу', 'моу', 'сш','школа', 'сош', 'мбвоу',
                                    'среднего образ', 'оош', 'общеобразов'],
                 'nonordinal_school': ['интернат', 'кадет'],
                 'cool_school': ['лице', 'гимназ'],
                 'npo': ['нпо', 'училище', 'пту', 'комбинат', 'пу '],
                 'spo': ['техникум', 'колледж', 'коллежд', 'спо'],
                 'vpo': ['фгбоу', 'впо', 'фбгоу', 'институт', 'академия', 'фгвооу', 'фгкоу', 'универ',
                         'консерватория', 'уриверситет', 'унивеситет', 'аквдемия', 'алтгу', 'гту']
                }