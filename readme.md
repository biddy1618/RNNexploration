# RNN and LSTM character generators based of Kazakh novel(s)

## RNN in Pytorch introduction and insight

We will try to force the RNN to produce next point following the sin wave, given input. The sequence passed along with the input will 20 points before that point. We will experiment with hidden states and number of layers while training RNN. Let's fucking roll into this procedure.

Just like it has been said, sequence length will be 20. There is going to be one input and one output. So, we will be training RNN with the following manner:

* Generate random point
* Generate 20 previous points prior that point
* Train RNN
    * Calculate loss
    * Optimize
* Observe the outputs

[Link to notebook for more details.](https://nbviewer.jupyter.org/github/biddy1618/RNNexploration/blob/master/RNN_exploration.ipynb)

## Character-level generator of text based on Kazakh novels

We will be building a text generator based of Mukhtar Auezov novels. Filrst, __make sure to upload the text file you want to generate text from__.

We will be using LSTM because they work well with long sequences of data. By long sequence, it was meant to say that if we pass for example sequence length as 50, then LSTM is less likely to have exploding/vanishing gradients, whereas RNN most defintely will.* First, scrape all the news urls.

## Reading and preprocessing of the data

Data is simply concatenation of 3 books of kazakh writer Mukhtar Auezov - "The Path of Abay I", "The Path of Abay II", and "Karash-karash" into one text file.

## Modeling

Character-level text generators are produced through 3 models of different RNN neural net
* LSTM
* GRU
* Plain RNN

Code along with detailed explanationa and annotations can be found in the [link to the notebook](https://nbviewer.jupyter.org/github/biddy1618/RNNexploration/blob/master/LSTM_char_generator.ipynb).

## Training and __output__

Models were trained for 300 epochs and sample outputs of models are as follows:

### _LSTM output_:
>Абай қарық жақсы елдің
тесебіне болды да қалып, жаңа боп алды. Бес сөзді астаған жерінің күңгісі келді. Барлық қоса жастықтар жағындағы киімдері бірі түріне біреу
көрінсін ғана тілегін қарап кетті.
>
>Бардың қайсары жайын, жаны да жірек
ала құндай. Қазыр біті болған күліп жарап жүріп қапты. Бірақ
алғашқы жоғынды жайлай, болмайды. Абай басқа кезге келді. Бозымақтан, жүрегі күйде бұрын жатып қалды.
>
>– Жаңағы қайраны жоқ. Бірақ қолан жайлап, жалаңақ, айдасыз қалам. Құл аттың ауырып келетін ел көңілімен
жарайтқан ауылдардың айтқанына жақан қолына берген жұртты алыстан, жылаған бірдей қалың жалған жүзі де боран жете жатты.
>
>Абай сыртқа байқады.
>
>Құнанбай болды да:
>
>– Жор сонау бермейміз, таңырып, аламыздығы мастаным етті дейді. Сен жар айтып тастамаймын! - деп айтты.
>
>– Ене барады» десе, жайлау қылды.
>
>– Абай, айтыңыз білмейді?.. - деп, жалғыз жеткізбей, бұрын қарады. Барлық құра боп
қалған жақын қалған. Арада кірісті. Ол сөздің
ар болмаса,
тоқтап қалды.
>
>Бірақ айтқаным да жанған жеңдей қара құйды.
>
>Барысына, алдақынан жүрген қара қара құнанды құнан қайтараның сары кеткілгенін қалың болды.
>
>Абайдың жалғыз кейін
салып жүрген күйін болатын деген жоқ-жы. Бұл қарыздың қорқылысы еді. Балаларын жақсы жасарып, бұрын қыралып, жақынына түсіп, сөйлесе береді. Қайта қыстаулық
жақындықты.
>
>Бірақ сенік артына басқа аузына барып жырқыларып, каторлап қайтпақ болғанда, жең
алып, бетін келеді. Бірақ таң
кішкенше жақсы бала босамайды.
>
>Осы жалған айыққан жанға қарай ауылдың қарасы дейтін. Оның айнағасы мен сылапан табасы бар. Айдап жетіп, соңғы боп, салып,
аласызып жүрді. Құнанбай қатты келгенде, жаналаса, қасы аталады.
>
>Қамқаның жақын айдай жүріп, жүрген жоқ. Бірақ барлық көшінің қызы де жүреді. Бірақ
сол кімістігіне жақын кiрісті да, бұйрығын бестіні тілеп алды. Баламаның аламалығы да күн басында жатқан сынысы жоқ. Семіз кешекес толық жер боп, қатып
келгенде, көп жар бар, балаларын теп ап, барып, жүзінен жең аттап алды. Оны біліп,
тынышқан бір
алысқа көп кәрі, аталыс тегіс кітіс бір жала атта

![lstm](https://raw.githubusercontent.com/biddy1618/RNNexploration/master/static/Training_hist_lstm.png)


### _GRU output_:
>Абайды. Біреу болған бала соны білгірген
екен. Сөзіне сайлауы жас айдап, жолға көп сүйді. Құнанбай мен Жабайды
астан түре алып келген еді. Қарсы, бір-ақ келін беретін. Ол бұл екен
деген бойы жас көрінеді. Бірақ олардың көзі жатыр, сенім де, осы
ауған күндің ауылдың қалың сүйініп құшақтап, тұрып айтқан.
>
>Бірақ қандап осы ауылдар бар ат білмейтін.
>
>Сынымен бір сәтте Құнанбай жол бойында топырып тастап, түгел қарап,
тағы қола болып еді. Барлы сол Абайды көріп, сыр бар еткен соң,
Абай арқаға кезектеп жатпай, сала тастады. Байтас батыр жасырын
қарсы алғандай. Сонысы екен. Жұмағұл бұл кезде айтпақан арасында бар еді.
Қыстақ берген қырылын тезінен бірі күндік, салалан сала білдірген жаңа
қосыла сол күрік бай қатты арттарынан қарайға қайтады. Сөзді айтқандай
болатын. Абайды құлай бүріп, алысқан бар дән бір сауат бұрып қалған
кім ағып, қып тұр.
>
>– Барым қара қыстарды алдың де керетті!
>
>– Сол жалған жатқан қанымыздан айналысқа басып:
>
>– Қоныс көп ағайылындарың, аран! Бірең сөрсі де артырасың! - деді. Осы
арада қыстаудың қарынан, жең жайын жала бастап, жүрек күйін айтып
қалған.
>
>Барсын да жақын жайы осындай болған болатын. Онан сары бір сөзі
қонақтарға ақырып жатыр еді. Құнанбай жақан қарып тұрған жағында жатқан
жері болатын.
>
>Абайдың күн жаңылып, тұрып, тағы да таңдайын қатты болатын.
>
>Бұл екі күндер балалақ емес тіледі.. Қытқа алған-сол, жақындар дос
жүнді.
>
>Алшынбайдың асыл көзі жосыны асып, кейін қосып, бірез қылғаш тұр еткен
қолын ерекше кеңдесіп тұрып тұрып, қатты байып қып тұрды. Базаралы бір
қалып еді.
>
>Абайдың және біз байлауы ала бергенде, алыстай, күйін айтқан.
>
>Соны айтқызған жоқ. Қара көшілері де қатты созаларды енді қалаға
көрінген.
>
>Қаратой күйеудің барласа да тұр екен. Оның жас қыстауы жайының жауыны
жақыл болатын. Осы ауылдаң басқа қайта бұл топтың көңілі жандара жанып
тұр.
>
>Құнанбай боп бала жолға жалғыз жүзі тартып, қолын қыстиды да: - Қартық
жүрген басын басы бар. Жалынып құсыққан көп екен.
>
>Абай содан бір аттылар тартып қалған екен. Байқасындай сияқты сөніле
қ

![gru](https://raw.githubusercontent.com/biddy1618/RNNexploration/master/static/Training_hist_gru.png)

### _RNN output_:
>Абайың,
көріміс босында қайса, тартып ап, артың жалғыз жүргеніндi берiп отырғың, қалай
бұлалған жалайды.
>
>Бірақ жақын жүрген сыңынан бiрдіңдiң
бiстер бай боп қалғын жоқ, түн бетін тарып, қайдан болса ды. Осы көшi екен. Боқарым жеке сеттi.
>
>Ой жоқ тағаны ескі жаққа каткен жақын таңғанған болған, бiр қылға кісiнi кiргені алған, сағы баласы, жал қатқар болат атты да алап қалды. Барсын айлыптың бал салдар күрiсiн еретіп жатқанды, арындар болатын. Сон сойтын қолға
жыбарын бұл екi жоқ.
>
>Боларары
қарында бар болатын!
 Абай бар жоқ бейге толағы көзге қарай күлгесің, келіп, жүргіз жақаны болырта жарағып, жарандың аса бұрын дәрсіз дылдарына қарсы жатарын тысын болмасан сенiң ма?
>
>Құлақ балып кіпінен жатып, жайылап берiп, қыйлайдан қалаң топ күн бiр атын жердiң, жүні айтатын
болып келді.
 Балағызы, болары тасырап қайда қарылды. Ой ештер бағыны жоқ. Сүнді бетендерін, саған араз салған жоқ. Бірақ сала мен
келді. Осы күзі байырып, жарастырды. Осы баламыл бір қыста жақыр бойын айтырған жылы атаулыра, жалғаз көп көзінен қалғоның бала тұрып, арты айтты.
>
>–
Еедеріне сойғандай. Сон мен жарған жерiнде қырқана айтап алып кеткіндей. Сүгелiнiң өзін құлақ құлық айтып, ауылдар кеттi. Болытып күстi. Солар бiр күне байланып:
>
>– Абай теген қасана тайтып тапты. Бірас қарағасында боп көрiп
керек, сол бері жесегi қарақсы берiп кетiпті.
Абайдың арылдай жақын, сөзге жерекен болыт жарқ болда да алғұн.
>
>Қозырысын балып келеді. Азалды ауны салып: 
– Енді осы бір ауылдар астынан, сарғалы екке жақан ауландай,,
бір басы де, түр жақана қылған жолында осы көңiлiнде
Арасында қолға қарай берде түсiп қаласанып еліп та жесерінін ені жетті. Осанеді де
келдіңдер болған. Олардан, арандық болады? – деп,. Қала бұл қайарымыңдар бар. Абыйға қарый жүйес болғанда, алдында көп жүгөнесінің берi жеткен боласың етi болада.
>
>Бұлыр да бар екен! - деген. Бұр сөзінде көз елiп, қайты жатып, олыс айтады.
>
>Құдарбек қастыр тартыстап жатқандық солып барған болда, осынан бiр алып, жалын қатты айықтарынан берiп жатқанының жал қосып, сөп 

![rnn](https://raw.githubusercontent.com/biddy1618/RNNexploration/master/static/Training_hist_rnn.png)

## Observations

Trained for 300 epochs _GRU_ model showed great results. It learned the essay structure well, both grammar and semantics also not bad. On the first glance, one wouldn't notice if the text was generated with recurrent network. _LSTM_ on the other hand seems to be lost. Structure and grammatics, along with punctuations are not close to perfect. Seems maybe I should try to train the network with different parameters and for longer/shorter epochs. _RNN_ seems to be performing better for the naked eye, but still not close to _GRU_. Very interesting that _LSTM_ did poor on the text generation.