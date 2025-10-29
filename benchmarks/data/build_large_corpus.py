"""Generate larger multilingual corpus (500 samples per language)."""

import json
import random
from pathlib import Path

# Expanded templates - MORE variety for non-Latin scripts
TEXTS = {
    'English': [
        # News (10)
        "The government announced new policies yesterday affecting millions of citizens across the nation.",
        "Scientists discovered breakthrough technology in renewable energy research at the university today.",
        "Economic indicators showed strong recovery with unemployment dropping to the lowest level in years.",
        "Climate change protesters gathered in the capital demanding immediate action from lawmakers.",
        "Technology companies reported record profits despite global economic uncertainty and challenges.",
        "Healthcare workers continue to face shortages of essential medical supplies and equipment.",
        "Education reform bill passed parliament after months of intense debate and negotiations.",
        "Transportation infrastructure received major funding boost for next five years of development.",
        "International trade agreements reached after lengthy negotiations between multiple countries.",
        "Digital privacy concerns raised as new surveillance technology deployed nationwide.",
        # Conversations (10)
        "Hello! How are you doing today? I hope everything is going well for you and your family.",
        "Thanks so much for your help yesterday with the project. I really appreciate all your effort.",
        "What time works best for you tomorrow? I'm available after three in the afternoon.",
        "Could you please send me that document when you get a chance? No rush at all.",
        "Let's meet for coffee next week and catch up. It's been too long since we talked.",
        "I completely agree with your suggestion. That sounds like a great plan to me.",
        "Sorry for the delay in responding. I've been incredibly busy with work lately.",
        "Have you seen the latest update? It looks really interesting and worth checking out.",
        "Please let me know if you need any additional information or help with anything.",
        "That's wonderful news! I'm so happy for you and excited to hear more details.",
        # Instructions (10)
        "First, preheat the oven to 180 degrees celsius. Then carefully mix the flour and sugar together in a large bowl.",
        "To reset your password, click the forgot password link below and enter your email address.",
        "Press and hold the power button for exactly 3 seconds then wait for the blue indicator light.",
        "Open the application settings menu and navigate to the security preferences section carefully.",
        "Connect the cable to port number two on the back panel before turning on the device.",
        "Save your work frequently by pressing control S or clicking the save button in the toolbar.",
        "Read the entire instruction manual thoroughly before attempting to assemble the furniture.",
        "Download the latest software update from the official website before installing any plugins.",
        "Measure twice and cut once to ensure accurate dimensions for the construction project.",
        "Follow the recipe instructions precisely for best results and perfect consistency every time.",
    ],
    'Spanish': [
        "El gobierno anunció nuevas políticas ayer que afectan a millones de ciudadanos en todo el país.",
        "Los científicos descubrieron tecnología revolucionaria en investigación de energía renovable en la universidad hoy.",
        "Los indicadores económicos mostraron una fuerte recuperación con el desempleo cayendo al nivel más bajo en años.",
        "Los manifestantes por el cambio climático se reunieron en la capital exigiendo acción inmediata de los legisladores.",
        "Las empresas tecnológicas reportaron ganancias récord a pesar de la incertidumbre económica global y los desafíos.",
        "Los trabajadores de la salud continúan enfrentando escasez de suministros médicos esenciales y equipamiento.",
        "El proyecto de ley de reforma educativa aprobó el parlamento después de meses de intenso debate y negociaciones.",
        "La infraestructura de transporte recibió un importante impulso de financiación para los próximos cinco años de desarrollo.",
        "Se alcanzaron acuerdos comerciales internacionales después de largas negociaciones entre múltiples países.",
        "Se plantearon preocupaciones sobre la privacidad digital al desplegarse nueva tecnología de vigilancia en todo el país.",
        "¡Hola! ¿Cómo estás hoy? Espero que todo vaya muy bien para ti y tu familia.",
        "Muchas gracias por tu ayuda ayer con el proyecto. Realmente aprecio todo tu esfuerzo.",
        "¿Qué hora te viene mejor mañana? Estoy disponible después de las tres de la tarde.",
        "¿Podrías enviarme ese documento cuando tengas oportunidad? No hay prisa en absoluto.",
        "Reunámonos para tomar café la próxima semana y ponernos al día. Ha pasado mucho tiempo desde que hablamos.",
        "Estoy completamente de acuerdo con tu sugerencia. Me parece un plan excelente.",
        "Disculpa el retraso en responder. He estado increíblemente ocupado con el trabajo últimamente.",
        "¿Has visto la última actualización? Se ve realmente interesante y vale la pena revisarla.",
        "Por favor avísame si necesitas información adicional o ayuda con cualquier cosa.",
        "¡Esas son noticias maravillosas! Estoy muy feliz por ti y emocionado de escuchar más detalles.",
        "Primero, precalienta el horno a 180 grados centígrados. Luego mezcla cuidadosamente la harina y el azúcar en un tazón grande.",
        "Para restablecer tu contraseña, haz clic en el enlace de contraseña olvidada abajo e ingresa tu dirección de correo electrónico.",
        "Presiona y mantén el botón de encendido durante exactamente 3 segundos y espera la luz indicadora azul.",
        "Abre el menú de configuración de la aplicación y navega cuidadosamente a la sección de preferencias de seguridad.",
        "Conecta el cable al puerto número dos en el panel trasero antes de encender el dispositivo.",
        "Guarda tu trabajo frecuentemente presionando control S o haciendo clic en el botón guardar en la barra de herramientas.",
        "Lee todo el manual de instrucciones completamente antes de intentar ensamblar los muebles.",
        "Descarga la última actualización de software del sitio web oficial antes de instalar cualquier complemento.",
        "Mide dos veces y corta una vez para asegurar dimensiones precisas para el proyecto de construcción.",
        "Sigue las instrucciones de la receta con precisión para obtener los mejores resultados y consistencia perfecta siempre.",
    ],
    'French': [
        "Le gouvernement a annoncé de nouvelles politiques hier affectant des millions de citoyens à travers la nation.",
        "Les scientifiques ont découvert une technologie révolutionnaire dans la recherche sur l'énergie renouvelable à l'université aujourd'hui.",
        "Les indicateurs économiques ont montré une forte reprise avec le chômage tombant au niveau le plus bas depuis des années.",
        "Les manifestants pour le climat se sont rassemblés dans la capitale exigeant une action immédiate des législateurs.",
        "Les entreprises technologiques ont rapporté des profits records malgré l'incertitude économique mondiale et les défis.",
        "Les travailleurs de la santé continuent de faire face à des pénuries de fournitures médicales essentielles et d'équipement.",
        "Le projet de loi de réforme de l'éducation a été adopté au parlement après des mois de débat intense et de négociations.",
        "L'infrastructure des transports a reçu un financement majeur pour les cinq prochaines années de développement.",
        "Des accords commerciaux internationaux ont été conclus après de longues négociations entre plusieurs pays.",
        "Des préoccupations concernant la vie privée numérique ont été soulevées avec le déploiement de nouvelles technologies de surveillance à l'échelle nationale.",
        "Bonjour! Comment allez-vous aujourd'hui? J'espère que tout va très bien pour vous et votre famille.",
        "Merci beaucoup pour votre aide hier avec le projet. J'apprécie vraiment tous vos efforts.",
        "Quelle heure vous convient le mieux demain? Je suis disponible après trois heures de l'après-midi.",
        "Pourriez-vous m'envoyer ce document quand vous aurez l'occasion? Pas de précipitation du tout.",
        "Retrouvons-nous pour un café la semaine prochaine et rattrapons le temps perdu. Cela fait trop longtemps que nous n'avons pas parlé.",
        "Je suis tout à fait d'accord avec votre suggestion. Cela me semble être un excellent plan.",
        "Désolé pour le retard dans ma réponse. J'ai été incroyablement occupé avec le travail dernièrement.",
        "Avez-vous vu la dernière mise à jour? Cela semble vraiment intéressant et mérite d'être consulté.",
        "Veuillez me faire savoir si vous avez besoin d'informations supplémentaires ou d'aide pour quoi que ce soit.",
        "C'est une nouvelle merveilleuse! Je suis si heureux pour vous et impatient d'entendre plus de détails.",
        "D'abord, préchauffez le four à 180 degrés celsius. Ensuite, mélangez soigneusement la farine et le sucre dans un grand bol.",
        "Pour réinitialiser votre mot de passe, cliquez sur le lien mot de passe oublié ci-dessous et entrez votre adresse e-mail.",
        "Appuyez et maintenez le bouton d'alimentation pendant exactement 3 secondes puis attendez le voyant bleu.",
        "Ouvrez le menu des paramètres de l'application et naviguez soigneusement vers la section des préférences de sécurité.",
        "Connectez le câble au port numéro deux sur le panneau arrière avant d'allumer l'appareil.",
        "Enregistrez votre travail fréquemment en appuyant sur contrôle S ou en cliquant sur le bouton enregistrer dans la barre d'outils.",
        "Lisez tout le manuel d'instructions complètement avant de tenter d'assembler les meubles.",
        "Téléchargez la dernière mise à jour du logiciel depuis le site web officiel avant d'installer des plugins.",
        "Mesurez deux fois et coupez une fois pour assurer des dimensions précises pour le projet de construction.",
        "Suivez les instructions de la recette avec précision pour de meilleurs résultats et une consistance parfaite à chaque fois.",
    ],
    'German': [
        "Die Regierung kündigte gestern neue Richtlinien an, die Millionen von Bürgern im ganzen Land betreffen.",
        "Wissenschaftler entdeckten heute bahnbrechende Technologie in der Erforschung erneuerbarer Energien an der Universität.",
        "Wirtschaftsindikatoren zeigten eine starke Erholung mit einer Arbeitslosigkeit, die auf den niedrigsten Stand seit Jahren gesunken ist.",
        "Klimaschutz-Demonstranten versammelten sich in der Hauptstadt und forderten sofortige Maßnahmen von den Gesetzgebern.",
        "Technologieunternehmen meldeten Rekordgewinne trotz globaler wirtschaftlicher Unsicherheit und Herausforderungen.",
        "Gesundheitspersonal steht weiterhin vor Engpässen bei lebenswichtigen medizinischen Versorgungsgütern und Ausrüstung.",
        "Das Bildungsreformgesetz wurde nach monatelanger intensiver Debatte und Verhandlungen vom Parlament verabschiedet.",
        "Die Verkehrsinfrastruktur erhielt eine bedeutende Finanzierungsspritze für die nächsten fünf Jahre der Entwicklung.",
        "Internationale Handelsabkommen wurden nach langwierigen Verhandlungen zwischen mehreren Ländern erreicht.",
        "Bedenken hinsichtlich der digitalen Privatsphäre wurden geäußert, als neue Überwachungstechnologie landesweit eingesetzt wurde.",
        "Hallo! Wie geht es Ihnen heute? Ich hoffe, dass alles sehr gut für Sie und Ihre Familie läuft.",
        "Vielen Dank für Ihre Hilfe gestern mit dem Projekt. Ich schätze wirklich all Ihre Bemühungen.",
        "Welche Zeit passt Ihnen morgen am besten? Ich bin nach drei Uhr nachmittags verfügbar.",
        "Könnten Sie mir dieses Dokument schicken, wenn Sie Gelegenheit haben? Keine Eile überhaupt.",
        "Lassen Sie uns nächste Woche auf einen Kaffee treffen und uns austauschen. Es ist zu lange her, dass wir gesprochen haben.",
        "Ich stimme Ihrem Vorschlag vollkommen zu. Das klingt für mich nach einem ausgezeichneten Plan.",
        "Entschuldigung für die verspätete Antwort. Ich war in letzter Zeit unglaublich beschäftigt mit der Arbeit.",
        "Haben Sie das neueste Update gesehen? Es sieht wirklich interessant aus und ist es wert, überprüft zu werden.",
        "Bitte lassen Sie mich wissen, wenn Sie zusätzliche Informationen oder Hilfe bei irgendetwas benötigen.",
        "Das sind wunderbare Neuigkeiten! Ich bin so glücklich für Sie und freue mich darauf, mehr Details zu hören.",
        "Zuerst heizen Sie den Ofen auf 180 Grad Celsius vor. Dann mischen Sie vorsichtig Mehl und Zucker in einer großen Schüssel.",
        "Um Ihr Passwort zurückzusetzen, klicken Sie auf den Link Passwort vergessen unten und geben Sie Ihre E-Mail-Adresse ein.",
        "Drücken und halten Sie den Ein-Aus-Knopf für genau 3 Sekunden und warten Sie dann auf das blaue Anzeigelicht.",
        "Öffnen Sie das Einstellungsmenü der Anwendung und navigieren Sie sorgfältig zum Abschnitt der Sicherheitseinstellungen.",
        "Verbinden Sie das Kabel mit Port Nummer zwei auf der Rückseite, bevor Sie das Gerät einschalten.",
        "Speichern Sie Ihre Arbeit häufig, indem Sie Strg S drücken oder auf die Schaltfläche Speichern in der Symbolleiste klicken.",
        "Lesen Sie die gesamte Bedienungsanleitung gründlich durch, bevor Sie versuchen, die Möbel zusammenzubauen.",
        "Laden Sie das neueste Software-Update von der offiziellen Website herunter, bevor Sie Plugins installieren.",
        "Messen Sie zweimal und schneiden Sie einmal, um genaue Abmessungen für das Bauprojekt sicherzustellen.",
        "Befolgen Sie die Rezeptanweisungen genau für beste Ergebnisse und perfekte Konsistenz jedes Mal.",
    ],
    'Portuguese': [
        "O governo anunciou novas políticas ontem afetando milhões de cidadãos em todo o país.",
        "Cientistas descobriram tecnologia revolucionária em pesquisa de energia renovável na universidade hoje.",
        "Indicadores econômicos mostraram forte recuperação com desemprego caindo para o nível mais baixo em anos.",
        "Manifestantes por mudanças climáticas se reuniram na capital exigindo ação imediata dos legisladores.",
        "Empresas de tecnologia reportaram lucros recordes apesar da incerteza econômica global e desafios.",
        "Trabalhadores da saúde continuam enfrentando escassez de suprimentos médicos essenciais e equipamentos.",
        "Projeto de lei de reforma educacional passou no parlamento após meses de intenso debate e negociações.",
        "Infraestrutura de transporte recebeu grande impulso de financiamento para os próximos cinco anos de desenvolvimento.",
        "Acordos comerciais internacionais foram alcançados após longas negociações entre múltiplos países.",
        "Preocupações com privacidade digital foram levantadas com implantação de nova tecnologia de vigilância em todo o país.",
        "Olá! Como você está hoje? Espero que tudo esteja indo muito bem para você e sua família.",
        "Muito obrigado pela sua ajuda ontem com o projeto. Eu realmente agradeço todo o seu esforço.",
        "Que horas funcionam melhor para você amanhã? Estou disponível depois das três da tarde.",
        "Você poderia me enviar aquele documento quando tiver chance? Sem pressa alguma.",
        "Vamos nos encontrar para um café na próxima semana e colocar o papo em dia. Faz tempo demais que não conversamos.",
        "Concordo completamente com sua sugestão. Isso me parece um plano excelente.",
        "Desculpe a demora em responder. Tenho estado incrivelmente ocupado com o trabalho ultimamente.",
        "Você viu a última atualização? Parece realmente interessante e vale a pena conferir.",
        "Por favor me avise se você precisar de informações adicionais ou ajuda com qualquer coisa.",
        "Essas são notícias maravilhosas! Estou tão feliz por você e animado para ouvir mais detalhes.",
        "Primeiro, preaqueça o forno a 180 graus celsius. Depois misture cuidadosamente a farinha e o açúcar em uma tigela grande.",
        "Para redefinir sua senha, clique no link esqueceu a senha abaixo e digite seu endereço de e-mail.",
        "Pressione e segure o botão de energia por exatamente 3 segundos e aguarde a luz indicadora azul.",
        "Abra o menu de configurações do aplicativo e navegue cuidadosamente até a seção de preferências de segurança.",
        "Conecte o cabo à porta número dois no painel traseiro antes de ligar o dispositivo.",
        "Salve seu trabalho frequentemente pressionando control S ou clicando no botão salvar na barra de ferramentas.",
        "Leia todo o manual de instruções completamente antes de tentar montar os móveis.",
        "Baixe a atualização de software mais recente do site oficial antes de instalar quaisquer plugins.",
        "Meça duas vezes e corte uma vez para garantir dimensões precisas para o projeto de construção.",
        "Siga as instruções da receita com precisão para melhores resultados e consistência perfeita todas as vezes.",
    ],
    'Italian': [
        "Il governo ha annunciato nuove politiche ieri che colpiscono milioni di cittadini in tutto il paese.",
        "Gli scienziati hanno scoperto tecnologia rivoluzionaria nella ricerca sull'energia rinnovabile all'università oggi.",
        "Gli indicatori economici hanno mostrato una forte ripresa con la disoccupazione che scende al livello più basso in anni.",
        "I manifestanti per il cambiamento climatico si sono riuniti nella capitale chiedendo azione immediata dai legislatori.",
        "Le aziende tecnologiche hanno riportato profitti record nonostante l'incertezza economica globale e le sfide.",
        "Gli operatori sanitari continuano ad affrontare carenze di forniture mediche essenziali e attrezzature.",
        "Il disegno di legge di riforma dell'istruzione è passato in parlamento dopo mesi di intenso dibattito e negoziati.",
        "L'infrastruttura dei trasporti ha ricevuto un importante incremento di finanziamenti per i prossimi cinque anni di sviluppo.",
        "Accordi commerciali internazionali raggiunti dopo lunghe negoziazioni tra più paesi.",
        "Preoccupazioni sulla privacy digitale sollevate con il dispiegamento di nuove tecnologie di sorveglianza a livello nazionale.",
        "Ciao! Come stai oggi? Spero che tutto vada molto bene per te e la tua famiglia.",
        "Grazie mille per il tuo aiuto ieri con il progetto. Apprezzo davvero tutti i tuoi sforzi.",
        "Che ora va meglio per te domani? Sono disponibile dopo le tre del pomeriggio.",
        "Potresti inviarmi quel documento quando hai un momento? Nessuna fretta affatto.",
        "Incontriamoci per un caffè la prossima settimana e mettiamoci in pari. È passato troppo tempo da quando abbiamo parlato.",
        "Sono completamente d'accordo con il tuo suggerimento. Mi sembra un piano eccellente.",
        "Scusa per il ritardo nella risposta. Sono stato incredibilmente impegnato con il lavoro ultimamente.",
        "Hai visto l'ultimo aggiornamento? Sembra davvero interessante e vale la pena controllare.",
        "Per favore fammi sapere se hai bisogno di informazioni aggiuntive o aiuto con qualsiasi cosa.",
        "Queste sono notizie meravigliose! Sono così felice per te ed entusiasta di sentire più dettagli.",
        "Prima, riscalda il forno a 180 gradi celsius. Poi mescola attentamente la farina e lo zucchero in una ciotola grande.",
        "Per reimpostare la password, fai clic sul link password dimenticata qui sotto e inserisci il tuo indirizzo email.",
        "Premi e tieni premuto il pulsante di accensione per esattamente 3 secondi e attendi la luce indicatrice blu.",
        "Apri il menu delle impostazioni dell'applicazione e naviga attentamente alla sezione delle preferenze di sicurezza.",
        "Collega il cavo alla porta numero due sul pannello posteriore prima di accendere il dispositivo.",
        "Salva il tuo lavoro frequentemente premendo control S o cliccando sul pulsante salva nella barra degli strumenti.",
        "Leggi tutto il manuale di istruzioni completamente prima di tentare di assemblare i mobili.",
        "Scarica l'ultimo aggiornamento software dal sito web ufficiale prima di installare qualsiasi plugin.",
        "Misura due volte e taglia una volta per assicurare dimensioni precise per il progetto di costruzione.",
        "Segui le istruzioni della ricetta con precisione per i migliori risultati e consistenza perfetta ogni volta.",
    ],
    'Russian': [
        "Правительство объявило вчера о новой политике, затрагивающей миллионы граждан по всей стране.",
        "Ученые сегодня открыли революционную технологию в исследовании возобновляемой энергии в университете.",
        "Экономические показатели показали сильное восстановление с безработицей, упавшей до самого низкого уровня за годы.",
        "Демонстранты за климат собрались в столице, требуя немедленных действий от законодателей.",
        "Технологические компании сообщили о рекордной прибыли, несмотря на глобальную экономическую неопределенность и проблемы.",
        "Медицинские работники продолжают сталкиваться с нехваткой необходимых медицинских принадлежностей и оборудования.",
        "Законопроект о реформе образования прошел через парламент после месяцев интенсивных дебатов и переговоров.",
        "Транспортная инфраструктура получила значительное финансирование на следующие пять лет развития.",
        "Международные торговые соглашения достигнуты после длительных переговоров между несколькими странами.",
        "Подняты опасения по поводу цифровой конфиденциальности с развертыванием новых технологий наблюдения по всей стране.",
        "Привет! Как дела сегодня? Надеюсь, у тебя и твоей семьи все очень хорошо.",
        "Большое спасибо за твою помощь вчера с проектом. Я действительно ценю все твои усилия.",
        "Какое время тебе лучше подходит завтра? Я свободен после трех часов дня.",
        "Не мог бы ты прислать мне этот документ, когда будет возможность? Совсем не срочно.",
        "Давай встретимся за кофе на следующей неделе и поговорим. Прошло слишком много времени с нашего последнего разговора.",
        "Я полностью согласен с твоим предложением. Это звучит как отличный план для меня.",
        "Извини за задержку с ответом. Я был невероятно занят работой в последнее время.",
        "Ты видел последнее обновление? Выглядит действительно интересно и стоит проверить.",
        "Пожалуйста, дай мне знать, если тебе нужна дополнительная информация или помощь с чем-либо.",
        "Это чудесные новости! Я так рад за тебя и с нетерпением жду подробностей.",
        "Сначала разогрейте духовку до 180 градусов по Цельсию. Затем тщательно смешайте муку и сахар в большой миске.",
        "Чтобы сбросить пароль, нажмите на ссылку забыли пароль ниже и введите свой адрес электронной почты.",
        "Нажмите и удерживайте кнопку питания ровно 3 секунды, затем дождитесь синего индикатора.",
        "Откройте меню настроек приложения и осторожно перейдите в раздел настроек безопасности.",
        "Подключите кабель к порту номер два на задней панели перед включением устройства.",
        "Сохраняйте свою работу часто, нажимая control S или щелкая кнопку сохранить на панели инструментов.",
        "Прочитайте всю инструкцию полностью перед тем как пытаться собрать мебель.",
        "Загрузите последнее обновление программного обеспечения с официального сайта перед установкой плагинов.",
        "Измерьте дважды и отрежьте один раз, чтобы обеспечить точные размеры для строительного проекта.",
        "Следуйте инструкциям рецепта точно для лучших результатов и идеальной консистенции каждый раз.",
    ],
    'Arabic': [
        "أعلنت الحكومة أمس عن سياسات جديدة تؤثر على ملايين المواطنين في جميع أنحاء البلاد.",
        "اكتشف العلماء اليوم تكنولوجيا ثورية في أبحاث الطاقة المتجددة في الجامعة.",
        "أظهرت المؤشرات الاقتصادية تعافياً قوياً مع انخفاض البطالة إلى أدنى مستوى لها منذ سنوات.",
        "تجمع المتظاهرون من أجل المناخ في العاصمة مطالبين بإجراءات فورية من المشرعين.",
        "أعلنت شركات التكنولوجيا عن أرباح قياسية على الرغم من عدم اليقين الاقتصادي العالمي والتحديات.",
        "يواصل العاملون في مجال الرعاية الصحية مواجهة نقص في الإمدادات الطبية الأساسية والمعدات.",
        "أقر البرلمان مشروع قانون إصلاح التعليم بعد شهور من النقاش المكثف والمفاوضات.",
        "حصلت البنية التحتية للنقل على دفعة تمويل كبيرة للسنوات الخمس القادمة من التطوير.",
        "تم التوصل إلى اتفاقيات تجارية دولية بعد مفاوضات طويلة بين عدة دول.",
        "أثيرت مخاوف بشأن الخصوصية الرقمية مع نشر تكنولوجيا المراقبة الجديدة على الصعيد الوطني.",
        "مرحباً! كيف حالك اليوم؟ أتمنى أن يكون كل شيء على ما يرام لك ولعائلتك.",
        "شكراً جزيلاً على مساعدتك أمس في المشروع. أنا حقاً أقدر كل جهودك.",
        "ما الوقت الأنسب لك غداً؟ أنا متاح بعد الساعة الثالثة بعد الظهر.",
        "هل يمكنك إرسال تلك الوثيقة لي عندما تتاح لك الفرصة؟ لا استعجال على الإطلاق.",
        "دعنا نلتقي لتناول القهوة الأسبوع المقبل ونلحق بالأخبار. لقد مر وقت طويل منذ أن تحدثنا.",
        "أنا أتفق تماماً مع اقتراحك. يبدو لي ذلك خطة ممتازة.",
        "آسف على التأخير في الرد. لقد كنت مشغولاً بشكل لا يصدق بالعمل مؤخراً.",
        "هل رأيت آخر تحديث؟ يبدو مثيراً للاهتمام حقاً ويستحق المراجعة.",
        "من فضلك أخبرني إذا كنت بحاجة إلى معلومات إضافية أو مساعدة في أي شيء.",
        "هذه أخبار رائعة! أنا سعيد جداً من أجلك ومتحمس لسماع المزيد من التفاصيل.",
        "أولاً، سخن الفرن إلى 180 درجة مئوية. ثم اخلط الدقيق والسكر بعناية في وعاء كبير.",
        "لإعادة تعيين كلمة المرور، انقر فوق رابط نسيت كلمة المرور أدناه وأدخل عنوان بريدك الإلكتروني.",
        "اضغط مع الاستمرار على زر الطاقة لمدة 3 ثوانٍ بالضبط ثم انتظر ضوء المؤشر الأزرق.",
        "افتح قائمة إعدادات التطبيق وانتقل بعناية إلى قسم تفضيلات الأمان.",
        "قم بتوصيل الكابل بالمنفذ رقم اثنين على اللوحة الخلفية قبل تشغيل الجهاز.",
        "احفظ عملك بشكل متكرر بالضغط على control S أو النقر فوق زر الحفظ في شريط الأدوات.",
        "اقرأ دليل التعليمات بالكامل بشكل شامل قبل محاولة تجميع الأثاث.",
        "قم بتنزيل أحدث تحديث للبرنامج من الموقع الرسمي قبل تثبيت أي إضافات.",
        "قس مرتين واقطع مرة واحدة لضمان أبعاد دقيقة لمشروع البناء.",
        "اتبع تعليمات الوصفة بدقة للحصول على أفضل النتائج والاتساق المثالي في كل مرة.",
    ],
    'Chinese': [
        "政府昨天宣布了影响全国数百万公民的新政策。",
        "科学家今天在大学发现了可再生能源研究的突破性技术。",
        "经济指标显示强劲复苏，失业率降至多年来的最低水平。",
        "气候变化抗议者聚集在首都，要求立法者立即采取行动。",
        "尽管全球经济不确定性和挑战，科技公司报告了创纪录的利润。",
        "医疗工作者继续面临基本医疗用品和设备的短缺。",
        "教育改革法案在经过数月激烈辩论和谈判后通过了议会。",
        "交通基础设施获得了未来五年发展的重大资金支持。",
        "经过多国之间的长期谈判，达成了国际贸易协议。",
        "随着新监控技术在全国范围内部署，数字隐私问题引起了关注。",
        "你好！你今天怎么样？我希望你和你的家人一切都很顺利。",
        "非常感谢你昨天在项目上的帮助。我真的很感激你所有的努力。",
        "明天什么时间对你最合适？我下午三点以后有空。",
        "你有机会的时候能把那个文件发给我吗？完全不着急。",
        "我们下周见面喝杯咖啡聊聊吧。我们已经太久没说话了。",
        "我完全同意你的建议。在我看来这是个很好的计划。",
        "抱歉回复晚了。我最近工作特别忙。",
        "你看到最新的更新了吗？看起来真的很有趣，值得查看。",
        "如果你需要任何额外的信息或帮助，请告诉我。",
        "这真是太好的消息了！我为你感到非常高兴，很想听到更多细节。",
        "首先，将烤箱预热到180摄氏度。然后在一个大碗里仔细混合面粉和糖。",
        "要重置密码，请点击下面的忘记密码链接并输入您的电子邮件地址。",
        "按住电源按钮恰好3秒钟，然后等待蓝色指示灯亮起。",
        "打开应用程序设置菜单，仔细导航到安全首选项部分。",
        "在打开设备之前，将电缆连接到后面板上的二号端口。",
        "通过按control S或点击工具栏中的保存按钮来频繁保存您的工作。",
        "在尝试组装家具之前，请彻底阅读整个说明手册。",
        "在安装任何插件之前，从官方网站下载最新的软件更新。",
        "测量两次，切割一次，以确保建筑项目的精确尺寸。",
        "精确遵循食谱说明，以获得最佳结果和每次完美的一致性。",
    ],
    'Hindi': [
        "सरकार ने कल नई नीतियों की घोषणा की जो पूरे देश में लाखों नागरिकों को प्रभावित करती हैं।",
        "वैज्ञानिकों ने आज विश्वविद्यालय में नवीकरणीय ऊर्जा अनुसंधान में क्रांतिकारी तकनीक की खोज की।",
        "आर्थिक संकेतकों ने मजबूत सुधार दिखाया जिसमें बेरोजगारी वर्षों में सबसे निचले स्तर पर आ गई।",
        "जलवायु परिवर्तन प्रदर्शनकारी राजधानी में इकट्ठा हुए और विधायकों से तत्काल कार्रवाई की मांग की।",
        "प्रौद्योगिकी कंपनियों ने वैश्विक आर्थिक अनिश्चितता और चुनौतियों के बावजूद रिकॉर्ड मुनाफे की सूचना दी।",
        "स्वास्थ्य कार्यकर्ता आवश्यक चिकित्सा आपूर्ति और उपकरण की कमी का सामना करना जारी रखते हैं।",
        "शिक्षा सुधार विधेयक महीनों की गहन बहस और बातचीत के बाद संसद से पारित हो गया।",
        "परिवहन अवसंरचना को अगले पांच वर्षों के विकास के लिए प्रमुख वित्त पोषण को बढ़ावा मिला।",
        "कई देशों के बीच लंबी बातचीत के बाद अंतर्राष्ट्रीय व्यापार समझौते हासिल किए गए।",
        "देशभर में नई निगरानी तकनीक की तैनाती के साथ डिजिटल गोपनीयता की चिंताएं उठाई गईं।",
        "नमस्ते! आज आप कैसे हैं? मुझे उम्मीद है कि आपके और आपके परिवार के लिए सब कुछ बहुत अच्छा चल रहा है।",
        "परियोजना में कल आपकी मदद के लिए बहुत-बहुत धन्यवाद। मैं वास्तव में आपके सभी प्रयासों की सराहना करता हूं।",
        "कल आपके लिए कौन सा समय सबसे अच्छा है? मैं दोपहर तीन बजे के बाद उपलब्ध हूं।",
        "क्या आप जब मौका मिले तो मुझे वह दस्तावेज़ भेज सकते हैं? बिल्कुल जल्दी नहीं है।",
        "आइए अगले सप्ताह कॉफी के लिए मिलें और बातचीत करें। हमने बात किए हुए बहुत समय हो गया है।",
        "मैं आपके सुझाव से पूरी तरह सहमत हूं। मुझे यह एक उत्कृष्ट योजना लगती है।",
        "जवाब देने में देरी के लिए क्षमा करें। मैं हाल ही में काम से अविश्वसनीय रूप से व्यस्त रहा हूं।",
        "क्या आपने नवीनतम अपडेट देखा? यह वास्तव में दिलचस्प लगता है और जांचने लायक है।",
        "कृपया मुझे बताएं यदि आपको अतिरिक्त जानकारी या किसी चीज़ में मदद की आवश्यकता है।",
        "यह अद्भुत समाचार है! मैं आपके लिए बहुत खुश हूं और अधिक विवरण सुनने के लिए उत्साहित हूं।",
        "सबसे पहले, ओवन को 180 डिग्री सेल्सियस तक प्रीहीट करें। फिर एक बड़े कटोरे में आटा और चीनी को सावधानी से मिलाएं।",
        "अपना पासवर्ड रीसेट करने के लिए, नीचे पासवर्ड भूल गए लिंक पर क्लिक करें और अपना ईमेल पता दर्ज करें।",
        "पावर बटन को ठीक 3 सेकंड के लिए दबाएं और रखें और फिर नीले संकेतक प्रकाश की प्रतीक्षा करें।",
        "एप्लिकेशन सेटिंग्स मेनू खोलें और सुरक्षा प्राथमिकताएं अनुभाग पर सावधानी से नेविगेट करें।",
        "डिवाइस चालू करने से पहले केबल को पीछे के पैनल पर पोर्ट नंबर दो से कनेक्ट करें।",
        "control S दबाकर या टूलबार में सेव बटन पर क्लिक करके अपने काम को बार-बार सेव करें।",
        "फर्नीचर को असेंबल करने का प्रयास करने से पहले पूरी निर्देश पुस्तिका को पूरी तरह से पढ़ें।",
        "कोई भी प्लगइन इंस्टॉल करने से पहले आधिकारिक वेबसाइट से नवीनतम सॉफ्टवेयर अपडेट डाउनलोड करें।",
        "निर्माण परियोजना के लिए सटीक आयाम सुनिश्चित करने के लिए दो बार मापें और एक बार काटें।",
        "सर्वोत्तम परिणामों और हर बार पूर्ण स्थिरता के लिए नुस्खा निर्देशों का सटीक रूप से पालन करें।",
    ],
}


def generate_large_corpus(samples_per_lang=500):
    """Generate large corpus with variations."""
    corpus = {}
    
    for lang, templates in TEXTS.items():
        samples = []
        samples_per_template = samples_per_lang // len(templates)
        
        for template in templates:
            for i in range(samples_per_template):
                # Variations
                varied = template
                
                # Vary numbers
                if '180' in varied:
                    varied = varied.replace('180', str(random.choice([160, 170, 180, 190, 200])))
                if '3' in varied and 'three' not in varied.lower():
                    varied = varied.replace('3', str(random.choice([2, 3, 4, 5])))
                if 'five' in varied.lower():
                    varied = varied.replace('five', random.choice(['three', 'four', 'five', 'six']))
                
                # Vary years/times
                if 'years' in varied:
                    if random.random() < 0.3:
                        varied = varied.replace('years', 'decades')
                
                # Occasional punctuation variation
                if random.random() < 0.2:
                    if not varied.endswith('.'):
                        varied += '.'
                
                samples.append(varied)
        
        # Shuffle
        random.shuffle(samples)
        corpus[lang] = samples[:samples_per_lang]  # Exact count
    
    return corpus


def split_corpus(corpus, train_ratio=0.8):
    """Split into train/test."""
    train = {}
    test = {}
    
    for lang, samples in corpus.items():
        split_idx = int(len(samples) * train_ratio)
        train[lang] = samples[:split_idx]
        test[lang] = samples[split_idx:]
    
    return train, test


def save_corpus(corpus, output_dir, split_name):
    """Save to files."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {}
    
    for lang, samples in corpus.items():
        lang_file = split_dir / f"{lang.lower()}.txt"
        with open(lang_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample + '\n')
        
        manifest[lang] = {
            'file': lang_file.name,
            'num_samples': len(samples),
            'num_chars': sum(len(s) for s in samples),
            'num_words': sum(len(s.split()) for s in samples),
        }
    
    with open(split_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {split_name}: {len(corpus)} languages, {sum(len(s) for s in corpus.values())} samples")


def main():
    """Generate large corpus."""
    random.seed(42)
    
    output_dir = Path('benchmarks/data/multilingual')
    
    print("\n" + "="*80)
    print("Generating Large Multilingual Corpus (500 samples/language)")
    print("="*80 + "\n")
    
    # Generate
    print("Generating samples...")
    corpus = generate_large_corpus(samples_per_lang=500)
    
    print(f"Generated:")
    for lang, samples in corpus.items():
        chars = sum(len(s) for s in samples)
        words = sum(len(s.split()) for s in samples)
        print(f"  {lang:<12}: {len(samples):>3} samples, {chars:>6} chars, {words:>5} words")
    
    # Split
    print("\nSplitting train/test...")
    train, test = split_corpus(corpus, train_ratio=0.8)
    
    # Save
    print()
    save_corpus(train, output_dir, 'train')
    save_corpus(test, output_dir, 'test')
    
    total_train = sum(len(s) for s in train.values())
    total_test = sum(len(s) for s in test.values())
    
    print(f"\n✓ Total train samples: {total_train}")
    print(f"✓ Total test samples: {total_test}")
    print(f"✓ Corpus saved to {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()