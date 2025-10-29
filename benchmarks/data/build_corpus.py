"""Generate multilingual corpus for experiments."""

import json
import random
from pathlib import Path

# Sample texts per language (3 domains × 3 templates each = 9 per language)
TEXTS = {
    'English': [
        # News
        "The government announced new policies yesterday affecting millions of citizens.",
        "Scientists discovered breakthrough technology in renewable energy research today.",
        "Economic indicators showed recovery with unemployment dropping significantly.",
        # Conversation
        "Hello! How are you doing today? I hope everything is going well for you.",
        "Thanks for your help yesterday. I really appreciate it very much.",
        "What time works best for you? I'm available after three pm tomorrow.",
        # Instructions
        "First, heat the oven to 180 degrees. Then mix the flour and sugar together.",
        "To reset your password, click the forgot password link below.",
        "Press the power button for 3 seconds and wait for the blue light.",
    ],
    'Spanish': [
        "El gobierno anunció nuevas políticas ayer afectando a millones de ciudadanos.",
        "Los científicos descubrieron tecnología revolucionaria en energía renovable hoy.",
        "Los indicadores económicos mostraron recuperación con el desempleo cayendo significativamente.",
        "¡Hola! ¿Cómo estás hoy? Espero que todo vaya muy bien para ti.",
        "Gracias por tu ayuda ayer. Lo aprecio muchísimo de verdad.",
        "¿Qué hora te viene mejor? Estoy disponible después de las tres de la tarde mañana.",
        "Primero, calienta el horno a 180 grados. Luego mezcla la harina y el azúcar juntos.",
        "Para restablecer tu contraseña, haz clic en el enlace de contraseña olvidada abajo.",
        "Presiona el botón de encendido durante 3 segundos y espera la luz azul.",
    ],
    'French': [
        "Le gouvernement a annoncé de nouvelles politiques hier affectant des millions de citoyens.",
        "Les scientifiques ont découvert une technologie révolutionnaire dans la recherche énergétique aujourd'hui.",
        "Les indicateurs économiques ont montré une reprise avec le chômage diminuant considérablement.",
        "Bonjour! Comment allez-vous aujourd'hui? J'espère que tout va très bien pour vous.",
        "Merci pour votre aide hier. Je l'apprécie énormément vraiment.",
        "Quelle heure vous convient le mieux? Je suis disponible après trois heures demain.",
        "D'abord, chauffez le four à 180 degrés. Ensuite mélangez la farine et le sucre ensemble.",
        "Pour réinitialiser votre mot de passe, cliquez sur le lien mot de passe oublié ci-dessous.",
        "Appuyez sur le bouton d'alimentation pendant 3 secondes et attendez la lumière bleue.",
    ],
    'German': [
        "Die Regierung kündigte gestern neue Richtlinien an, die Millionen von Bürgern betreffen.",
        "Wissenschaftler entdeckten heute revolutionäre Technologie in der Energieforschung.",
        "Wirtschaftsindikatoren zeigten Erholung mit signifikant sinkender Arbeitslosigkeit.",
        "Hallo! Wie geht es Ihnen heute? Ich hoffe, dass alles sehr gut für Sie läuft.",
        "Danke für Ihre Hilfe gestern. Ich schätze es wirklich sehr.",
        "Welche Zeit passt Ihnen am besten? Ich bin morgen nach drei Uhr verfügbar.",
        "Zuerst heizen Sie den Ofen auf 180 Grad vor. Dann vermischen Sie Mehl und Zucker zusammen.",
        "Um Ihr Passwort zurückzusetzen, klicken Sie auf den Link Passwort vergessen unten.",
        "Drücken Sie den Ein-Aus-Knopf für 3 Sekunden und warten Sie auf das blaue Licht.",
    ],
    'Portuguese': [
        "O governo anunciou novas políticas ontem afetando milhões de cidadãos.",
        "Cientistas descobriram tecnologia revolucionária em pesquisa energética hoje.",
        "Indicadores econômicos mostraram recuperação com desemprego caindo significativamente.",
        "Olá! Como você está hoje? Espero que tudo esteja indo muito bem para você.",
        "Obrigado pela sua ajuda ontem. Eu realmente agradeço muito.",
        "Que horas funcionam melhor para você? Estou disponível depois das três da tarde amanhã.",
        "Primeiro, aqueça o forno a 180 graus. Depois misture a farinha e o açúcar juntos.",
        "Para redefinir sua senha, clique no link esqueceu a senha abaixo.",
        "Pressione o botão de energia por 3 segundos e aguarde a luz azul.",
    ],
    'Italian': [
        "Il governo ha annunciato nuove politiche ieri che colpiscono milioni di cittadini.",
        "Gli scienziati hanno scoperto tecnologia rivoluzionaria nella ricerca energetica oggi.",
        "Gli indicatori economici hanno mostrato ripresa con disoccupazione in calo significativo.",
        "Ciao! Come stai oggi? Spero che tutto vada molto bene per te.",
        "Grazie per il tuo aiuto ieri. Lo apprezzo davvero moltissimo.",
        "Che ora va meglio per te? Sono disponibile dopo le tre del pomeriggio domani.",
        "Prima, riscalda il forno a 180 gradi. Poi mescola la farina e lo zucchero insieme.",
        "Per reimpostare la password, clicca sul link password dimenticata qui sotto.",
        "Premi il pulsante di accensione per 3 secondi e attendi la luce blu.",
    ],
    'Russian': [
        "Правительство объявило вчера о новой политике, затрагивающей миллионы граждан.",
        "Ученые сегодня открыли революционную технологию в исследовании энергетики.",
        "Экономические показатели показали восстановление со значительным снижением безработицы.",
        "Привет! Как дела сегодня? Надеюсь, у тебя все очень хорошо.",
        "Спасибо за твою помощь вчера. Я действительно очень ценю это.",
        "Какое время тебе лучше подходит? Я свободен завтра после трех часов дня.",
        "Сначала разогрейте духовку до 180 градусов. Затем смешайте муку и сахар вместе.",
        "Чтобы сбросить пароль, нажмите на ссылку забыли пароль ниже.",
        "Нажмите кнопку питания на 3 секунды и дождитесь синего света.",
    ],
    'Arabic': [
        "أعلنت الحكومة أمس عن سياسات جديدة تؤثر على ملايين المواطنين.",
        "اكتشف العلماء اليوم تكنولوجيا ثورية في أبحاث الطاقة المتجددة.",
        "أظهرت المؤشرات الاقتصادية تعافيًا مع انخفاض البطالة بشكل كبير.",
        "مرحباً! كيف حالك اليوم؟ أتمنى أن يكون كل شيء على ما يرام معك.",
        "شكراً على مساعدتك أمس. أنا أقدر ذلك حقاً كثيراً.",
        "ما الوقت الأنسب لك؟ أنا متاح بعد الساعة الثالثة مساءً غداً.",
        "أولاً، سخن الفرن إلى 180 درجة. ثم اخلط الدقيق والسكر معاً.",
        "لإعادة تعيين كلمة المرور، انقر على رابط نسيت كلمة المرور أدناه.",
        "اضغط على زر الطاقة لمدة 3 ثوانٍ وانتظر الضوء الأزرق.",
    ],
    'Chinese': [
        "政府昨天宣布了影响数百万公民的新政策。",
        "科学家今天在可再生能源研究中发现了突破性技术。",
        "经济指标显示复苏，失业率大幅下降。",
        "你好！你今天怎么样？我希望你一切都很顺利。",
        "谢谢你昨天的帮助。我真的非常感激。",
        "什么时间对你最合适？我明天下午三点以后有空。",
        "首先，将烤箱加热到180度。然后将面粉和糖混合在一起。",
        "要重置密码，请点击下面的忘记密码链接。",
        "按住电源按钮3秒钟，等待蓝灯亮起。",
    ],
    'Hindi': [
        "सरकार ने कल नई नीतियों की घोषणा की जो लाखों नागरिकों को प्रभावित करती हैं।",
        "वैज्ञानिकों ने आज नवीकरणीय ऊर्जा अनुसंधान में क्रांतिकारी तकनीक की खोज की।",
        "आर्थिक संकेतकों ने बेरोजगारी में महत्वपूर्ण गिरावट के साथ सुधार दिखाया।",
        "नमस्ते! आज आप कैसे हैं? मुझे उम्मीद है कि आपके लिए सब कुछ बहुत अच्छा चल रहा है।",
        "कल आपकी मदद के लिए धन्यवाद। मैं वास्तव में इसकी बहुत सराहना करता हूं।",
        "आपके लिए कौन सा समय सबसे अच्छा है? मैं कल दोपहर तीन बजे के बाद उपलब्ध हूं।",
        "पहले, ओवन को 180 डिग्री तक गर्म करें। फिर आटा और चीनी को एक साथ मिलाएं।",
        "अपना पासवर्ड रीसेट करने के लिए, नीचे पासवर्ड भूल गए लिंक पर क्लिक करें।",
        "पावर बटन को 3 सेकंड के लिए दबाएं और नीली लाइट की प्रतीक्षा करें।",
    ],
}


def generate_corpus(samples_per_template=15):
    """Generate corpus by repeating templates with variation."""
    corpus = {}
    
    for lang, templates in TEXTS.items():
        samples = []
        for template in templates:
            # Add variations (numbers, slight changes)
            for i in range(samples_per_template):
                # Simple variation: replace numbers
                varied = template
                if '3' in varied:
                    varied = varied.replace('3', str(random.choice([2, 3, 4, 5])))
                if '180' in varied:
                    varied = varied.replace('180', str(random.choice([160, 170, 180, 190, 200])))
                samples.append(varied)
        
        # Shuffle
        random.shuffle(samples)
        corpus[lang] = samples
    
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
        # Save language file
        lang_file = split_dir / f"{lang.lower()}.txt"
        with open(lang_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample + '\n')
        
        manifest[lang] = {
            'file': lang_file.name,
            'num_samples': len(samples),
            'num_words': sum(len(s.split()) for s in samples),
        }
    
    # Save manifest
    with open(split_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {split_name}: {len(corpus)} languages, {sum(len(s) for s in corpus.values())} samples")


def main():
    """Generate corpus."""
    random.seed(42)
    
    output_dir = Path('benchmarks/data/multilingual')
    
    print("\n" + "="*80)
    print("Generating Multilingual Corpus")
    print("="*80 + "\n")
    
    # Generate (~135 samples per language = 9 templates × 15 repetitions)
    corpus = generate_corpus(samples_per_template=15)
    
    # Split
    train, test = split_corpus(corpus, train_ratio=0.8)
    
    # Save
    save_corpus(train, output_dir, 'train')
    save_corpus(test, output_dir, 'test')
    
    print(f"\n✓ Corpus saved to {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()