"""
scripts/generate_complex_dataset.py
----------------------------------
Generates a multi-label synthetic dataset for Mindify containing both single-intent
and compound multi-intent sentences across 36 life taxonomy domains.
Exports to data/multilabel_training_data.csv with 'text' and 'labels' columns.
"""

import os
import random
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "multilabel_training_data.csv")

# Sentence templates
TEMPLATES_A = [
    "I want to {verb} {noun}",
    "My goal is to {verb} {noun}",
    "I intend to manifest {noun}",
    "I choose to {verb} {noun} every day",
    "I am actively working to {verb} my {noun}",
    "I am calling in {noun}",
    "I am focused on how to {verb} my {noun}",
    "I am ready to {verb} {noun}",
    "My focus is to {verb} {noun}",
    "I am dedicated to building better {noun}"
]

TEMPLATES_B = [
    "I want to {verb} {noun}",
    "I goal is to {verb} {noun}",
    "I choose to {verb} {noun}",
    "I am working to {verb} {noun}",
    "I am focused on how to {verb} {noun}",
    "I am ready to {verb} {noun}"
]

CONNECTORS = [
    " and ",
    " while also trying to ",
    " plus ",
    " so that I can ",
    " along with ",
    " in order to ",
    " as well as ",
    " while working to "
]

# Vocabulary mapping across categories
CATEGORY_VOCAB = {
    "Financial Goals & Wealth": {
        "verbs": ["increase", "save", "invest in", "build", "pay off", "manifest", "accumulate", "grow", "secure"],
        "nouns": ["unexpected income", "my savings account", "credit card debt", "passive income streams", "a high yield portfolio", "financial freedom", "an emergency fund"]
    },
    "Career & Professional Growth": {
        "verbs": ["achieve", "land", "earn", "negotiate", "expand", "lead", "build", "master"],
        "nouns": ["a senior engineering role", "a major promotion", "a higher salary", "my own startup company", "a corporate leadership position", "my dream job"]
    },
    "Relationships & Social Connections": {
        "verbs": ["attract", "deepen", "build", "nurture", "connect with", "heal", "find"],
        "nouns": ["my soulmate and life partner", "deep meaningful friendships", "a loving romantic connection", "family relationships", "a supportive community"]
    },
    "Health, Wellness & Lifestyle": {
        "verbs": ["improve", "prioritize", "restore", "build", "maintain", "heal", "strengthen"],
        "nouns": ["my physical fitness", "deep restful sleep", "a clean organic diet", "my mental peace", "a consistent workout routine", "body strength and stamina"]
    },
    "Environment & Material Upgrades": {
        "verbs": ["buy", "upgrade", "decorate", "renovate", "declutter", "create", "find"],
        "nouns": ["my dream home", "a modern minimalist apartment", "my ideal electric vehicle", "a quiet home office setup", "a cozy living space"]
    },
    "Mindset & Emotional Well-Being": {
        "verbs": ["cultivate", "release", "build", "embrace", "strengthen", "heal", "master"],
        "nouns": ["unshakable self-confidence", "inner peace and calm", "past emotional trauma", "a resilient growth mindset", "self-compassion and grace"]
    },
    "Personal Habits & Time Management": {
        "verbs": ["master", "eliminate", "establish", "optimize", "stop", "maintain", "build"],
        "nouns": ["a productive morning routine", "procrastination and screen time", "work-life balance", "time management skills", "a daily journaling habit"]
    },
    "Creativity, Hobbies & Learning": {
        "verbs": ["learn", "master", "create", "publish", "practice", "explore", "produce"],
        "nouns": ["a new foreign language", "digital video editing skills", "my personal blog and podcast", "oil painting and drawing", "acoustic guitar playing"]
    },
    "Social Impact & Contribution": {
        "verbs": ["donate to", "volunteer at", "mentor", "launch", "support", "give back to"],
        "nouns": ["local animal shelters", "community youth programs", "environmental conservation efforts", "young aspiring entrepreneurs", "charitable organizations"]
    },
    "Education & Academic Milestones": {
        "verbs": ["pass", "graduate with", "win", "complete", "study for", "excel in"],
        "nouns": ["my university final exams", "top honors in my degree", "an academic scholarship", "my master's thesis research", "a competitive certification"]
    },
    "Independence & Freedom": {
        "verbs": ["achieve", "build", "transition to", "live as", "create", "enjoy"],
        "nouns": ["complete location independence", "a remote digital nomad lifestyle", "a freelance consulting business", "autonomous working hours", "geographic freedom"]
    },
    "Spiritual Growth & Alignment": {
        "verbs": ["align with", "deepen", "connect with", "manifest", "trust", "practice"],
        "nouns": ["my higher self and intuition", "daily spiritual meditation", "universal energy and flow", "my soul's divine purpose", "gratitude and presence"]
    },
    "Major Life & Family Milestones": {
        "verbs": ["celebrate", "welcome", "buy", "plan", "start", "prepare for"],
        "nouns": ["a beautiful wedding ceremony", "our first newborn baby", "family property and land", "a happy peaceful retirement", "our dream family home"]
    },
    "Legal & Administrative Resolution": {
        "verbs": ["resolve", "obtain", "secure", "settle", "finalize", "complete"],
        "nouns": ["my permanent residency green card", "a favorable legal dispute settlement", "official passport and visa approval", "my business trademark registration"]
    },
    "Intellectual & Cognitive Development": {
        "verbs": ["enhance", "master", "sharpen", "expand", "unlock", "develop"],
        "nouns": ["my memory and recall", "deep flow state focus", "bilingual language skills", "speed reading techniques", "cognitive capacity and mental speed"]
    },
    "Athletic & Physical Performance": {
        "verbs": ["shatter", "achieve", "train for", "win", "improve", "build"],
        "nouns": ["my powerlifting PR", "a new personal record", "a full marathon race", "the regional championship", "athletic agility and speed"]
    },
    "Eco-Conscious & Sustainable Living": {
        "verbs": ["build", "transition to", "install", "cultivate", "embrace", "practice"],
        "nouns": ["an off-grid sustainable cabin", "a zero-waste eco-conscious lifestyle", "solar energy power systems", "a thriving organic homesteading garden"]
    },
    "Long-Term Relationship Longevity": {
        "verbs": ["nurture", "deepen", "celebrate", "build", "maintain", "strengthen"],
        "nouns": ["our golden anniversary milestone", "profound emotional intimacy", "harmonious co-parenting practices", "our journey of aging together"]
    },
    "Crisis Recovery & Fresh Starts": {
        "verbs": ["rebuild", "heal from", "overcome", "relocate to", "navigate", "reclaim"],
        "nouns": ["a fresh start after my layoff", "financial recovery after bankruptcy", "emotional healing post-breakup", "a new life as I relocate"]
    },
    "Digital Detox & Tech Balance": {
        "verbs": ["reduce", "unplug from", "limit", "establish", "practice", "reclaim"],
        "nouns": ["daily screen time and scrolling", "social media addiction", "digital distraction habits", "a healthy technology boundary"]
    },
    "Mindful Nutrition & Culinary Arts": {
        "verbs": ["master", "cook", "prepare", "adopt", "enjoy", "nourish"],
        "nouns": ["mindful eating and conscious meals", "organic home-cooked recipes", "culinary arts and cooking skills", "a balanced whole-food diet"]
    },
    "Travel, Exploration & Adventure": {
        "verbs": ["explore", "travel to", "experience", "plan", "embark on", "discover"],
        "nouns": ["exotic international destinations", "an epic backpacking adventure", "new cultures and languages", "scenic mountain trekking routes"]
    },
    "Parenting & Child Development": {
        "verbs": ["raise", "nurture", "guide", "support", "foster", "practice"],
        "nouns": ["happy resilient children", "gentle parenting techniques", "early child development milestones", "strong parent-child emotional bonds"]
    },
    "Sleep, Rest & Recovery Science": {
        "verbs": ["optimize", "improve", "achieve", "establish", "prioritize", "master"],
        "nouns": ["deep REM and slow-wave sleep", "a relaxing evening wind-down routine", "circadian rhythm alignment", "restful sleep hygiene habits"]
    },
    "Generational Legacy & Ancestral Healing": {
        "verbs": ["build", "establish", "manifest", "break", "heal", "secure"],
        "nouns": ["generational wealth and trust funds", "our family empire and heritage", "deep ancestral healing", "breaking old family curses"]
    },
    "Shadow Work & Deep Unconscious Integration": {
        "verbs": ["integrate", "embrace", "heal", "face", "transmute", "release"],
        "nouns": ["repressed shadow traits and dark aspects", "deep shame and hidden fears", "karmic loops and patterns", "my unconscious mind and instincts"]
    },
    "Civic Leadership & Systemic Change": {
        "verbs": ["lead", "spearhead", "advocate for", "run for", "champion", "enact"],
        "nouns": ["elected office and civic leadership", "systemic change and policy reform", "human rights and social justice", "community advocacy and reform"]
    },
    "Radical Detachment & Monk Mode": {
        "verbs": ["detach from", "enter", "practice", "embrace", "execute", "disappear into"],
        "nouns": ["radical detachment from outcomes", "monk mode solitary focus", "a strict dopamine detox", "ascetic minimalism and anonymity"]
    },
    "Interpersonal Severing & Deep Closure": {
        "verbs": ["sever", "cut", "achieve", "release", "forgive", "walk away from"],
        "nouns": ["energetic cords with toxic partners", "permanent closure from abusive exes", "unhealthy ties with toxic parents", "peaceful divorce agreements"]
    },
    "Advanced Esoteric & Metaphysical Mastery": {
        "verbs": ["master", "experience", "shift", "clear", "align with", "manifest"],
        "nouns": ["lucid dreaming and astral projection", "quantum jumping and reality shifting", "5D consciousness and flow", "somatic energy clearing"]
    },
    "Neurodivergence & Executive Function": {
        "verbs": ["overcome", "regulate", "manage", "unmask", "navigate", "harness"],
        "nouns": ["executive dysfunction and focus", "ADHD hyperfocus and activation energy", "sensory overload and calm", "autistic burnout and recovery"]
    },
    "Everyday Adulting & Life Admin": {
        "verbs": ["conquer", "complete", "organize", "manage", "file", "pass"],
        "nouns": ["annual taxes and paperwork", "my driving test exam", "my credit score improvement", "inbox zero and email admin"]
    },
    "Early Parenthood & Childcare": {
        "verbs": ["navigate", "master", "practice", "manage", "nurture", "support"],
        "nouns": ["postpartum recovery and mental health", "toddler tantrums and meltdowns", "gentle parenting techniques", "infant sleep training routines"]
    },
    "Midlife Transitions & Empty Nesting": {
        "verbs": ["embrace", "navigate", "reinvent", "downsize", "welcome", "step into"],
        "nouns": ["this empty nest season of freedom", "menopause and physical midlife changes", "my identity in midlife", "aging gracefully with vitality"]
    },
    "Tech, Gear & Digital Lifestyle": {
        "verbs": ["build", "upgrade", "optimize", "achieve", "customize", "setup"],
        "nouns": ["my dream PC build and setup", "the ultimate battlestation desk", "high rank in competitive gaming", "smart home automation systems"]
    },
    "Micro-Hobbies & Crafting Mastery": {
        "verbs": ["master", "create", "craft", "bake", "sew", "knit"],
        "nouns": ["sourdough bread baking skills", "custom woodworking projects", "knitting and sewing projects", "DIY crafting and handmade art"]
    },
    "Elder Care & Family Stewardship": {
        "verbs": ["navigate", "manage", "support", "balance", "honor", "care for"],
        "nouns": ["aging parents and elder care", "caregiver burnout and self-care", "nursing home arrangements", "estate planning and family affairs", "dementia care and role reversal"]
    },
    "Workplace Dynamics & Colleague Boundaries": {
        "verbs": ["navigate", "manage", "establish", "handle", "maintain", "protect"],
        "nouns": ["a toxic boss and micromanagement", "office politics and drama", "HR reporting and compliance", "healthy boundaries to log off on time", "passive-aggressive coworkers"]
    },
    "Chronic Illness & Invisible Disability": {
        "verbs": ["manage", "navigate", "advocate for", "pace", "honor", "heal from"],
        "nouns": ["autoimmune condition and flare-ups", "chronic pain management", "long covid recovery and fatigue", "daily spoon theory energy pacing", "medical advocacy and patient rights"]
    },
    "Substance Recovery & Vice Cessation": {
        "verbs": ["achieve", "maintain", "quit", "overcome", "commit to", "embrace"],
        "nouns": ["total alcohol sobriety and recovery", "quitting smoking and nicotine", "relapse prevention strategies", "AA meetings and 12 step recovery", "a clean and sober lifestyle"]
    },
    "Event Planning & Milestone Hosting": {
        "verbs": ["plan", "host", "organize", "coordinate", "manage", "execute"],
        "nouns": ["wedding planning and vendor coordination", "hosting holiday family dinners", "family reunion logistics and catering", "RSVP tracking and event budgets", "social battery balance during events"]
    },
    "Homeownership & Property Maintenance": {
        "verbs": ["manage", "complete", "handle", "repair", "execute", "maintain"],
        "nouns": ["home renovation and contractor work", "roof leaks and emergency repairs", "yard work and landscaping maintenance", "DIY home improvement projects", "property tax and home insurance"]
    },
    "Adult Friendship & Platonic Intimacy": {
        "verbs": ["build", "nurture", "make", "cultivate", "deepen", "connect with"],
        "nouns": ["making new adult friends", "deep platonic intimacy and trust", "my inner circle of best friends", "long-distance friendship connections", "supportive chosen family"]
    },
    "Intuitive Eating & Diet Culture Recovery": {
        "verbs": ["practice", "adopt", "rebuild", "embrace", "heal", "unlearn"],
        "nouns": ["intuitive eating and body wisdom", "food guilt and binge eating recovery", "diet culture unlearning and freedom", "food neutrality and peaceful meals", "stopping calorie counting and restriction"]
    },
    "Hard Conversations & Conflict Navigation": {
        "verbs": ["navigate", "have", "conduct", "master", "speak", "address"],
        "nouns": ["hard conversations with direct honesty", "conflict resolution and clarity", "direct communication without fear", "passive-aggressive behavior and boundaries", "speaking up for my boundaries"]
    },
    "Guilt-Free Rest & Hustle Culture Detox": {
        "verbs": ["embrace", "practice", "unlearn", "enjoy", "prioritize", "reclaim"],
        "nouns": ["productivity guilt and constant busywork", "hustle culture detox and rest", "the joy of doing nothing", "true leisure and restful weekends", "resting without feeling guilty"]
    },
    "Career Re-entry & Late-in-Life Pivots": {
        "verbs": ["navigate", "execute", "embrace", "pivot to", "start", "re-enter"],
        "nouns": ["career re-entry after a long gap", "starting over in a new industry", "going back to school at 40 and beyond", "a late-in-life career pivot", "re-entering the professional workforce"]
    },
    "Nervous System Regulation & Somatic Healing": {
        "verbs": ["regulate", "soothe", "practice", "master", "unclench", "activate"],
        "nouns": ["fight-or-flight nervous system response", "somatic healing and tension release", "vagus nerve stimulation techniques", "soothing panic attacks and anxiety", "unclenching my jaw and shoulders"]
    },
    "Modern Dating & Romantic Vulnerability": {
        "verbs": ["navigate", "embrace", "date", "open", "trust", "build"],
        "nouns": ["modern dating apps and swiping", "ghosting and rejection resilience", "first date nerves and authenticity", "romantic vulnerability and trust", "healing rejection and dating confidence"]
    },
    "Academic Pressure & High-Stakes Testing": {
        "verbs": ["pass", "master", "conquer", "excel in", "prepare for", "ace"],
        "nouns": ["my medical board exams and tests", "my master's thesis defense presentation", "entrance exams like MCAT and GRE", "test anxiety and exam pressure", "intensive study schedules and focus"]
    },
    "Sleep Optimization & Circadian Health": {
        "verbs": ["optimize", "improve", "master", "establish", "align", "calm"],
        "nouns": ["insomnia and night racing thoughts", "a consistent sleep schedule", "waking up early with vibrant energy", "deep REM sleep and restoration", "circadian rhythm alignment"]
    },
    "Driving Confidence & Commute Anxiety": {
        "verbs": ["build", "master", "overcome", "navigate", "enjoy", "handle"],
        "nouns": ["driving anxiety and panic on highways", "highway driving confidence and speed", "driver's license test and parallel parking", "scenic road trips and long drives", "traffic congestion and commute stress"]
    },
    "Language Acquisition & Accent Confidence": {
        "verbs": ["learn", "master", "speak", "build", "expand", "overcome"],
        "nouns": ["learning a new foreign language", "speaking confidently with native speakers", "accent insecurity and pronunciation", "bilingual fluency and vocabulary", "language acquisition and grammar"]
    },
    "Physical Rehabilitation & Physiotherapy": {
        "verbs": ["recover from", "heal", "restore", "practice", "relieve", "manage"],
        "nouns": ["physical therapy exercises and routines", "recovering from surgery or injury", "daily mobility stretches and exercises", "knee pain and back pain relief", "restoring full body joint mobility"]
    },
    "Solo Living & Domestic Independence": {
        "verbs": ["embrace", "enjoy", "master", "build", "create", "cook"],
        "nouns": ["living alone in my empty apartment", "cooking healthy meals for one", "solo living domestic independence", "peaceful solitary home sanctuary", "enjoying quiet alone time"]
    },
    "Time Management & Punctuality Discipline": {
        "verbs": ["master", "overcome", "leave", "arrive", "manage", "build"],
        "nouns": ["chronic lateness and time blindness", "leaving on time for appointments", "punctuality discipline and timing", "managing the clock with calm focus", "rushing and schedule anxiety"]
    },
    "Financial Literacy & Basic Investing": {
        "verbs": ["build", "master", "invest in", "understand", "learn", "grow"],
        "nouns": ["index funds and 401k retirement plans", "basic financial literacy and compounding", "compound interest and wealth accumulation", "checking my bank account without anxiety", "investing basics and stock portfolios"]
    },
    "Digital Declutter & Cybersecurity Hygiene": {
        "verbs": ["achieve", "organize", "master", "clean", "secure", "unsubscribe from"],
        "nouns": ["inbox zero and email organization", "digital declutter and photo storage", "strong password hygiene and security", "unsubscribing from marketing emails", "cybersecurity hygiene and identity safety"]
    },
    "Personal Safety & Situational Awareness": {
        "verbs": ["build", "maintain", "practice", "enhance", "navigate", "protect"],
        "nouns": ["situational awareness when walking alone", "home security systems and safety", "managing hyper-vigilance and fear", "feeling safe in my neighborhood", "defending personal safety boundaries"]
    },
    "Retirement & Golden Years Transition": {
        "verbs": ["step into", "embrace", "navigate", "manage", "enjoy", "structure"],
        "nouns": ["retiring comfortably on a fixed income", "golden years freedom and relaxation", "pension management and retirement funds", "structuring unstructured time after work", "life after a long professional career"]
    },
    "Freelancing & Solopreneurship Survival": {
        "verbs": ["protect", "raise", "price", "manage", "resolve", "assert"],
        "nouns": ["overdue invoices and late client payments", "scope creep boundaries and client demands", "freelance pricing and raising my rates", "solopreneur business autonomy and freedom", "pricing my worth without apology"]
    },
    "Immigration, Visas & Bureaucracy": {
        "verbs": ["navigate", "await", "prepare for", "complete", "handle", "manage"],
        "nouns": ["green card application and visa processing", "embassy interview and visa approval", "passport renewals and immigration paperwork", "bureaucratic waiting periods and stress", "work visa sponsorship and legal status"]
    },
    "Shared Living & Roommate Dynamics": {
        "verbs": ["cultivate", "enforce", "manage", "split", "respect", "communicate"],
        "nouns": ["roommate communication and chore charts", "splitting utility bills and shared expenses", "shared apartment quiet hours and rules", "living with roommates and healthy boundaries", "peaceful shared living environment"]
    },
    "Pre-Marital & Financial Merging": {
        "verbs": ["build", "combine", "discuss", "align", "create", "unite"],
        "nouns": ["prenup discussions and pre-marital alignment", "combining bank accounts and joint finances", "financial transparency with my partner", "joint bank account management and savings", "wedding budget planning and shared goals"]
    },
    "Urban Commuting & Public Transit": {
        "verbs": ["protect", "navigate", "handle", "regulate", "endure", "transform"],
        "nouns": ["subway delays and rush hour transit", "public transportation sensory exhaustion", "daily urban commuting stress and noise", "crowded train platforms and bus rides", "regulating my nervous system during transit"]
    },
    "Independent Publishing & Creative Launch": {
        "verbs": ["step into", "launch", "publish", "share", "run", "overcome"],
        "nouns": ["self-publishing a book or creative work", "Kickstarter campaigns and indie funding", "launch day vulnerability and excitement", "pressing the publish button with courage", "indie creative release and book launches"]
    },
    "Value & Long-Term Investing": {
        "verbs": ["conduct", "analyze", "master", "build", "evaluate", "read"],
        "nouns": ["fundamental analysis and balance sheets", "ROIC calculation and stock valuation", "long-term investor mindset and patience", "resilient investment portfolio growth", "reading financial statements and balance sheets"]
    },
    "Short-Form Video Production": {
        "verbs": ["master", "edit", "export", "arrange", "optimize", "create"],
        "nouns": ["editing reels and short video clips", "aspect ratio optimization and 9x16 framing", "video timeline editing and transition cuts", "high video quality export settings", "content creation and reel storytelling"]
    },
    "Nature & Wildlife Ecotourism": {
        "verbs": ["explore", "reconnect with", "spot", "visit", "trek", "experience"],
        "nouns": ["hill stations and summer season travel", "wildlife spotting in national parks", "nature trails and eco-hiking routes", "ecotourism and sustainable travel spots", "reconnecting with nature and wildlife"]
    },
    "Independent Software Development": {
        "verbs": ["build", "transform", "code", "design", "deploy", "solve"],
        "nouns": ["problem statement and software architecture", "building an app from scratch", "clean coding standards and practices", "cloud deployment and backend APIs", "independent software development tools"]
    },
    "Digital Community Moderation": {
        "verbs": ["protect", "enforce", "moderate", "maintain", "cultivate", "filter"],
        "nouns": ["comment sections and community guidelines", "blocking online trolls and toxic spam", "digital community moderation rules", "online safety and respectful discussion", "protecting digital community spaces"]
    },
    "Personal Vehicle Maintenance": {
        "verbs": ["perform", "check", "maintain", "service", "schedule", "inspect"],
        "nouns": ["routine oil changes and oil filter service", "trusted mechanic visits and vehicle inspections", "checking car mileage and maintenance logs", "tire pressure checks and tire rotation", "personal vehicle routine maintenance"]
    },
    "Tenant & Landlord Navigation": {
        "verbs": ["navigate", "handle", "manage", "submit", "negotiate", "protect"],
        "nouns": ["landlord communications and lease agreements", "rent increase negotiations and tenant rights", "moving apartments and security deposit return", "maintenance requests and lease terms", "protecting my peaceful living space"]
    },
    "Holistic Home Organization": {
        "verbs": ["declutter", "organize", "donate", "swap", "clean", "arrange"],
        "nouns": ["decluttering messy rooms and closet organization", "donating clothes and seasonal wardrobe swaps", "garage storage solutions and neat pantries", "creating a clean peaceful home sanctuary", "holistic home organization routines"]
    },
    "Expert Mentorship & Methodology Study": {
        "verbs": ["study", "master", "find", "analyze", "adopt", "evaluate"],
        "nouns": ["stock-picking frameworks and valuation methodology", "finding an expert mentor and studying styles", "deep domain study and proven frameworks", "investment style research and expert methodology", "accelerating mastery through proven mentorship"]
    },
    "Mindful Media Consumption": {
        "verbs": ["read", "watch", "listen to", "stop", "curate", "protect"],
        "nouns": ["reading more books and deep literature", "documentaries and intentional educational media", "audiobooks and high-quality podcasts", "stopping doom-scrolling and mindless social feeds", "protecting my attention span with mindful media"]
    },
    "Extracurricular Parenting": {
        "verbs": ["manage", "drive", "coordinate", "balance", "support", "organize"],
        "nouns": ["carpool duties and kids sports schedules", "piano lessons and extracurricular activities", "PTA meetings and school drop-off routines", "balancing over-scheduled kids and family time", "supportive parenting for extracurricular growth"]
    },
    "Subscription & Expense Auditing": {
        "verbs": ["audit", "cancel", "negotiate", "identify", "plug", "review"],
        "nouns": ["canceling unused subscriptions and recurring fees", "negotiating monthly utility and internet bills", "identifying hidden fees and auto-pay charges", "meticulous financial expense auditing", "plugging money leaks and maximizing monthly savings"]
    }
}


def generate_single_intent_sample(cat: str) -> tuple[str, str]:
    vocab = CATEGORY_VOCAB[cat]
    template = random.choice(TEMPLATES_A)
    verb = random.choice(vocab["verbs"])
    noun = random.choice(vocab["nouns"])
    text = template.format(verb=verb, noun=noun) + "."
    return text, cat


def generate_multi_intent_sample(cat1: str, cat2: str) -> tuple[str, str]:
    v1 = CATEGORY_VOCAB[cat1]
    v2 = CATEGORY_VOCAB[cat2]

    t1 = random.choice(TEMPLATES_A)
    verb1 = random.choice(v1["verbs"])
    noun1 = random.choice(v1["nouns"])
    part1 = t1.format(verb=verb1, noun=noun1)

    t2 = random.choice(TEMPLATES_B)
    verb2 = random.choice(v2["verbs"])
    noun2 = random.choice(v2["nouns"])
    part2 = t2.format(verb=verb2, noun=noun2)

    connector = random.choice(CONNECTORS)
    text = part1 + connector + part2 + "."
    labels_str = f"{cat1}|{cat2}"
    return text, labels_str


def generate_hard_negative_sample(cat_target: str, cat_sister: str) -> tuple[str, str]:
    """
    Generate a hard negative sample for cat_target that uses vocabulary/noun elements
    from cat_sister (e.g. mixing budgeting words into investing prompts), but is labeled ONLY with cat_target.
    This forces decision boundaries to be sharper and eliminates false multi-label pollution.
    """
    vt = CATEGORY_VOCAB[cat_target]
    vs = CATEGORY_VOCAB[cat_sister]

    template = random.choice(TEMPLATES_A)
    verb = random.choice(vt["verbs"])
    noun_target = random.choice(vt["nouns"])
    noun_sister = random.choice(vs["nouns"])

    blends = [
        f"{template.format(verb=verb, noun=noun_target)} while staying mindful of {noun_sister}.",
        f"{template.format(verb=verb, noun=noun_target)} without letting {noun_sister} distract my focus.",
        f"{template.format(verb=verb, noun=noun_target)} as I manage {noun_sister}."
    ]
    text = random.choice(blends)
    return text, cat_target


def build_dataset() -> pd.DataFrame:
    rows = []
    all_categories = list(CATEGORY_VOCAB.keys())

    # 1. Single-intent samples (100 per category)
    for cat in all_categories:
        for _ in range(100):
            text, label = generate_single_intent_sample(cat)
            rows.append({"text": text, "labels": label})

    # 2. Multi-intent compound samples (1,500 samples)
    for _ in range(1500):
        cat1, cat2 = random.sample(all_categories, 2)
        text, labels_str = generate_multi_intent_sample(cat1, cat2)
        rows.append({"text": text, "labels": labels_str})

    # 3. Hard negative pass: sister category vocabulary mixing (20 samples per category)
    for cat in all_categories:
        sisters = [c for c in all_categories if c != cat]
        for _ in range(20):
            sister_cat = random.choice(sisters)
            text, label = generate_hard_negative_sample(cat, sister_cat)
            rows.append({"text": text, "labels": label})

    df = pd.DataFrame(rows)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = build_dataset()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[Multi-Label Dataset Generator] Exported dataset to: {OUTPUT_CSV}")
    print(f"[Multi-Label Dataset Generator] Total samples: {len(df)}")


if __name__ == "__main__":
    main()
