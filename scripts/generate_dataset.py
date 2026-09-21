"""
scripts/generate_dataset.py
---------------------------
Bootstrap and augment a synthetic manifestation dataset for the 36 life goal taxonomy categories.
Generates 200+ unique, varied sentences per category (Total 7,500+ rows) and exports to data/training_data.csv.
"""

import os
import random
import pandas as pd

# Define target dataset path
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "training_data.csv")

# Expanded sentence templates for maximum syntactic and vocabulary variance
TEMPLATES = [
    "I want to {verb} {noun}.",
    "My goal is to {verb} {noun}.",
    "I intend to manifest {noun}.",
    "I choose to {verb} {noun} every day.",
    "I am actively working to {verb} my {noun}.",
    "I am calling in {noun} into my life.",
    "I am focused on how to {verb} my {noun}.",
    "I am ready to {verb} {noun} with confidence.",
    "My manifestation focus is to {verb} {noun}.",
    "I am dedicated to building better {noun}.",
    "I deserve to {verb} {noun} effortlessly.",
    "Step by step, I will {verb} {noun}.",
    "I am committed to {verb} {noun}.",
    "I need to {verb} {noun} for my growth.",
    "My top priority is to {verb} {noun}.",
    "I am determined to {verb} {noun}.",
    "I welcome the opportunity to {verb} {noun}.",
    "I am fully aligned to {verb} {noun}.",
    "I claim my power to {verb} {noun}.",
    "Every day I take action to {verb} {noun}.",
]

# Vocabulary mappings for all 36 categories
CATEGORY_VOCAB = {
    "Financial Goals & Wealth": {
        "verbs": ["increase", "save", "invest in", "build", "pay off", "manifest", "accumulate", "grow", "secure", "maximize", "generate", "expand"],
        "nouns": [
            "unexpected income", "my savings account", "credit card debt", "passive income streams",
            "a high yield investment portfolio", "financial freedom", "a monthly bonus", "my emergency fund",
            "stock market dividends", "my net worth", "wealth and abundance", "my personal budget",
            "real estate investments", "capital growth", "financial stability"
        ],
        "handcrafted": [
            "I want to save more money every month.",
            "My goal is to clear my credit card debt.",
            "I intend to manifest unexpected income and financial abundance.",
            "I am investing in index funds and long term stocks.",
            "I choose to build a high yield savings account.",
            "I want to double my monthly passive income.",
            "I am ready to achieve total financial independence.",
            "I am paying off my student loans and personal debt.",
            "I attract wealth prosperity and financial security.",
            "My target is to build an emergency fund with six months of expenses.",
            "I manage my financial assets with discipline and wisdom.",
            "I create scalable revenue streams and long term wealth."
        ]
    },
    "Career & Professional Growth": {
        "verbs": ["achieve", "land", "earn", "negotiate", "expand", "lead", "build", "master", "secure", "advance", "accelerate", "scale"],
        "nouns": [
            "a senior engineering role", "a major promotion", "a higher salary", "my own startup company",
            "a corporate leadership position", "a successful business project", "my dream job",
            "a client contract", "my professional network", "executive management skills",
            "industry recognition", "a lucrative partnership", "career advancement opportunities"
        ],
        "handcrafted": [
            "I want to get a promotion at my software job.",
            "My goal is to negotiate a higher annual salary.",
            "I intend to launch my own independent business.",
            "I am landing my dream role as a lead product designer.",
            "I choose to build a strong professional reputation.",
            "I am scaling my company revenue and acquiring new clients.",
            "I want to become a recognized industry leader.",
            "I am expanding my professional network with key mentors.",
            "I am mastering advanced technical leadership skills.",
            "My target is to secure a remote senior developer offer.",
            "I step into executive leadership with confidence and clarity.",
            "I execute high impact projects that advance my career."
        ]
    },
    "Relationships & Social Connections": {
        "verbs": ["attract", "deepen", "build", "nurture", "connect with", "heal", "find", "cultivate", "foster", "strengthen", "embrace"],
        "nouns": [
            "my soulmate and life partner", "deep meaningful friendships", "a loving romantic connection",
            "family relationships", "a supportive community", "healthy boundaries in dating",
            "authentic social bonds", "mutual trust and respect", "compassionate friendships",
            "heart-centered connections", "positive social influences"
        ],
        "handcrafted": [
            "I want to attract my ideal soulmate and life partner.",
            "My goal is to build deep meaningful friendships.",
            "I intend to heal my family relationships with forgiveness.",
            "I am creating healthy boundaries in romantic relationships.",
            "I choose to surround myself with supportive authentic friends.",
            "I am meeting like-minded people in my local community.",
            "I want to cultivate unconditional love and mutual respect.",
            "I am nurturing my marriage with daily quality time.",
            "I attract positive uplifting people into my social circle.",
            "My focus is on deepening heart-centered social connections.",
            "I communicate with honesty warmth and empathy.",
            "I build lasting friendships grounded in trust."
        ]
    },
    "Health, Wellness & Lifestyle": {
        "verbs": ["improve", "prioritize", "restore", "build", "maintain", "heal", "strengthen", "vitalize", "rejuvenate", "optimize", "boost"],
        "nouns": [
            "my physical fitness", "deep restful sleep", "a clean organic diet", "my mental peace",
            "a consistent workout routine", "body strength and mobility", "gut health and energy",
            "my immune system", "daily morning stretching", "my posture and stamina",
            "cardiovascular health", "holistic wellness habits"
        ],
        "handcrafted": [
            "I want to improve my physical fitness and stamina.",
            "My goal is to get 8 hours of deep restful sleep every night.",
            "I intend to eat a clean nutritious plant-focused diet.",
            "I am building a consistent morning workout routine.",
            "I choose to heal my gut health and boost my energy levels.",
            "I am prioritizing my mental health and emotional well-being.",
            "I want to lose weight and feel great in my body.",
            "I am strengthening my core muscles and overall posture.",
            "I practice daily hydration and mindful recovery.",
            "My goal is to run a marathon and build peak athletic endurance.",
            "I nourish my body with wholesome organic foods.",
            "I honor my physical body with rest movement and care."
        ]
    },
    "Environment & Material Upgrades": {
        "verbs": ["buy", "upgrade", "decorate", "renovate", "declutter", "create", "find", "organize", "transform", "refresh", "design"],
        "nouns": [
            "my dream home", "a modern minimalist apartment", "my ideal electric vehicle",
            "a quiet home office setup", "my personal wardrobe", "a cozy living space",
            "a serene backyard garden", "my interior design aesthetic", "a clutter-free sanctuary",
            "functional living areas", "inspiring work environment"
        ],
        "handcrafted": [
            "I want to buy my dream home in a peaceful neighborhood.",
            "My goal is to declutter my apartment and create a minimalist space.",
            "I intend to upgrade my home office with an ergonomic desk.",
            "I am renovating my kitchen with high quality finishes.",
            "I choose to purchase my ideal reliable electric vehicle.",
            "I am transforming my living room into a cozy sanctuary.",
            "I want to build a beautiful backyard garden with solar lights.",
            "I am refreshing my wardrobe with sustainable quality clothes.",
            "I manifest a spacious sunlit apartment near the coast.",
            "My target is to organize every room in my house.",
            "I design an inspiring environment that nurtures peace.",
            "I curate my living space with beauty and harmony."
        ]
    },
    "Mindset & Emotional Well-Being": {
        "verbs": ["cultivate", "release", "build", "embrace", "strengthen", "heal", "master", "transform", "overcome", "anchor", "ground"],
        "nouns": [
            "unshakable self-confidence", "inner peace and calm", "past emotional trauma",
            "a resilient growth mindset", "self-compassion and grace", "mindfulness habits",
            "my self-esteem", "overthinking and anxiety", "emotional balance",
            "positive self-talk", "inner stability"
        ],
        "handcrafted": [
            "I want to cultivate unshakable self-confidence and self-worth.",
            "My goal is to release overthinking and calm my anxiety.",
            "I intend to heal past emotional trauma with therapy and grace.",
            "I am building a resilient growth mindset in times of stress.",
            "I choose self-compassion over harsh self-criticism.",
            "I am practicing daily mindfulness and emotional grounding.",
            "I want to overcome fear of failure and step into my power.",
            "I am developing emotional stability and inner tranquility.",
            "I release negative self-talk and embrace empowering thoughts.",
            "My focus is on forgiving myself and moving forward with peace.",
            "I respond to life challenges with calm inner strength.",
            "I anchor myself in present moment peace and gratitude."
        ]
    },
    "Personal Habits & Time Management": {
        "verbs": ["master", "eliminate", "establish", "optimize", "stop", "maintain", "build", "refine", "streamline", "structure", "prioritize"],
        "nouns": [
            "a productive morning routine", "procrastination and screen time", "work-life balance",
            "time management skills", "a daily journaling habit", "focused deep work sessions",
            "my daily task schedule", "digital distractions", "consistent habits",
            "time blocking methods", "daily productivity momentum"
        ],
        "handcrafted": [
            "I want to master a productive 5am morning routine.",
            "My goal is to eliminate procrastination and phone distractions.",
            "I intend to establish a healthy work-life balance.",
            "I am optimizing my daily calendar with time blocking.",
            "I choose to maintain a daily gratitude journaling habit.",
            "I am completing 2 hours of uninterrupted deep work every day.",
            "I want to stop scrolling social media before bedtime.",
            "I am building discipline around my daily priorities.",
            "I stick to my weekly schedule with consistency and ease.",
            "My focus is on managing my energy and time effectively.",
            "I eliminate time-wasting habits and focus on what matters.",
            "I build daily systems that compound into long term success."
        ]
    },
    "Creativity, Hobbies & Learning": {
        "verbs": ["learn", "master", "create", "publish", "practice", "explore", "produce", "compose", "design", "illustrate", "write"],
        "nouns": [
            "a new foreign language", "digital video editing skills", "my personal blog and podcast",
            "oil painting and drawing", "acoustic guitar playing", "full stack web development",
            "creative writing and storytelling", "photography techniques", "my artistic craft",
            "interactive web applications", "original musical compositions"
        ],
        "handcrafted": [
            "I want to learn fluent Spanish and travel abroad.",
            "My goal is to publish my creative fantasy novel.",
            "I intend to master digital video editing in Premiere.",
            "I am launching my weekly creative podcast and newsletter.",
            "I choose to practice playing piano for 30 minutes daily.",
            "I am taking an online course in full stack web development.",
            "I want to produce high quality digital art and illustrations.",
            "I am exploring landscape photography and color grading.",
            "I build my creative skills through consistent daily practice.",
            "My goal is to design and code an interactive web app.",
            "I express my unique artistic voice through creative projects.",
            "I commit to daily skill-building in my favorite hobbies."
        ]
    },
    "Social Impact & Contribution": {
        "verbs": ["donate to", "volunteer at", "mentor", "launch", "support", "give back to", "advocate for", "fund", "sponsor", "organize"],
        "nouns": [
            "local animal shelters", "community youth programs", "environmental conservation efforts",
            "young aspiring entrepreneurs", "charitable organizations", "non-profit initiatives",
            "underprivileged students", "ocean clean up projects", "social causes",
            "educational grants", "community sustainability initiatives"
        ],
        "handcrafted": [
            "I want to donate to local animal rescue shelters.",
            "My goal is to volunteer 5 hours a week at a food bank.",
            "I intend to mentor young aspiring software engineers.",
            "I am launching a community clean up initiative.",
            "I choose to support environmental conservation organizations.",
            "I am funding educational grants for underprivileged students.",
            "I want to use my skills to contribute to open source non-profits.",
            "I am giving back to my local community through charity work.",
            "I advocate for positive social change and sustainability.",
            "My target is to sponsor tree planting and carbon reduction.",
            "I dedicate time and resource to uplifting vulnerable groups.",
            "I make a positive lasting impact on my local community."
        ]
    },
    "Education & Academic Milestones": {
        "verbs": ["pass", "graduate with", "win", "complete", "study for", "excel in", "submit", "publish", "earn", "defend"],
        "nouns": [
            "my university final exams", "top honors in my degree", "an academic scholarship",
            "my master's thesis research", "college admission applications", "a competitive certification",
            "my medical school entrance test", "academic research papers", "my diploma",
            "doctoral dissertation defense", "peer reviewed journal submissions"
        ],
        "handcrafted": [
            "I want to pass my university final exams with straight As.",
            "My goal is to graduate with top honors in computer science.",
            "I intend to win a full academic research scholarship.",
            "I am completing my master's thesis defense successfully.",
            "I choose to study consistently for my professional certification.",
            "I am accepted into my first choice graduate school program.",
            "I want to submit my scientific paper to a peer reviewed journal.",
            "I am scoring in the top percentile on my standardized test.",
            "I excel in my coursework through diligent daily study.",
            "My target is to earn my engineering diploma this semester.",
            "I achieve high test scores through structured exam study.",
            "I complete my academic degree with pride and distinction."
        ]
    },
    "Independence & Freedom": {
        "verbs": ["achieve", "build", "transition to", "live as", "create", "enjoy", "secure", "reclaim", "expand", "establish"],
        "nouns": [
            "complete location independence", "a remote digital nomad lifestyle", "a freelance consulting business",
            "autonomous working hours", "geographic and financial freedom", "passive revenue streams",
            "sovereign decision making", "self-directed living", "lifestyle freedom",
            "unrestricted personal schedule", "work from anywhere flexibility"
        ],
        "handcrafted": [
            "I want to achieve complete location independence.",
            "My goal is to live as a digital nomad traveling the world.",
            "I intend to transition to full-time remote freelance work.",
            "I am building automated passive income to fund my freedom.",
            "I choose to work on my own schedule from anywhere.",
            "I am creating a life where I control my own time.",
            "I want the freedom to live and work abroad near the ocean.",
            "I am breaking free from corporate micromanagement.",
            "I enjoy complete autonomy over my career and choices.",
            "My focus is on securing geographic and financial liberty.",
            "I build autonomous income systems that support my travel.",
            "I reclaim sovereign control over my daily schedule and location."
        ]
    },
    "Spiritual Growth & Alignment": {
        "verbs": ["align with", "deepen", "connect with", "manifest", "trust", "practice", "discover", "awaken", "ground", "surrender"],
        "nouns": [
            "my higher self and intuition", "daily spiritual meditation", "universal energy and flow",
            "my soul's divine purpose", "synchronicities and signs", "chakra balance and healing",
            "spiritual wisdom and alignment", "gratitude and presence", "inner stillness",
            "transcendent consciousness", "soulful mindfulness practices"
        ],
        "handcrafted": [
            "I want to align with my higher self and inner wisdom.",
            "My goal is to practice 20 minutes of daily silent meditation.",
            "I intend to trust the timing of the universe and divine flow.",
            "I am connecting deeply with my soul's true purpose.",
            "I choose to recognize synchronicities and signs of alignment.",
            "I am balancing my energy centers and spiritual awareness.",
            "I want to cultivate unconditional love and spiritual peace.",
            "I am grounding myself in gratitude presence and trust.",
            "I surrender worry and open my heart to spiritual guidance.",
            "My focus is on awakening my intuition and inner light.",
            "I practice deep meditation to connect with universal energy.",
            "I live in peaceful alignment with my highest divine self."
        ]
    },
    "Major Life & Family Milestones": {
        "verbs": ["celebrate", "welcome", "buy", "plan", "start", "prepare for", "build", "nurture", "expand", "cherish"],
        "nouns": [
            "a beautiful wedding ceremony", "our first newborn baby", "family property and land",
            "a happy peaceful retirement", "our dream family home", "a healthy pregnancy",
            "a milestone anniversary", "family heritage and legacy", "a stable family life",
            "multigenerational wealth", "cherished family traditions"
        ],
        "handcrafted": [
            "I want to plan our beautiful dream wedding ceremony.",
            "My goal is to welcome a healthy happy newborn baby.",
            "I intend to purchase family land and build a homestead.",
            "I am preparing for a comfortable and joyful retirement.",
            "I choose to create a warm nurturing home environment for my children.",
            "I am celebrating 10 wonderful years of loving marriage.",
            "I want to support my aging parents with care and comfort.",
            "I am expanding our family with love and stability.",
            "I build a strong lasting legacy for future generations.",
            "My focus is on creating unforgettable family memories.",
            "I nurture my children with patience love and wisdom.",
            "I create a joyful peaceful home life for my whole family."
        ]
    },
    "Legal & Administrative Resolution": {
        "verbs": ["resolve", "obtain", "secure", "settle", "finalize", "complete", "file", "acquire", "register", "process"],
        "nouns": [
            "my permanent residency green card", "a favorable legal dispute settlement", "official passport and visa approval",
            "my business trademark registration", "important administrative paperwork", "a fair contract agreement",
            "legal ownership documents", "court proceedings smoothly", "tax documentation",
            "citizenship application papers", "estate planning documents"
        ],
        "handcrafted": [
            "I want to secure my permanent residency green card approval.",
            "My goal is to resolve all pending legal paperwork smoothly.",
            "I intend to obtain my international work visa approval.",
            "I am settling our business contract dispute fairly.",
            "I choose to finalize my trademark and copyright registrations.",
            "I am filing my annual business taxes accurately and on time.",
            "I want to receive final court clearance and legal peace.",
            "I am completing all administrative property transfer documents.",
            "I receive positive legal outcomes and official documentation.",
            "My target is to finalize all immigration and visa paperwork.",
            "I navigate government filings with clarity and efficiency.",
            "I secure legal clearance and official documentation."
        ]
    },
    "Intellectual & Cognitive Development": {
        "verbs": ["enhance", "master", "sharpen", "expand", "unlock", "develop", "accelerate", "train", "boost", "elevate"],
        "nouns": [
            "my memory and recall", "deep flow state focus", "bilingual language skills",
            "speed reading techniques", "cognitive capacity and mental speed", "accelerated learning habits",
            "my brain health and neuroplasticity", "information retention skills", "complex problem solving ability",
            "analytical thinking skills", "mental sharpness and clarity"
        ],
        "handcrafted": [
            "I want to sharpen my memory and focus every day.",
            "My goal is to enter a deep flow state while studying.",
            "I intend to become bilingual and master a second language.",
            "I am practicing speed reading techniques to absorb books faster.",
            "I choose to expand my cognitive capacity and mental speed.",
            "I am accelerating my learning and information retention.",
            "I want to improve my cognitive brain health through neuroplasticity exercises.",
            "I am building complex problem solving and logical thinking skills.",
            "I master new concepts rapidly with sharp focus.",
            "My focus is on unlocking peak cognitive performance and intellect.",
            "I train my brain daily to increase cognitive speed and focus.",
            "I master complex technical concepts with rapid learning ability."
        ]
    },
    "Athletic & Physical Performance": {
        "verbs": ["shatter", "achieve", "train for", "win", "improve", "build", "master", "optimize", "break", "exceed"],
        "nouns": [
            "my powerlifting PR", "a new personal record", "a full marathon race",
            "the regional championship", "athletic agility and mobility", "body flexibility and balance",
            "sports performance and speed", "post-workout physical recovery", "elite strength and endurance",
            "sprint velocity and power", "competition readiness"
        ],
        "handcrafted": [
            "I want to shatter my powerlifting PR in the squat and deadlift.",
            "My goal is to set a new personal record in my next race.",
            "I intend to train for and finish my first full marathon.",
            "I am working hard to win the sports championship this season.",
            "I choose to improve my athletic agility speed and footwork.",
            "I am enhancing my body flexibility through daily yoga and stretching.",
            "I want to optimize my post-workout physical recovery and rest.",
            "I am building elite strength power and cardiovascular endurance.",
            "I achieve peak athletic performance in every competition.",
            "My focus is on mastering sports agility and physical excellence.",
            "I push past my physical limits and set new athletic records.",
            "I train with consistency and power to achieve athletic mastery."
        ]
    },
    "Eco-Conscious & Sustainable Living": {
        "verbs": ["build", "transition to", "install", "cultivate", "embrace", "practice", "adopt", "create", "reduce", "minimize"],
        "nouns": [
            "an off-grid sustainable cabin", "a zero-waste eco-conscious lifestyle", "solar energy power systems",
            "a thriving organic homesteading garden", "ethical purchasing choices", "sustainable living habits",
            "a deep connection with nature", "low-impact environmental practices", "permaculture home design",
            "carbon footprint reduction", "renewable resource usage"
        ],
        "handcrafted": [
            "I want to build an off-grid solar-powered cabin.",
            "My goal is to transition to a zero-waste eco-conscious lifestyle.",
            "I intend to install solar energy panels on my home.",
            "I am cultivating a thriving organic homesteading garden.",
            "I choose to make ethical and sustainable purchasing decisions.",
            "I am practicing low-impact eco-conscious living in harmony with nature.",
            "I want to start homesteading and growing my own food.",
            "I am reducing my carbon footprint with sustainable daily habits.",
            "I embrace green living permaculture and renewable energy.",
            "My target is to live in intentional harmony with the natural environment.",
            "I build a zero-waste sustainable home using permaculture.",
            "I protect the natural environment through conscious daily choices."
        ]
    },
    "Long-Term Relationship Longevity": {
        "verbs": ["nurture", "deepen", "celebrate", "build", "maintain", "strengthen", "cherish", "foster", "renew", "sustain"],
        "nouns": [
            "our golden anniversary milestone", "profound emotional intimacy", "harmonious co-parenting practices",
            "our journey of aging together", "a resilient decades-long marriage", "an unshakeable soul connection",
            "lasting partnership trust", "daily romantic connection", "a lifetime of shared memories",
            "enduring mutual respect", "relationship resilience and commitment"
        ],
        "handcrafted": [
            "I want to celebrate our milestone wedding anniversary with joy.",
            "My goal is to nurture profound emotional intimacy in our relationship.",
            "I intend to practice supportive and harmonious co-parenting.",
            "I am looking forward to aging together with my life partner.",
            "I choose to build a resilient decades-long marriage.",
            "I am deepening our unshakeable soul connection and trust.",
            "I want to foster lifelong love patience and mutual respect.",
            "I am prioritizing daily romance and connection with my spouse.",
            "We build a secure partnership that grows stronger over decades.",
            "My focus is on creating a lasting marriage rooted in deep affection.",
            "I cherish aging alongside my soulmate in deep love and intimacy.",
            "We sustain a vibrant loving marriage through lifelong commitment."
        ]
    },
    "Crisis Recovery & Fresh Starts": {
        "verbs": ["rebuild", "heal from", "overcome", "relocate to", "navigate", "reclaim", "achieve", "start", "bounce back from", "overhaul"],
        "nouns": [
            "a fresh start after my layoff", "financial recovery after bankruptcy", "emotional healing post-breakup",
            "a new life as I relocate", "profound healing after loss", "my ultimate career comeback",
            "a secure life after hardship", "my identity and personal strength", "a positive new chapter",
            "resilience after unexpected setback", "a clean slate in a new city"
        ],
        "handcrafted": [
            "I want to rebuild my life stronger after a sudden layoff.",
            "My goal is to heal emotionally post-breakup and move forward.",
            "I intend to recover financially after bankruptcy and debt.",
            "I am relocating to a new city for a fresh start.",
            "I choose to heal from personal loss and rebuild my spirit.",
            "I am executing my ultimate career and life comeback.",
            "I want to reclaim my identity and confidence after adversity.",
            "I am embracing this fresh start with courage and hope.",
            "I transform past crisis into a powerful foundation for growth.",
            "My focus is on healing rebuilding and starting a bright new chapter.",
            "I rise stronger from adversity and embrace a powerful fresh start.",
            "I rebuild my life with absolute confidence after recent setbacks."
        ]
    },
    "Digital Detox & Tech Balance": {
        "verbs": ["reduce", "unplug from", "limit", "establish", "practice", "reclaim", "cure", "minimize", "curb", "manage"],
        "nouns": [
            "daily screen time and scrolling", "social media addiction", "digital distraction habits",
            "a healthy technology boundary", "offline presence and mindfulness", "a phone-free evening routine",
            "digital detox weekends", "mental peace away from devices", "mindful technology use",
            "healthy relationship with my phone", "screen-free focus time"
        ],
        "handcrafted": [
            "I want to reduce my daily screen time and phone scrolling.",
            "My goal is to take a full weekend social media digital detox.",
            "I intend to establish a strict phone-free evening routine.",
            "I am unplugging from digital distractions to focus on real life.",
            "I choose to practice digital minimalism and limit notifications.",
            "I am reclaiming my attention span from constant screen scrolling.",
            "I want to stop checking my phone immediately after waking up.",
            "I am building healthy boundaries with social media usage.",
            "I replace screen time with reading outdoor walks and presence.",
            "My focus is on cultivating mental calm through a digital detox."
        ]
    },
    "Mindful Nutrition & Culinary Arts": {
        "verbs": ["master", "cook", "prepare", "adopt", "enjoy", "nourish", "explore", "learn", "cultivate", "eat"],
        "nouns": [
            "mindful eating and conscious meals", "organic home-cooked recipes", "culinary arts and cooking skills",
            "a balanced whole-food diet", "gourmet plant-based cooking", "meal planning and prep habits",
            "healthy relationship with food", "nutritious home dining", "culinary creativity in the kitchen",
            "wholesome nutritional wellness", "homemade chef-quality meals"
        ],
        "handcrafted": [
            "I want to master gourmet plant-based cooking skills.",
            "My goal is to cook healthy organic home-cooked meals daily.",
            "I intend to adopt mindful eating habits and slow down at meals.",
            "I am exploring culinary arts and learning new healthy recipes.",
            "I choose to nourish my body with wholesome whole foods.",
            "I am establishing a weekly meal prep routine for nutrition.",
            "I want to eliminate processed foods and eat clean organic meals.",
            "I am cultivating creativity and joy in my home kitchen.",
            "I prepare delicious chef-quality meals with love and care.",
            "My focus is on mindful nutrition and culinary wellness."
        ]
    },
    "Travel, Exploration & Adventure": {
        "verbs": ["explore", "travel to", "experience", "plan", "embark on", "discover", "visit", "trek", "immerse in", "journey across"],
        "nouns": [
            "exotic international destinations", "an epic backpacking adventure", "new cultures and languages",
            "world heritage national parks", "a solo travel journey abroad", "scenic mountain trekking routes",
            "historic global landmarks", "unforgettable travel experiences", "coastal island adventures",
            "cross-cultural connections", "wandering and world exploration"
        ],
        "handcrafted": [
            "I want to travel to exotic international destinations.",
            "My goal is to embark on an epic solo backpacking journey.",
            "I intend to explore new cultures and learn local languages.",
            "I am visiting famous world heritage sites and national parks.",
            "I choose to plan an adventurous mountain trekking expedition.",
            "I am immersing myself in authentic global travel experiences.",
            "I want to discover hidden coastal islands and vibrant cities.",
            "I am creating lifelong memories through world exploration.",
            "I embrace the freedom of travel adventure and discovery.",
            "My focus is on expanding my horizons through international travel."
        ]
    },
    "Parenting & Child Development": {
        "verbs": ["raise", "nurture", "guide", "support", "foster", "practice", "encourage", "cultivate", "teach", "instill"],
        "nouns": [
            "happy resilient children", "gentle parenting techniques", "early child development milestones",
            "positive emotional growth in my kids", "a safe loving home for my children", "play-based learning activities",
            "strong parent-child emotional bonds", "healthy communication with my kids", "valuable life lessons in parenting",
            "confidence and kindness in my children", "family unity and growth"
        ],
        "handcrafted": [
            "I want to raise happy resilient and confident children.",
            "My goal is to practice gentle parenting and active listening.",
            "I intend to nurture my child's emotional growth and creativity.",
            "I am creating a safe loving and supportive home environment.",
            "I choose to guide my kids with patience warmth and boundaries.",
            "I am engaging in play-based learning and quality time.",
            "I want to instill strong moral values and kindness in my children.",
            "I am strengthening my emotional bond with my kids daily.",
            "I support my child through every developmental milestone with love.",
            "My focus is on being a patient loving and present parent."
        ]
    },
    "Sleep, Rest & Recovery Science": {
        "verbs": ["optimize", "improve", "achieve", "establish", "prioritize", "master", "enjoy", "restore", "enhance", "maintain"],
        "nouns": [
            "deep REM and slow-wave sleep", "a relaxing evening wind-down routine", "circadian rhythm alignment",
            "restful sleep hygiene habits", "complete physical and nervous system recovery", "sleep quality and duration",
            "deep relaxation before bed", "morning energy and alertness", "a peaceful bedroom sleep environment",
            "nervous system regulation", "optimal sleep science metrics"
        ],
        "handcrafted": [
            "I want to achieve 8 hours of deep restful REM sleep.",
            "My goal is to optimize my circadian rhythm and sleep schedule.",
            "I intend to establish a soothing evening wind-down routine.",
            "I am practicing sleep hygiene for rapid physical recovery.",
            "I choose to prioritize nervous system relaxation before bed.",
            "I am waking up feeling completely energized and refreshed.",
            "I want to create a dark quiet bedroom optimized for sleep.",
            "I am tracking my sleep metrics and improving deep sleep quality.",
            "I honor my body's need for deep rest and nightly restoration.",
            "My focus is on mastering sleep science and recovery habits."
        ]
    },
    "Generational Legacy & Ancestral Healing": {
        "verbs": ["build", "establish", "manifest", "break", "heal", "secure", "create", "protect", "honors", "pass down"],
        "nouns": [
            "generational wealth and trust funds", "our family empire and heritage", "deep ancestral healing",
            "breaking old family curses", "a lasting legacy for my descendants", "multigenerational financial security",
            "an unbreakable family foundation", "ancestral wisdom and strength", "legacy real estate holdings",
            "centuries of family prosperity", "bloodline freedom and security"
        ],
        "handcrafted": [
            "I want to build generational wealth and a trust fund for my family.",
            "My goal is to manifest deep ancestral healing and break old family curses.",
            "I intend to establish a multi-generational family empire and trust.",
            "I am breaking generational trauma and securing financial freedom.",
            "I choose to protect our family heritage and pass down wisdom.",
            "I am building a lasting legacy that will echo for centuries.",
            "I want to secure multigenerational wealth and property for my descendants.",
            "I am healing my bloodline and establishing ultimate stability.",
            "I create an unbreakable family legacy built on love and wealth.",
            "My focus is on ancestral healing and securing my family's future."
        ]
    },
    "Shadow Work & Deep Unconscious Integration": {
        "verbs": ["integrate", "embrace", "heal", "face", "transmute", "release", "accept", "uncover", "reclaim", "harmonize"],
        "nouns": [
            "repressed shadow traits and dark aspects", "deep shame and hidden fears", "karmic loops and patterns",
            "my unconscious mind and instincts", "radical self-acceptance and wholeness", "repressed emotional trauma",
            "hidden shadow wisdom", "unconscious behavioral triggers", "shadow integration and peace",
            "the hidden parts of my psyche", "karmic healing and release"
        ],
        "handcrafted": [
            "I want to practice shadow work and integrate my dark traits.",
            "My goal is to cultivate radical self-acceptance and heal deep shame.",
            "I intend to break free from repeating karmic loops.",
            "I am exploring my unconscious mind and facing my fears.",
            "I choose to transmute deep emotional trauma into strength.",
            "I am unearthing hidden shadow aspects with compassion.",
            "I want to achieve complete psychological wholeness through shadow work.",
            "I am integrating my unconscious triggers and finding inner peace.",
            "I embrace all hidden aspects of myself without judgment.",
            "My focus is on deep shadow integration and self-forgiveness."
        ]
    },
    "Civic Leadership & Systemic Change": {
        "verbs": ["lead", "spearhead", "advocate for", "run for", "champion", "enact", "drive", "reform", "organize", "transform"],
        "nouns": [
            "elected office and civic leadership", "systemic change and policy reform", "human rights and social justice",
            "community advocacy and reform", "ethical public service as mayor", "systemic progress in government",
            "civic duty and public policy", "grassroots political activism", "transformative social policy",
            "equity and systemic reform", "courageous civic representation"
        ],
        "handcrafted": [
            "I want to run for elected office and serve as mayor.",
            "My goal is to spearhead systemic policy reform for human rights.",
            "I intend to lead civic initiatives and drive community change.",
            "I am advocating for systemic equity and social justice.",
            "I choose to champion policy reform through public service.",
            "I am executing my civic duty with integrity and courage.",
            "I want to organize grassroots activism for systemic reform.",
            "I am building coalitions to transform local government policy.",
            "I drive systemic impact through bold compassionate leadership.",
            "My focus is on civic leadership public service and policy reform."
        ]
    },
    "Radical Detachment & Monk Mode": {
        "verbs": ["detach from", "enter", "practice", "embrace", "execute", "disappear into", "cultivate", "adopt", "maintain", "master"],
        "nouns": [
            "radical detachment from outcomes", "monk mode solitary focus", "a strict dopamine detox",
            "ascetic minimalism and anonymity", "off-grid solitary retreat", "freedom from external validation",
            "monk-like discipline and silence", "radical non-attachment to possessions", "deep focus away from society",
            "complete quiet and detachment", "unshakable inner solitude"
        ],
        "handcrafted": [
            "I want to enter monk mode and disappear into deep focus.",
            "My goal is to practice radical detachment from external outcomes.",
            "I intend to execute a full dopamine detox and minimize noise.",
            "I am adopting ascetic minimalism and living simply.",
            "I choose to detach from seeking approval or validation.",
            "I am cultivating monk-like discipline and absolute focus.",
            "I want to retreat off-grid for deep solitary reflection.",
            "I am mastering non-attachment and emotional independence.",
            "I embrace silence solitude and radical detachment.",
            "My focus is on monk mode deep focus and inner stillness."
        ]
    },
    "Interpersonal Severing & Deep Closure": {
        "verbs": ["sever", "cut", "achieve", "release", "forgive", "walk away from", "end", "close", "disconnect from", "finalize"],
        "nouns": [
            "energetic cords with toxic partners", "permanent closure from abusive exes", "unhealthy ties with toxic parents",
            "peaceful divorce agreements", "unforgivable past grievances", "unhealthy relationship dynamic",
            "toxic emotional attachments", "past relationship trauma", "permanent boundary severing",
            "complete closure and detachment", "relationship cord-cutting"
        ],
        "handcrafted": [
            "I want to cut energetic cords with an abusive ex.",
            "My goal is to achieve peaceful divorce and permanent closure.",
            "I intend to sever toxic ties with abusive family members.",
            "I am forgiving the unforgivable and releasing past pain.",
            "I choose to walk away permanently from toxic relationships.",
            "I am establishing firm boundaries and total closure.",
            "I want to heal relational trauma and cut toxic attachments.",
            "I am reclaiming my peace by ending unhealthy relationships.",
            "I disconnect from toxic energy and step into freedom.",
            "My focus is on deep closure forgiveness and personal peace."
        ]
    },
    "Advanced Esoteric & Metaphysical Mastery": {
        "verbs": ["master", "experience", "shift", "clear", "align with", "manifest", "navigate", "unlock", "transcend", "awaken"],
        "nouns": [
            "lucid dreaming and astral projection", "quantum jumping and reality shifting", "5D consciousness and flow",
            "somatic energy clearing", "metaphysical laws of creation", "multidimensional awareness",
            "subtle energy body mastery", "quantum manifestation techniques", "transcendent spiritual states",
            "metaphysical mastery and shift", "astral consciousness exploration"
        ],
        "handcrafted": [
            "I want to master lucid dreaming and astral projection.",
            "My goal is to experience a quantum jump into a new reality.",
            "I intend to practice somatic energy clearing for 5D alignment.",
            "I am learning reality shifting and metaphysical laws.",
            "I choose to awaken multidimensional consciousness and awareness.",
            "I am mastering energy body clearing and subtle frequencies.",
            "I want to navigate metaphysical energy with power and grace.",
            "I am unlocking quantum manifestation through deep meditation.",
            "I align with 5D frequency and transcend physical limits.",
            "My focus is on metaphysical mastery and reality shifting."
        ]
    },
    "Neurodivergence & Executive Function": {
        "verbs": ["overcome", "regulate", "manage", "unmask", "navigate", "harness", "embrace", "support", "boost", "balance"],
        "nouns": [
            "executive dysfunction and focus", "ADHD hyperfocus and activation energy", "sensory overload and calm",
            "unmasking my authentic neurodivergent self", "autistic burnout and recovery", "dopamine regulation habits",
            "executive function systems", "sensory friendly environment", "my neurodivergent brain style",
            "activation energy and task initiation"
        ],
        "handcrafted": [
            "I want to manage ADHD executive dysfunction and hyperfocus.",
            "My goal is to recover from autistic burnout and sensory overload.",
            "I intend to unmask my authentic self with self-compassion.",
            "I am building activation energy to initiate daily tasks.",
            "I choose to create a sensory friendly low-demand space.",
            "I am honoring my neurodivergent brain and energy levels.",
            "I want to regulate my nervous system during sensory overload.",
            "I am establishing supportive executive function routines.",
            "I embrace my neurodivergence hyperfocus and creative mind.",
            "My focus is on unmasking burnout recovery and ADHD balance."
        ]
    },
    "Everyday Adulting & Life Admin": {
        "verbs": ["conquer", "complete", "organize", "manage", "file", "pass", "tackle", "check off", "clear", "master"],
        "nouns": [
            "annual taxes and paperwork", "my driving test exam", "my credit score improvement",
            "inbox zero and email admin", "household chores and errands", "important administrative tasks",
            "daily adulting responsibilities", "personal record filing", "financial life admin",
            "to-do list management"
        ],
        "handcrafted": [
            "I want to conquer my annual taxes and paperwork on time.",
            "My goal is to pass my driving test with confidence.",
            "I intend to improve my credit score and manage my bills.",
            "I am achieving inbox zero and clearing administrative clutter.",
            "I choose to tackle my household chores and daily errands.",
            "I am taking methodical control of my adulting responsibilities.",
            "I want to organize important legal and tax documents.",
            "I am completing my life admin tasks step by step.",
            "I handle paperwork errands and adulting with calm ease.",
            "My focus is on life admin paperwork and task organization."
        ]
    },
    "Early Parenthood & Childcare": {
        "verbs": ["navigate", "master", "practice", "manage", "nurture", "support", "embrace", "handle", "soothe", "balance"],
        "nouns": [
            "postpartum recovery and mental health", "toddler tantrums and meltdowns", "gentle parenting techniques",
            "infant sleep training routines", "breastfeeding and baby nutrition", "early parenthood chaos and joy",
            "newborn sleep schedules", "toddler developmental milestones", "maternal and paternal instincts",
            "childcare daily balance"
        ],
        "handcrafted": [
            "I want to navigate postpartum recovery with grace and rest.",
            "My goal is to practice gentle parenting during toddler tantrums.",
            "I intend to establish a consistent infant sleep training routine.",
            "I am supporting my baby through early developmental stages.",
            "I choose to trust my parental instincts in newborn care.",
            "I am managing the beautiful chaos of early parenthood.",
            "I want to enjoy breastfeeding and nurturing my newborn baby.",
            "I am remaining a calm anchor during toddler meltdowns.",
            "I balance childcare rest and gentle parenting daily.",
            "My focus is on early parenthood gentle guidance and baby care."
        ]
    },
    "Midlife Transitions & Empty Nesting": {
        "verbs": ["embrace", "navigate", "reinvent", "downsize", "welcome", "step into", "enjoy", "prioritize", "rediscover", "flourish in"],
        "nouns": [
            "this empty nest season of freedom", "menopause and physical midlife changes", "my identity in midlife",
            "the physical clutter of downsizing", "aging gracefully with vitality", "a new era of self-discovery",
            "midlife transition and renewal", "freedom to prioritize myself", "a serene empty nest home",
            "second act life goals"
        ],
        "handcrafted": [
            "I want to embrace this empty nest chapter with joy and freedom.",
            "My goal is to navigate menopause and midlife changes with grace.",
            "I intend to reinvent my identity and pursue new passions.",
            "I am downsizing my home and clearing unnecessary weight.",
            "I choose to step into aging gracefully and prioritizing myself.",
            "I am discovering new purpose in this midlife transition.",
            "I want to celebrate the peace and freedom of empty nesting.",
            "I am designing a vibrant second act filled with passion.",
            "I flourish in midlife with wisdom maturity and vitality.",
            "My focus is on midlife self-discovery empty nesting and freedom."
        ]
    },
    "Tech, Gear & Digital Lifestyle": {
        "verbs": ["build", "upgrade", "optimize", "achieve", "customize", "setup", "streamline", "enhance", "master", "configure"],
        "nouns": [
            "my dream PC build and setup", "the ultimate battlestation desk", "high rank in competitive gaming",
            "my live streaming equipment", "smart home automation systems", "tech gear and hardware upgrades",
            "high performance gaming rig", "clean cable management setup", "digital content creator gear",
            "peak streaming performance"
        ],
        "handcrafted": [
            "I want to build my dream gaming PC and desk setup.",
            "My goal is to achieve top rank in my favorite competitive game.",
            "I intend to upgrade my live streaming gear and microphone.",
            "I am configuring smart home automation for my house.",
            "I choose to optimize my PC performance and display.",
            "I am creating an inspiring battlestation and tech space.",
            "I want to master digital content creation and video streaming.",
            "I am upgrading my hardware for maximum speed and graphics.",
            "I curate the ultimate digital tech gear and setup.",
            "My focus is on PC building gaming rank and smart home tech."
        ]
    },
    "Micro-Hobbies & Crafting Mastery": {
        "verbs": ["master", "create", "craft", "bake", "sew", "knit", "woodwork", "pour", "design", "handcraft"],
        "nouns": [
            "sourdough bread baking skills", "custom woodworking and furniture", "knitting and crochet projects",
            "DIY crafting and handmade art", "sewing tailored clothing", "artisanal pottery and ceramics",
            "handcrafted leather goods", "creative micro-hobbies", "intricate handmade creations",
            "mindful craft techniques"
        ],
        "handcrafted": [
            "I want to master baking perfect sourdough bread at home.",
            "My goal is to craft custom woodworking projects in my workshop.",
            "I intend to learn knitting and sew my own clothing.",
            "I am exploring DIY micro-hobbies and handmade crafts.",
            "I choose to pour focus into mindful pottery and ceramics.",
            "I am creating beautiful handmade art with patience and care.",
            "I want to hone my sewing and tailoring techniques.",
            "I am building a cozy home studio for crafting and baking.",
            "I find deep peace in working with my hands on micro-hobbies.",
            "My focus is on sourdough baking DIY crafting and woodworking."
        ]
    },
    "Elder Care & Family Stewardship": {
        "verbs": ["navigate", "manage", "support", "balance", "honor", "care for", "arrange", "handle", "coordinate", "provide"],
        "nouns": [
            "aging parents and elder care", "caregiver burnout and self-care", "nursing home arrangements",
            "estate planning and family affairs", "dementia care and role reversal", "senior medical care and support",
            "assisted living facilities", "family estate legal documents", "elderly health management"
        ],
        "handcrafted": [
            "I want to navigate elder care for my aging parents with patience.",
            "My goal is to manage caregiver burnout while supporting my mother.",
            "I intend to arrange nursing home care and estate affairs for my family.",
            "I am navigating role reversal and dementia care for my aging father.",
            "I choose to honor my family stewardship while protecting my mental health.",
            "I am arranging assisted living facilities and senior medical care.",
            "I manage caregiver responsibilities and family estate planning.",
            "My focus is on supporting aging parents and avoiding caregiver burnout."
        ]
    },
    "Workplace Dynamics & Colleague Boundaries": {
        "verbs": ["navigate", "manage", "establish", "handle", "maintain", "protect", "address", "deal with", "set", "enforce"],
        "nouns": [
            "a toxic boss and micromanagement", "office politics and drama", "HR reporting and compliance",
            "healthy boundaries to log off on time", "passive-aggressive coworkers", "workplace boundaries and peace",
            "corporate stress and company culture", "unprofessional office behavior", "workplace communication limits"
        ],
        "handcrafted": [
            "I want to set healthy workplace boundaries with a toxic boss.",
            "My goal is to log off on time and ignore office politics.",
            "I intend to handle passive-aggressive coworkers with professional distance.",
            "I am reporting workplace harassment to HR with confidence.",
            "I choose to protect my mental energy from corporate stress and office drama.",
            "I manage difficult coworker interactions with clear boundaries.",
            "I maintain professional peace and log off at the end of the workday.",
            "My focus is on navigating toxic workplace dynamics and office boundaries."
        ]
    },
    "Chronic Illness & Invisible Disability": {
        "verbs": ["manage", "navigate", "advocate for", "pace", "honor", "heal from", "support", "handle", "accommodate", "treat"],
        "nouns": [
            "autoimmune condition and flare-ups", "chronic pain management", "long covid recovery and fatigue",
            "daily spoon theory energy pacing", "medical advocacy and patient rights", "invisible disability accommodations",
            "chronic fatigue syndrome support", "somatic symptoms and pain relief", "daily energy limits"
        ],
        "handcrafted": [
            "I want to manage my autoimmune condition and chronic pain with grace.",
            "My goal is to practice daily spoon theory energy pacing during flare-ups.",
            "I intend to advocate for my medical rights and disability accommodations.",
            "I am navigating long covid recovery and chronic fatigue at my own pace.",
            "I choose to honor my body's daily energy limits without guilt.",
            "I handle chronic illness flare-ups with rest and medical self-advocacy.",
            "I pace my daily physical spoons and manage chronic pain.",
            "My focus is on invisible disability accommodations and chronic wellness."
        ]
    },
    "Substance Recovery & Vice Cessation": {
        "verbs": ["achieve", "maintain", "quit", "overcome", "commit to", "embrace", "sustain", "master", "stay", "build"],
        "nouns": [
            "total alcohol sobriety and recovery", "quitting smoking and nicotine", "relapse prevention strategies",
            "AA meetings and 12 step recovery", "gambling addiction cessation", "a clean and sober lifestyle",
            "sobriety milestones and freedom", "vice cessation and self-control", "daily addiction recovery"
        ],
        "handcrafted": [
            "I want to achieve total sobriety and quit drinking alcohol.",
            "My goal is to quit smoking and overcome nicotine addiction.",
            "I intend to maintain relapse prevention and attend recovery meetings.",
            "I am building a clean, sober life free from gambling and vices.",
            "I choose to embrace daily sobriety and reclaim control over my future.",
            "I stay sober every single day and celebrate my recovery milestone.",
            "I overcome addictive cravings and maintain clean vice cessation.",
            "My focus is on alcohol sobriety addiction recovery and clean living."
        ]
    },
    "Event Planning & Milestone Hosting": {
        "verbs": ["plan", "host", "organize", "coordinate", "manage", "execute", "navigate", "enjoy", "arrange", "budget"],
        "nouns": [
            "wedding planning and vendor coordination", "hosting holiday family dinners", "family reunion logistics and catering",
            "RSVP tracking and event budgets", "social battery balance during events", "memorable milestone celebrations",
            "birthday party event planning", "caterer contracts and venues", "festive holiday hosting"
        ],
        "handcrafted": [
            "I want to plan my dream wedding without overwhelming stress.",
            "My goal is to host holiday family dinners with ease and joy.",
            "I intend to manage caterers and RSVPs for our family reunion.",
            "I am balancing my social battery while hosting milestone events.",
            "I choose to organize memorable celebrations while enjoying the moment.",
            "I coordinate vendors venues and event planning budgets smoothly.",
            "I host festive holiday gatherings with calm organization.",
            "My focus is on wedding planning milestone hosting and event management."
        ]
    },
    "Homeownership & Property Maintenance": {
        "verbs": ["manage", "complete", "handle", "repair", "execute", "maintain", "tackle", "finance", "renovate", "upgrade"],
        "nouns": [
            "home renovation and contractor work", "roof leaks and emergency repairs", "yard work and landscaping maintenance",
            "DIY home improvement projects", "property tax and home insurance", "routine house maintenance",
            "plumbing and electrical repairs", "property value and home upkeep", "contractor schedules"
        ],
        "handcrafted": [
            "I want to manage home renovation projects with reliable contractors.",
            "My goal is to fix roof leaks and handle emergency home repairs.",
            "I intend to tackle DIY home improvement and yard work maintenance.",
            "I am organizing property tax payments and home insurance policies.",
            "I choose to maintain a safe, functioning, and beautiful home.",
            "I repair plumbing issues and upgrade home property features.",
            "I complete routine house maintenance and landscaping tasks.",
            "My focus is on homeownership repairs property tax and DIY renovations."
        ]
    },
    "Adult Friendship & Platonic Intimacy": {
        "verbs": ["build", "nurture", "make", "cultivate", "deepen", "connect with", "embrace", "find", "honor", "maintain"],
        "nouns": [
            "making new adult friends", "deep platonic intimacy and trust", "my inner circle of best friends",
            "long-distance friendship connections", "outgrowing old friendships gracefully", "supportive chosen family",
            "meaningful adult friend groups", "authentic platonic relationships", "regular friend catch-ups"
        ],
        "handcrafted": [
            "I want to make new adult friends and build platonic intimacy.",
            "My goal is to cultivate deep meaningful friendships in adulthood.",
            "I intend to nurture my inner circle of supportive best friends.",
            "I am maintaining long-distance friendship connections with care.",
            "I choose to accept outgrowing old friendships without bitterness.",
            "I connect with authentic adult friends who value my presence.",
            "I build a loving chosen family and supportive friend circle.",
            "My focus is on making adult friends platonic intimacy and connection."
        ]
    },
    "Intuitive Eating & Diet Culture Recovery": {
        "verbs": ["practice", "adopt", "rebuild", "embrace", "heal", "unlearn", "release", "honor", "enjoy", "master"],
        "nouns": [
            "intuitive eating and body wisdom", "food guilt and binge eating recovery", "diet culture unlearning and freedom",
            "food neutrality and peaceful meals", "stopping calorie counting and restriction", "body trust and intuitive hunger",
            "peaceful relationship with food", "mindful eating without judgment", "liberation from diet culture"
        ],
        "handcrafted": [
            "I want to practice intuitive eating and release food guilt.",
            "My goal is to heal from binge eating and diet culture restriction.",
            "I intend to embrace food neutrality and stop calorie counting.",
            "I am rebuilding absolute trust in my body's hunger signals.",
            "I choose to unlearn diet culture rules and enjoy peaceful meals.",
            "I honor my appetite and eat intuitively without shame.",
            "I liberate my mind from body judgment and restrictive diets.",
            "My focus is on intuitive eating diet culture recovery and food peace."
        ]
    },
    "Hard Conversations & Conflict Navigation": {
        "verbs": ["navigate", "have", "conduct", "master", "speak", "address", "resolve", "handle", "approach", "communicate"],
        "nouns": [
            "hard conversations with direct honesty", "conflict resolution and clarity", "direct communication without fear",
            "passive-aggressive behavior and boundaries", "speaking up for my boundaries", "handling constructive criticism",
            "clear honest dialog in relationships", "navigating tough discussions", "diplomatic conflict management"
        ],
        "handcrafted": [
            "I want to have hard conversations with calm, direct honesty.",
            "My goal is to master conflict resolution without fear of confrontation.",
            "I intend to speak up for my needs and address passive-aggressive behavior.",
            "I am handling constructive criticism and tough discussions gracefully.",
            "I choose to communicate clearly and clear the air in my relationships.",
            "I navigate difficult interpersonal conflict with poise and strength.",
            "I voice my boundaries directly and resolve misunderstandings.",
            "My focus is on hard conversations conflict resolution and clear speech."
        ]
    },
    "Guilt-Free Rest & Hustle Culture Detox": {
        "verbs": ["embrace", "practice", "unlearn", "enjoy", "prioritize", "reclaim", "allow", "savor", "honour", "master"],
        "nouns": [
            "productivity guilt and constant busywork", "hustle culture detox and rest", "the joy of doing nothing",
            "true leisure and restful weekends", "resting without feeling guilty", "uninterrupted nap time and relaxation",
            "wasting time happily without remorse", "slowing down my daily pace", "restful work-life balance"
        ],
        "handcrafted": [
            "I want to embrace guilt-free rest and unlearn hustle culture.",
            "My goal is to release productivity guilt and enjoy doing nothing.",
            "I intend to prioritize true leisure and restful weekends.",
            "I am reclaiming my right to rest without feeling lazy.",
            "I choose to slow down my pace and savor quiet relaxation.",
            "I detox from hustle culture and honor my biological need for rest.",
            "I waste time happily knowing my worth is independent of output.",
            "My focus is on guilt-free rest hustle culture detox and true leisure."
        ]
    },
    "Career Re-entry & Late-in-Life Pivots": {
        "verbs": ["navigate", "execute", "embrace", "pivot to", "start", "re-enter", "transition into", "pursue", "master", "launch"],
        "nouns": [
            "career re-entry after a long gap", "starting over in a new industry", "going back to school at 40 and beyond",
            "a late-in-life career pivot", "re-entering the professional workforce", "reinventing my professional identity",
            "transferable skills and fresh resume", "second career opportunities", "bold professional new beginnings"
        ],
        "handcrafted": [
            "I want to navigate career re-entry after a stay-at-home gap.",
            "My goal is to execute a late-in-life career pivot with total confidence.",
            "I intend to go back to university at 40 and learn new skills.",
            "I am re-entering the professional workforce with pride and experience.",
            "I choose to reinvent my career and embrace a bold new industry.",
            "I leverage my past life experience in this exciting career pivot.",
            "I step into my new professional identity without imposter syndrome.",
            "My focus is on career re-entry late-in-life pivots and retraining."
        ]
    },
    "Nervous System Regulation & Somatic Healing": {
        "verbs": ["regulate", "soothe", "practice", "master", "unclench", "activate", "calm", "release", "heal", "ground"],
        "nouns": [
            "fight-or-flight nervous system response", "somatic healing and tension release", "vagus nerve stimulation techniques",
            "soothing panic attacks and anxiety", "unclenching my jaw and shoulders", "deep diaphragmatic breathwork",
            "somatic body grounding exercises", "calming my biological stress response", "nervous system safety and peace"
        ],
        "handcrafted": [
            "I want to regulate my nervous system and soothe fight-or-flight.",
            "My goal is to practice somatic healing and vagus nerve stimulation.",
            "I intend to unclench my jaw and shoulders through deep breathwork.",
            "I am calming panic attacks and grounding my body in safety.",
            "I choose to release stored physical tension and somatic trauma.",
            "I activate my parasympathetic nervous system for deep bodily peace.",
            "I soothe my nervous system and bring my biology back to balance.",
            "My focus is on nervous system regulation somatic release and breathwork."
        ]
    },
    "Modern Dating & Romantic Vulnerability": {
        "verbs": ["navigate", "embrace", "date", "open", "trust", "build", "approach", "handle", "experience", "master"],
        "nouns": [
            "modern dating apps and swiping", "ghosting and rejection resilience", "first date nerves and authenticity",
            "romantic vulnerability and trust", "healing rejection and dating confidence", "swiping with healthy boundaries",
            "trusting someone new in dating", "dating app burnout and detox", "vulnerable romantic connections"
        ],
        "handcrafted": [
            "I want to navigate modern dating apps with healthy boundaries.",
            "My goal is to embrace romantic vulnerability without fear of rejection.",
            "I intend to handle ghosting gracefully and protect my self-worth.",
            "I am dating with confidence authenticity and clear intentions.",
            "I choose to open my heart to new romantic connections.",
            "I trust someone new while honoring my personal boundaries.",
            "I release first date nerves and show up as my authentic self.",
            "My focus is on modern dating romantic vulnerability and healthy boundaries."
        ]
    },
    "Academic Pressure & High-Stakes Testing": {
        "verbs": ["pass", "master", "conquer", "excel in", "prepare for", "ace", "complete", "study for", "overcome", "tackle"],
        "nouns": [
            "my medical board exams and tests", "my master's thesis defense presentation", "entrance exams like MCAT and GRE",
            "test anxiety and exam pressure", "intensive study schedules and focus", "academic excellence and high scores",
            "final university exams", "high-stakes test preparation", "academic performance under pressure"
        ],
        "handcrafted": [
            "I want to pass my board exams with confidence and high scores.",
            "My goal is to master my thesis defense presentation effortlessly.",
            "I intend to overcome test anxiety and conquer MCAT preparation.",
            "I am building intensive study schedules for academic excellence.",
            "I choose to trust my preparation and excel under test pressure.",
            "I study with sharp focus and complete my final university exams.",
            "I ace high-stakes testing with calm confidence and deep knowledge.",
            "My focus is on academic pressure exam preparation and thesis mastery."
        ]
    },
    "Sleep Optimization & Circadian Health": {
        "verbs": ["optimize", "improve", "master", "establish", "align", "calm", "prioritize", "restore", "achieve", "soothe"],
        "nouns": [
            "insomnia and night racing thoughts", "a consistent sleep schedule", "waking up early with vibrant energy",
            "deep REM sleep and restoration", "circadian rhythm alignment", "a relaxing evening night routine",
            "restful sleep hygiene habits", "calm bedtime mind and body", "sleep optimization and health"
        ],
        "handcrafted": [
            "I want to optimize my sleep schedule and overcome insomnia.",
            "My goal is to wake up early feeling refreshed and energized.",
            "I intend to align my circadian rhythm with a relaxing night routine.",
            "I am calming racing thoughts before bedtime for deep REM sleep.",
            "I choose to prioritize restful sleep hygiene and night recovery.",
            "I achieve deep restorative sleep every single night.",
            "I soothe my mind at night and enjoy uninterrupted sleep.",
            "My focus is on sleep optimization circadian health and evening routines."
        ]
    },
    "Driving Confidence & Commute Anxiety": {
        "verbs": ["build", "master", "overcome", "navigate", "enjoy", "handle", "conquer", "drive with", "pass", "experience"],
        "nouns": [
            "driving anxiety and panic on highways", "highway driving confidence and speed", "driver's license test and parallel parking",
            "scenic road trips and long drives", "traffic congestion and commute stress", "feeling safe behind the wheel",
            "smooth parallel parking maneuvers", "confident highway merging", "calm daily driving commute"
        ],
        "handcrafted": [
            "I want to build driving confidence and overcome highway anxiety.",
            "My goal is to pass my driver's test and master parallel parking.",
            "I intend to enjoy long scenic road trips behind the wheel.",
            "I am navigating heavy traffic and commutes with total calm.",
            "I choose to feel safe focused and relaxed while driving.",
            "I conquer highway driving panic and merge with smooth control.",
            "I drive independently to any destination with peace of mind.",
            "My focus is on driving confidence commute anxiety and highway safety."
        ]
    },
    "Language Acquisition & Accent Confidence": {
        "verbs": ["learn", "master", "speak", "build", "expand", "overcome", "acquire", "practice", "achieve", "embrace"],
        "nouns": [
            "learning a new foreign language", "speaking confidently with native speakers", "accent insecurity and pronunciation",
            "bilingual fluency and vocabulary", "language acquisition and grammar", "conversational confidence in new tongues",
            "expanding foreign language vocabulary", "smooth native pronunciation", "bilingual communication skills"
        ],
        "handcrafted": [
            "I want to learn a new foreign language and speak fluently.",
            "My goal is to speak to native speakers without accent insecurity.",
            "I intend to expand my foreign language vocabulary every day.",
            "I am overcoming fear of mistakes while acquiring new languages.",
            "I choose to embrace my unique accent and communicate clearly.",
            "I achieve bilingual fluency and express myself with confidence.",
            "I master grammar and pronunciation in my target language.",
            "My focus is on language acquisition accent confidence and fluency."
        ]
    },
    "Physical Rehabilitation & Physiotherapy": {
        "verbs": ["recover from", "heal", "restore", "practice", "relieve", "manage", "rebuild", "strengthen", "complete", "navigate"],
        "nouns": [
            "physical therapy exercises and routines", "recovering from surgery or injury", "daily mobility stretches and exercises",
            "knee pain and back pain relief", "restoring full body joint mobility", "re-injury anxiety and safe movement",
            "physiotherapy strength routines", "pain-free physical movement", "surgical recovery milestones"
        ],
        "handcrafted": [
            "I want to recover from knee injury through physical therapy.",
            "My goal is to practice daily mobility stretches for back pain relief.",
            "I intend to restore full pain-free joint mobility after surgery.",
            "I am conquering re-injury anxiety and moving with confidence.",
            "I choose to honor my body's slow physical rehabilitation process.",
            "I rebuild muscle strength and flexibility with daily physiotherapy.",
            "I heal from sports injury and enjoy active pain-free living.",
            "My focus is on physical rehabilitation physiotherapy and joint mobility."
        ]
    },
    "Solo Living & Domestic Independence": {
        "verbs": ["embrace", "enjoy", "master", "build", "create", "cook", "maintain", "cultivate", "experience", "honor"],
        "nouns": [
            "living alone in my empty apartment", "cooking healthy meals for one", "solo living domestic independence",
            "peaceful solitary home sanctuary", "enjoying quiet alone time", "taking complete care of myself",
            "domestic autonomy and solo living", "independent living confidence", "my serene single space"
        ],
        "handcrafted": [
            "I want to embrace solo living and domestic independence.",
            "My goal is to enjoy cooking healthy meals for one in my home.",
            "I intend to cultivate a peaceful solitary sanctuary in my empty apartment.",
            "I am building confidence in taking total care of myself alone.",
            "I choose to celebrate domestic independence and alone time.",
            "I thrive in my solo apartment with complete peace and autonomy.",
            "I create a beautiful home life where my own company is enough.",
            "My focus is on solo living domestic independence and self-reliance."
        ]
    },
    "Time Management & Punctuality Discipline": {
        "verbs": ["master", "overcome", "leave", "arrive", "manage", "build", "practice", "eliminate", "schedule", "control"],
        "nouns": [
            "chronic lateness and time blindness", "leaving on time for appointments", "punctuality discipline and timing",
            "managing the clock with calm focus", "rushing and schedule anxiety", "daily time management skills",
            "arriving early with peace of mind", "structured calendar planning", "timely habits and discipline"
        ],
        "handcrafted": [
            "I want to master time management and overcome chronic lateness.",
            "My goal is to leave on time and arrive early for appointments.",
            "I intend to eliminate time blindness and manage the clock with calm.",
            "I am building punctuality discipline into my daily schedule.",
            "I choose to replace chaotic rushing with grounded timely habits.",
            "I manage my calendar with total control and arrive punctually.",
            "I respect my energy and time boundaries by leaving on time.",
            "My focus is on time management punctuality discipline and calendar order."
        ]
    },
    "Financial Literacy & Basic Investing": {
        "verbs": ["build", "master", "invest in", "understand", "learn", "grow", "check", "secure", "manage", "accumulate"],
        "nouns": [
            "index funds and 401k retirement plans", "basic financial literacy and compounding", "compound interest and wealth accumulation",
            "checking my bank account without anxiety", "investing basics and stock portfolios", "building an unshakeable financial foundation",
            "financial empowerment and literacy", "smart money management rules", "automating savings and investments"
        ],
        "handcrafted": [
            "I want to master basic financial literacy and invest in index funds.",
            "My goal is to check my bank account without fear or anxiety.",
            "I intend to understand 401k plans and compound interest growth.",
            "I am building an unshakeable financial foundation for my future.",
            "I choose to learn smart money management and automated investing.",
            "I grow my wealth steadily through index funds and compound interest.",
            "I replace financial ignorance with educated money empowerment.",
            "My focus is on financial literacy index funds 401k and basic investing."
        ]
    },
    "Digital Declutter & Cybersecurity Hygiene": {
        "verbs": ["achieve", "organize", "master", "clean", "secure", "unsubscribe from", "declutter", "protect", "manage", "maintain"],
        "nouns": [
            "inbox zero and email organization", "digital declutter and photo storage", "strong password hygiene and security",
            "unsubscribing from marketing emails", "cybersecurity hygiene and identity safety", "clearing digital desktop static",
            "secure password manager setup", "organizing digital files and folders", "privacy protection online"
        ],
        "handcrafted": [
            "I want to achieve inbox zero and digital declutter.",
            "My goal is to organize digital photos and clear desktop static.",
            "I intend to use strong password hygiene and protect my identity.",
            "I am unsubscribing from unnecessary marketing emails and newsletters.",
            "I choose to practice daily cybersecurity hygiene and data safety.",
            "I clean my digital files and maintain organized cloud storage.",
            "I secure my accounts with two-factor authentication and passwords.",
            "My focus is on digital declutter inbox zero and cybersecurity hygiene."
        ]
    },
    "Personal Safety & Situational Awareness": {
        "verbs": ["build", "maintain", "practice", "enhance", "navigate", "protect", "honor", "defend", "feel", "ensure"],
        "nouns": [
            "situational awareness when walking alone", "home security systems and safety", "managing hyper-vigilance and fear",
            "feeling safe in my neighborhood", "defending personal safety boundaries", "grounded personal safety confidence",
            "home lock security habits", "walking with physical confidence", "safe environment navigation"
        ],
        "handcrafted": [
            "I want to build situational awareness when walking alone at night.",
            "My goal is to install home security systems and feel safe.",
            "I intend to manage hyper-vigilance while honoring personal safety.",
            "I am walking with fierce grounded confidence in any neighborhood.",
            "I choose to defend my physical space and safety boundaries.",
            "I navigate public spaces with sharp situational awareness.",
            "I secure my home and live with peace and physical confidence.",
            "My focus is on personal safety situational awareness and home security."
        ]
    },
    "Retirement & Golden Years Transition": {
        "verbs": ["step into", "embrace", "navigate", "manage", "enjoy", "structure", "transition to", "honor", "welcome", "build"],
        "nouns": [
            "retiring comfortably on a fixed income", "golden years freedom and relaxation", "pension management and retirement funds",
            "structuring unstructured time after work", "life after a long professional career", "this unwritten chapter of freedom",
            "meaningful retirement hobbies and travel", "golden years peace and fulfillment", "redefining identity in retirement"
        ],
        "handcrafted": [
            "I want to step joyfully into retirement and golden years freedom.",
            "My goal is to manage my fixed income and pension funds with ease.",
            "I intend to structure unstructured time after a full career.",
            "I am embracing life after work with passion and purpose.",
            "I choose to honor my decades of hard work in retirement.",
            "I redefine my identity and flourish in my golden years.",
            "I enjoy peaceful days filled with meaningful retirement leisure.",
            "My focus is on retirement transition fixed income and golden years."
        ]
    },
    "Freelancing & Solopreneurship Survival": {
        "verbs": ["protect", "raise", "price", "manage", "resolve", "assert", "build", "navigate", "collect", "honor"],
        "nouns": [
            "overdue invoices and late client payments", "scope creep boundaries and client demands", "freelance pricing and raising my rates",
            "solopreneur business autonomy and freedom", "pricing my worth without apology", "freelance financial stability and income",
            "managing scope creep and client contracts", "solopreneur career independence", "setting professional freelance rates"
        ],
        "handcrafted": [
            "I want to handle overdue invoices and manage scope creep gracefully.",
            "My goal is to raise my freelance rates and price my worth without apology.",
            "I intend to protect my creative time and solopreneur energy.",
            "I am asserting strong contract boundaries against client scope creep.",
            "I choose to collect overdue invoices promptly with professional calm.",
            "I build a thriving solopreneur business based on high value.",
            "I price my freelance services confidently and attract ideal clients.",
            "My focus is on freelancing solopreneur survival and fair pricing."
        ]
    },
    "Immigration, Visas & Bureaucracy": {
        "verbs": ["navigate", "await", "prepare for", "complete", "handle", "manage", "endure", "secure", "submit", "trust"],
        "nouns": [
            "green card application and visa processing", "embassy interview and visa approval", "passport renewals and immigration paperwork",
            "bureaucratic waiting periods and stress", "work visa sponsorship and legal status", "navigating legal immigration bureaucracy",
            "securing my permanent residence card", "embassy paperwork and document submission", "immigration status peace of mind"
        ],
        "handcrafted": [
            "I want to navigate my green card application with patience and calm.",
            "My goal is to prepare for my embassy interview and visa approval.",
            "I intend to handle bureaucratic immigration paperwork with ease.",
            "I am enduring the visa waiting period with unshakeable inner peace.",
            "I choose to trust the legal process for my work visa approval.",
            "I secure my permanent residence and feel safe in my home.",
            "I manage official embassy paperwork with clarity and organization.",
            "My focus is on immigration visas bureaucracy and legal status."
        ]
    },
    "Shared Living & Roommate Dynamics": {
        "verbs": ["cultivate", "enforce", "manage", "split", "respect", "communicate", "build", "maintain", "create", "protect"],
        "nouns": [
            "roommate communication and chore charts", "splitting utility bills and shared expenses", "shared apartment quiet hours and rules",
            "living with roommates and healthy boundaries", "peaceful shared living environment", "respecting personal space in a shared home",
            "roommate conflict resolution and harmony", "fair division of household chores", "comfortable shared apartment living"
        ],
        "handcrafted": [
            "I want to cultivate a peaceful shared apartment with my roommates.",
            "My goal is to split utility bills and chore charts fairly.",
            "I intend to communicate roommate boundaries clearly and calmly.",
            "I am enforcing quiet hours and respecting shared living spaces.",
            "I choose to resolve roommate conflicts with mature open dialogue.",
            "I enjoy a clean harmonious home with respectful housemates.",
            "I protect my daily peace while living with roommates.",
            "My focus is on shared living roommate dynamics and household harmony."
        ]
    },
    "Pre-Marital & Financial Merging": {
        "verbs": ["build", "combine", "discuss", "align", "create", "unite", "navigate", "establish", "plan", "merge"],
        "nouns": [
            "prenup discussions and pre-marital alignment", "combining bank accounts and joint finances", "financial transparency with my partner",
            "joint bank account management and savings", "wedding budget planning and shared goals", "pre-marital financial agreement",
            "open money conversations in relationships", "uniting financial visions for marriage", "shared marital wealth planning"
        ],
        "handcrafted": [
            "I want to discuss pre-marital finances and prenup agreements openly.",
            "My goal is to combine bank accounts with total transparency.",
            "I intend to align our joint wedding budget and financial vision.",
            "I am building an unbreakable foundation of trust and money harmony.",
            "I choose to navigate pre-marital financial merging with love.",
            "We unite our financial goals into a strong shared future.",
            "I embrace honest money conversations with my future spouse.",
            "My focus is on pre-marital financial merging and joint accounts."
        ]
    },
    "Urban Commuting & Public Transit": {
        "verbs": ["protect", "navigate", "handle", "regulate", "endure", "transform", "manage", "maintain", "overcome", "survive"],
        "nouns": [
            "subway delays and rush hour transit", "public transportation sensory exhaustion", "daily urban commuting stress and noise",
            "crowded train platforms and bus rides", "regulating my nervous system during transit", "grounded energy during peak commute hours",
            "mindful transit commuting habits", "city transit noise and crowding", "arriving calm after a long commute"
        ],
        "handcrafted": [
            "I want to protect my internal peace during subway delays.",
            "My goal is to regulate my nervous system on crowded trains.",
            "I intend to navigate rush hour transit with calm grounded energy.",
            "I am overcoming public transportation sensory exhaustion.",
            "I choose to use my daily commute as a quiet mindful retreat.",
            "I arrive at my destination calm focused and completely regulated.",
            "I handle city transit delays with ease and unshakeable patience.",
            "My focus is on urban commuting public transit and commute calm."
        ]
    },
    "Independent Publishing & Creative Launch": {
        "verbs": ["step into", "launch", "publish", "share", "run", "overcome", "release", "celebrate", "present", "embrace"],
        "nouns": [
            "self-publishing a book or creative work", "Kickstarter campaigns and indie funding", "launch day vulnerability and excitement",
            "pressing the publish button with courage", "indie creative release and book launches", "sharing my authentic creative gifts",
            "overcoming launch day vulnerability", "independent author publishing success", "presenting my art to the world"
        ],
        "handcrafted": [
            "I want to press the publish button and self-publish my book.",
            "My goal is to run a successful Kickstarter campaign for my launch.",
            "I intend to embrace launch day vulnerability with pride and joy.",
            "I am sharing my authentic creative gifts fearlessly with the world.",
            "I choose to celebrate my indie release and creative launch.",
            "I transform creative vulnerability into my greatest strength.",
            "I publish my work independently and trust its value.",
            "My focus is on independent publishing creative launch and self-publishing."
        ]
    },
    "Value & Long-Term Investing": {
        "verbs": ["conduct", "analyze", "master", "build", "evaluate", "read", "secure", "maintain", "practice", "grow"],
        "nouns": [
            "fundamental analysis and balance sheets", "ROIC calculation and stock valuation", "long-term investor mindset and patience",
            "resilient investment portfolio growth", "reading financial statements and balance sheets", "discounted cash flow valuation model",
            "disciplined value investing strategy", "analyzing company fundamentals and moat", "long-term wealth accumulation rules"
        ],
        "handcrafted": [
            "I want to conduct fundamental analysis and read company balance sheets.",
            "My goal is to calculate ROIC and master stock valuation discipline.",
            "I intend to build a resilient long-term investment portfolio.",
            "I am practicing patience and logic as a long-term value investor.",
            "I choose to evaluate stock valuations with pure fundamental analysis.",
            "I grow my wealth steadily over decades through value investing.",
            "I analyze balance sheets calmly and select strong moat companies.",
            "My focus is on value long-term investing fundamental analysis and ROIC."
        ]
    },
    "Short-Form Video Production": {
        "verbs": ["master", "edit", "export", "arrange", "optimize", "create", "enhance", "produce", "share", "refine"],
        "nouns": [
            "editing reels and short video clips", "aspect ratio optimization and 9x16 framing", "video timeline editing and transition cuts",
            "high video quality export settings", "content creation and reel storytelling", "audio extraction and sound syncing",
            "editing engaging video content", "short-form video production skills", "refining my editing timeline workflow"
        ],
        "handcrafted": [
            "I want to master editing reels and short-form video production.",
            "My goal is to optimize aspect ratio and export high video quality.",
            "I intend to refine my editing timeline and audio sync workflow.",
            "I am creating engaging short video reels with clear vision.",
            "I choose to improve my video editing skills with every single upload.",
            "I share my creative short-form videos with impact and clarity.",
            "I arrange timeline cuts smoothly for maximum audience engagement.",
            "My focus is on short-form video production editing reels and aspect ratio."
        ]
    },
    "Nature & Wildlife Ecotourism": {
        "verbs": ["explore", "reconnect with", "spot", "visit", "trek", "experience", "protect", "enjoy", "discover", "navigate"],
        "nouns": [
            "hill stations and summer season travel", "wildlife spotting in national parks", "nature trails and eco-hiking routes",
            "ecotourism and sustainable travel spots", "reconnecting with nature and wildlife", "peaceful forest hikes and mountain air",
            "exploring protected national parks", "quiet natural world exploration", "wildlife sanctuary trail walks"
        ],
        "handcrafted": [
            "I want to visit hill stations during the summer season.",
            "My goal is to go wildlife spotting in national parks.",
            "I intend to trek on peaceful nature trails and reset my nervous system.",
            "I am embracing ecotourism and sustainable wildlife exploration.",
            "I choose to reconnect deeply with the quiet rhythm of nature.",
            "I explore national parks and marvel at wild ecosystems.",
            "I experience profound peace while walking on mountain nature trails.",
            "My focus is on nature wildlife ecotourism hill stations and national parks."
        ]
    },
    "Independent Software Development": {
        "verbs": ["build", "transform", "code", "design", "deploy", "solve", "architect", "master", "develop", "refactor"],
        "nouns": [
            "problem statement and software architecture", "building an app from scratch", "clean coding standards and practices",
            "cloud deployment and backend APIs", "independent software development tools", "elegant code solutions for real-world problems",
            "building full-stack software applications", "software design patterns and refactoring", "deploying web and mobile app projects"
        ],
        "handcrafted": [
            "I want to formulate a clear problem statement and build an app.",
            "My goal is to write clean code and design software architecture.",
            "I intend to deploy my independent software projects to production.",
            "I am transforming complex problems into elegant usable code.",
            "I choose to master software architecture and build real-world tools.",
            "I build independent software applications that help thousands.",
            "I refactor code gracefully and maintain robust deployment pipelines.",
            "My focus is on independent software development coding and architecture."
        ]
    },
    "Digital Community Moderation": {
        "verbs": ["protect", "enforce", "moderate", "maintain", "cultivate", "filter", "block", "manage", "foster", "guide"],
        "nouns": [
            "comment sections and community guidelines", "blocking online trolls and toxic spam", "digital community moderation rules",
            "online safety and respectful discussion", "protecting digital community spaces", "fostering safe online group interactions",
            "moderating forum comments and user posts", "enforcing community safety guidelines", "healthy online discussion environments"
        ],
        "handcrafted": [
            "I want to moderate comment sections and enforce community guidelines.",
            "My goal is to block internet trolls and maintain online safety.",
            "I intend to cultivate a safe respectful digital community space.",
            "I am protecting our digital discussion group from toxic static.",
            "I choose to filter out negative trolls and foster positive connection.",
            "I enforce community guidelines firmly and fairly every day.",
            "I keep my online community clean safe and welcoming for all.",
            "My focus is on digital community moderation guidelines and online safety."
        ]
    },
    "Personal Vehicle Maintenance": {
        "verbs": ["perform", "check", "maintain", "service", "schedule", "inspect", "ensure", "replace", "manage", "master"],
        "nouns": [
            "routine oil changes and oil filter service", "trusted mechanic visits and vehicle inspections", "checking car mileage and maintenance logs",
            "tire pressure checks and tire rotation", "personal vehicle routine maintenance", "car brake inspections and fluid top-ups", "reliable safe vehicle travel habits",
            "maintaining car health and longevity", "automotive servicing routines"
        ],
        "handcrafted": [
            "I want to schedule a routine oil change with my trusted mechanic.",
            "My goal is to check tire pressure and monitor vehicle mileage.",
            "I intend to perform personal vehicle maintenance for reliable travel.",
            "I am keeping my car in peak condition with regular servicing.",
            "I choose to maintain my vehicle proactively for peace of mind.",
            "I check brake fluids and tire pressure before every long trip.",
            "I manage routine auto maintenance with total independence.",
            "My focus is on personal vehicle maintenance oil change and tire pressure."
        ]
    },
    "Tenant & Landlord Navigation": {
        "verbs": ["navigate", "handle", "manage", "submit", "negotiate", "protect", "communicate with", "enforce", "resolve", "maintain"],
        "nouns": [
            "landlord communications and lease agreements", "rent increase negotiations and tenant rights", "moving apartments and security deposit return",
            "maintenance requests and lease terms", "protecting my peaceful living space", "tenant rights and apartment lease renewals",
            "landlord maintenance request follow-ups", "security deposit protection and apartment living", "tenant landlord communication skills"
        ],
        "handcrafted": [
            "I want to handle landlord communications and lease agreements calmly.",
            "My goal is to negotiate rent increases and protect my security deposit.",
            "I intend to submit maintenance requests and manage moving apartments.",
            "I am standing firm in my rights while renewing my apartment lease.",
            "I choose to protect the comfort and stability of my living space.",
            "I resolve landlord maintenance issues with clarity and confidence.",
            "I navigate apartment lease terms and tenant rights effortlessly.",
            "My focus is on tenant landlord navigation lease rent increase and security deposit."
        ]
    },
    "Holistic Home Organization": {
        "verbs": ["declutter", "organize", "donate", "swap", "clean", "arrange", "transform", "maintain", "simplify", "create"],
        "nouns": [
            "decluttering messy rooms and closet organization", "donating clothes and seasonal wardrobe swaps", "garage storage solutions and neat pantries",
            "creating a clean peaceful home sanctuary", "holistic home organization routines", "organizing messy rooms and clutter-free living",
            "seasonal wardrobe swap and closet clutter", "donating unneeded items and garage storage", "peaceful home clutter-free organization"
        ],
        "handcrafted": [
            "I want to declutter messy rooms and master closet organization.",
            "My goal is to donate unused clothes and execute seasonal swaps.",
            "I intend to optimize garage storage and clear home clutter.",
            "I am transforming my living space into a peaceful functioning sanctuary.",
            "I choose to maintain a clean organized home that reflects my inner peace.",
            "I organize closets and donate old clothes with joy and gratitude.",
            "I create clutter-free spaces that clear static from my mind.",
            "My focus is on holistic home organization decluttering closet and garage storage."
        ]
    },
    "Expert Mentorship & Methodology Study": {
        "verbs": ["study", "master", "find", "analyze", "adopt", "evaluate", "accelerate", "learn from", "apply", "deepen"],
        "nouns": [
            "stock-picking frameworks and valuation methodology", "finding an expert mentor and studying styles", "deep domain study and proven frameworks",
            "investment style research and expert methodology", "accelerating mastery through proven mentorship", "studying expert investment frameworks and valuation",
            "mastering stock-picking valuation methodology", "finding a mentor and evaluating investment styles", "deep study of world-class expert methodologies"
        ],
        "handcrafted": [
            "I want to study stock-picking frameworks and valuation methodology.",
            "My goal is to find an expert mentor and master their investment style.",
            "I intend to engage in deep study of proven expert frameworks.",
            "I am accelerating my domain mastery through rigorous methodology study.",
            "I choose to learn directly from the best minds in my field.",
            "I apply valuation methodology and stock-picking models with confidence.",
            "I evaluate investment styles and adopt proven success habits.",
            "My focus is on expert mentorship methodology study stock-picking and valuation."
        ]
    },
    "Mindful Media Consumption": {
        "verbs": ["read", "watch", "listen to", "stop", "curate", "protect", "enrich", "choose", "limit", "engage with"],
        "nouns": [
            "reading more books and deep literature", "documentaries and intentional educational media", "audiobooks and high-quality podcasts",
            "stopping doom-scrolling and mindless social feeds", "protecting my attention span with mindful media", "intentional entertainment and curated reading",
            "audiobook listening and inspiring documentaries", "reducing screen time and doom-scrolling habits", "enriching my mind with high-quality content"
        ],
        "handcrafted": [
            "I want to read more books and watch educational documentaries.",
            "My goal is to stop doom-scrolling and embrace audiobooks.",
            "I intend to curate my media feeds for intentional entertainment.",
            "I am protecting my attention span and enriching my intellect daily.",
            "I choose high-quality books and documentaries over passive scrolling.",
            "I listen to audiobooks and reclaim hours of focus every day.",
            "I stop mindless scrolling and nourish my mind with great literature.",
            "My focus is on mindful media consumption read books documentaries stop scrolling."
        ]
    },
    "Extracurricular Parenting": {
        "verbs": ["manage", "drive", "coordinate", "balance", "support", "organize", "attend", "navigate", "create", "foster"],
        "nouns": [
            "carpool duties and kids sports schedules", "piano lessons and extracurricular activities", "PTA meetings and school drop-off routines",
            "balancing over-scheduled kids and family time", "supportive parenting for extracurricular growth", "coordinating kids sports carpools and lessons",
            "school drop-off logistics and PTA involvement", "managing busy family schedules and kids activities", "balanced household rhythm for parenting"
        ],
        "handcrafted": [
            "I want to coordinate carpool duties for kids sports and piano lessons.",
            "My goal is to manage school drop-off and attend PTA meetings.",
            "I intend to balance extracurricular activities without over-scheduling.",
            "I am supporting my children's growth while protecting household harmony.",
            "I choose to navigate busy family schedules with patience and calm.",
            "I handle carpool logistics and piano lesson drop-offs smoothly.",
            "I balance kids sports and school activities while keeping my own energy grounded.",
            "My focus is on extracurricular parenting carpool kids sports piano PTA drop-off."
        ]
    },
    "Subscription & Expense Auditing": {
        "verbs": ["audit", "cancel", "negotiate", "identify", "plug", "review", "eliminate", "optimize", "reduce", "track"],
        "nouns": [
            "canceling unused subscriptions and recurring fees", "negotiating monthly utility and internet bills", "identifying hidden fees and auto-pay charges",
            "meticulous financial expense auditing", "plugging money leaks and maximizing monthly savings", "reviewing auto-pay accounts for hidden charges",
            "canceling unwanted subscriptions and negotiating bills", "financial expense audit and bill reduction", "reclaiming money from unused auto-pay services"
        ],
        "handcrafted": [
            "I want to cancel unused subscriptions and negotiate monthly bills.",
            "My goal is to uncover hidden fees and review auto-pay charges.",
            "I intend to perform a thorough financial audit of monthly expenses.",
            "I am plugging financial leaks and maximizing my hard-earned savings.",
            "I choose to take meticulous control of my recurring subscription fees.",
            "I negotiate internet bills and cancel unused gym memberships easily.",
            "I audit my auto-pay accounts regularly to protect my wealth.",
            "My focus is on subscription expense auditing cancel subscription negotiate bill hidden fees."
        ]
    }
}


def build_synthetic_dataset() -> pd.DataFrame:
    rows = []

    for cat_name, data in CATEGORY_VOCAB.items():
        handcrafted_list = data.get("handcrafted", [])
        verbs = data.get("verbs", [])
        nouns = data.get("nouns", [])

        # 1. Add handcrafted sentences
        for sentence in handcrafted_list:
            rows.append({"text": sentence, "category": cat_name})

        # 2. Programmatically generate templated sentences until at least 210 rows per category
        target_count = 210
        seen = set(s.lower() for s in handcrafted_list)

        attempts = 0
        while len([r for r in rows if r["category"] == cat_name]) < target_count and attempts < 3000:
            attempts += 1
            template = random.choice(TEMPLATES)
            verb = random.choice(verbs)
            noun = random.choice(nouns)
            sentence = template.format(verb=verb, noun=noun)

            if sentence.lower() not in seen:
                seen.add(sentence.lower())
                rows.append({"text": sentence, "category": cat_name})

    df = pd.DataFrame(rows)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = build_synthetic_dataset()
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[Dataset Generator] Created synthetic dataset at: {OUTPUT_CSV}")
    print(f"[Dataset Generator] Total rows: {len(df)}")
    print(f"[Dataset Generator] Categories count: {df['category'].nunique()}")
    print("[Dataset Generator] Class breakdown:")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
