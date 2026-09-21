"""
NLP/processor.py
----------------
Phase 1 – Lexical Processing: Spell Correction (pyspellchecker), Tokenization & Lemmatization
Phase 2 – Syntactic Processing: Dependency Parsing, POS Tagging & S-V-O Locus of Control Analysis
Phase 3 – Semantic Processing: Scikit-Learn Machine Learning Intent Classifier (TF-IDF + LinearSVC) & Named Entity Recognition (NER)
Phase 4 – Discourse Integration: TextRank Sentence Summarization & NLTK VADER Sentiment Analysis
Phase 5 – Pragmatic Processing: Temporal Focus (Rumination vs. Anticipatory Anxiety), Cognitive Pattern Flags & NLG Modular Assembly
--------------------------------------------------
100% local execution — no external APIs.
"""

import os
os.environ["GIT_PYTHON_REFRESH"] = "0"

import random
import re
import ssl
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
import pytextrank
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  SETUP & MODEL INITIALIZATION                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── SSL Bypass for Silent NLTK Downloads ─────────────────────────────────────
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

try:
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
except Exception as _exc:
    print(f"[Mindify] Warning during NLTK download: {_exc}", flush=True)

# Instantiate NLTK VADER Sentiment Intensity Analyzer & PySpellChecker
sia = SentimentIntensityAnalyzer()
spell = SpellChecker()

# ── spaCy Model Loading with PyTextRank Pipeline Extension ──────────────────
MODEL_NAME = "en_core_web_md"


def _load_spacy_model() -> spacy.Language:
    try:
        model = spacy.load(MODEL_NAME)
    except OSError:
        print(
            f"[Mindify] spaCy model '{MODEL_NAME}' not found. "
            "Downloading now (one-time setup)…",
            flush=True,
        )
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", MODEL_NAME],
                check=True,
            )
        except Exception:
            url = f"https://github.com/explosion/spacy-models/releases/download/{MODEL_NAME}-3.7.1/{MODEL_NAME}-3.7.1-py3-none-any.whl"
            subprocess.run(["uv", "pip", "install", url], check=False)
        model = spacy.load(MODEL_NAME)

    if "textrank" not in model.pipe_names:
        model.add_pipe("textrank")

    return model


nlp: spacy.Language = _load_spacy_model()

# ── SentenceTransformer & Scikit-Learn Multi-Label Intent Classifier Loading 
MODEL_PKL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "intent_classifier.pkl")
TRANSFORMER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "transformer_classifier.pkl")
LABEL_BINARIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "label_binarizer.pkl")

# Cache SentenceTransformer embedder in memory on startup
print("[Mindify] Loading SentenceTransformer('all-MiniLM-L6-v2')...", flush=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def _load_intent_model():
    """
    Load the pre-trained Scikit-Learn TF-IDF + LinearSVC single-intent classifier model.
    """
    if os.path.exists(MODEL_PKL_PATH):
        return joblib.load(MODEL_PKL_PATH)

    print("[Mindify] Single-intent model pkl not found. Generating dataset and training model...", flush=True)
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    gen_script = os.path.join(scripts_dir, "generate_dataset.py")
    train_script = os.path.join(scripts_dir, "train_model.py")

    subprocess.run([sys.executable, gen_script], check=True)
    subprocess.run([sys.executable, train_script], check=True)

    return joblib.load(MODEL_PKL_PATH)


def _load_transformer_model():
    """
    Load the pre-trained MultiOutputClassifier(LogisticRegression) and MultiLabelBinarizer.
    Triggers automatic training if missing.
    """
    if os.path.exists(TRANSFORMER_MODEL_PATH) and os.path.exists(LABEL_BINARIZER_PATH):
        try:
            clf = joblib.load(TRANSFORMER_MODEL_PATH)
            binarizer = joblib.load(LABEL_BINARIZER_PATH)
            return clf, binarizer
        except Exception:
            pass

    print("[Mindify] Transformer classifier artifacts not found. Training model...", flush=True)
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    train_script = os.path.join(scripts_dir, "train_transformer_model.py")
    subprocess.run([sys.executable, train_script], check=True)

    clf = joblib.load(TRANSFORMER_MODEL_PATH)
    binarizer = joblib.load(LABEL_BINARIZER_PATH)
    return clf, binarizer


intent_model = _load_intent_model()
transformer_clf, label_binarizer = _load_transformer_model()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║               PHASE 1 – LEXICAL PROCESSING (SPELL CORRECTION)            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _correct_spelling(text: str) -> tuple[str, list[dict[str, str]]]:
    """
    Phase 1 (Lexical): Tokenize raw input text, identify misspellings using
    pyspellchecker, correct misspelled tokens, and return reconstructed string
    alongside a list of corrections for diagnostic output.
    """
    words = re.findall(r"\b[A-Za-z]+\b", text)
    if not words:
        return text, []

    unknown = spell.unknown(words)
    corrections_list: list[dict[str, str]] = []
    corrected_text = text

    for word in unknown:
        if len(word) <= 2 or word.isupper():
            continue

        correction = spell.correction(word)
        if correction and correction.lower() != word.lower():
            corrections_list.append({"original": word, "corrected": correction})
            pattern = re.compile(rf"\b{re.escape(word)}\b")
            corrected_text = pattern.sub(correction, corrected_text)

    return corrected_text, corrections_list


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║             PHASE 2 – SYNTACTIC PROCESSING (LOCUS OF CONTROL)            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _analyze_locus_of_control(doc: spacy.tokens.Doc) -> str:
    """
    Phase 2 (Syntactic): Perform Subject-Verb-Object (S-V-O) dependency parsing
    to classify the user's Locus of Control.

      - Internal (Self-Empowered): Main subject is first-person ("I", "me", "my", "we").
      - External (Other-Dependent): Main subject is a third party (e.g. "boss", "market", "exam").
    """
    root_verb = next((t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)

    subj = None
    if root_verb:
        subj = next(
            (c for c in root_verb.children if c.dep_ in {"nsubj", "nsubjpass"}),
            None,
        )

    if not subj:
        subj = next(
            (t for t in doc if t.dep_ in {"nsubj", "nsubjpass"}),
            None,
        )

    if subj:
        subj_lemma = subj.lemma_.lower()
        if subj_lemma in {"i", "me", "my", "myself", "we", "us", "our"}:
            return "Internal (Self-Empowered)"
        else:
            return f"External (Other-Dependent: '{subj.text}')"

    return "Internal (Self-Empowered)"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           PHASE 3 – SEMANTIC PROCESSING (NER & ML INTENT CLASSIFIER)     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _extract_entities(doc: spacy.tokens.Doc) -> list[dict[str, str]]:
    """
    Phase 3 (Semantic): Scan doc.ents for Named Entities (ORG, PERSON, DATE, TIME, GPE).
    Returns structured list of extracted entity dictionaries.
    """
    entities: list[dict[str, str]] = []
    target_labels = {"ORG", "PERSON", "DATE", "TIME", "GPE", "WORK_OF_ART", "EVENT", "LAW"}

    for ent in doc.ents:
        if ent.label_ in target_labels:
            entities.append({
                "text": ent.text.strip(),
                "label": ent.label_,
            })

    return entities


def _predict_intent_ml(goal_sentence_text: str) -> str:
    """
    Phase 3 (Semantic): Predict single intent category using trained Scikit-Learn TF-IDF + LinearSVC model.
    """
    try:
        predicted = intent_model.predict([goal_sentence_text])[0]
        return str(predicted)
    except Exception:
        return "Default"


def _predict_multilabel_intents(goal_sentence_text: str) -> tuple[list[str], list[dict[str, float]], list[dict[str, str]]]:
    """
    Phase 3 (Semantic): Predict intent domains using SentenceTransformer embeddings + Calibrated MultiOutputClassifier.
    Uses Dynamic Confidence Calibration & Adaptive Margin Thresholding:
    - Sorts probabilities for all 90 classes in descending order.
    - Automatically selects primary label (top_1).
    - Selects secondary label (top_2) ONLY if its probability is within 0.12 of top_1.
    - Outputs top_intents_with_probabilities for the top 3 predicted categories with exact calibrated confidence.
    """
    try:
        user_vec = embedder.encode([goal_sentence_text])
        probs_list = transformer_clf.predict_proba(user_vec)

        classes = label_binarizer.classes_
        class_probs = []
        for i, class_name in enumerate(classes):
            prob = float(probs_list[i][0][1])
            class_probs.append((class_name, prob))

        # Sort probabilities in descending order
        class_probs.sort(key=lambda x: x[1], reverse=True)

        top_1_cat, top_1_prob = class_probs[0]
        top_2_cat, top_2_prob = class_probs[1]

        detected_categories = [top_1_cat]
        # Margin thresholding: select top_2 ONLY if within 0.12 margin of top_1
        if (top_1_prob - top_2_prob) <= 0.12 and top_2_prob >= 0.10:
            detected_categories.append(top_2_cat)

        top_intents_with_probabilities = [
            {"category": cat, "probability": round(prob, 4)}
            for cat, prob in class_probs[:3]
        ]

        debug_payload = [
            {"category": cat, "probability": round(prob, 4), "status": "matched"}
            for cat, prob in class_probs
            if cat in detected_categories
        ]
        return detected_categories, top_intents_with_probabilities, debug_payload
    except Exception:
        fallback = _predict_intent_ml(goal_sentence_text)
        fallback_top = [{"category": fallback, "probability": 1.0}]
        return [fallback], fallback_top, [{"category": fallback, "status": "single_fallback", "probability": 1.0}]


def _extract_intent(doc: spacy.tokens.Doc | spacy.tokens.Span) -> tuple[str, str]:
    """Extract primary verb and direct object via dependency parse."""
    root = next((t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)
    if root:
        xcomp = next((c for c in root.children if c.dep_ == "xcomp" and c.pos_ == "VERB"), None)
        target_verb = xcomp if xcomp else root

        verb_lemma = target_verb.lemma_.lower()
        dobj = next(
            (c for c in target_verb.children if c.dep_ == "dobj"),
            next((c for c in root.children if c.dep_ == "dobj"), None),
        )
        noun_lemma = dobj.lemma_.lower() if dobj else ""
        return verb_lemma, noun_lemma

    content = [t for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    verb = next((t.lemma_.lower() for t in content if t.pos_ == "VERB"), "")
    noun = next(
        (t.lemma_.lower() for t in content if t.pos_ in {"NOUN", "PROPN"}), ""
    )
    return verb, noun


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║        PHASE 4 – PRAGMATIC PROCESSING (TEMPORAL FOCUS & ANXIETY)         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PROBLEMATIC_PATTERNS: dict[str, list[str]] = {
    "Absolutist": [
        "never", "always", "every", "none", "all", "everything",
        "nothing", "everybody", "nobody",
    ],
    "Negative Framing": [
        "quit", "stop", "fail", "avoid", "lose", "hate", "waste", "bad",
        "wrong", "not", "can't", "cannot", "don't", "won't", "impossible",
        "give up",
    ],
    "Rigid/Modal": [
        "must", "should", "need", "have", "ought", "supposed",
        "required", "force", "compel",
    ],
}


def _analyze_temporal_focus(
    doc: spacy.tokens.Doc, flags: dict[str, list[str]]
) -> tuple[str, dict[str, list[str]]]:
    """
    Phase 4 (Pragmatic): Analyze verb tenses and modal auxiliaries to detect
    Temporal Anxiety / Focus.
    """
    past_count = sum(1 for t in doc if t.tag_ in {"VBD", "VBN"})
    future_count = sum(
        1 for t in doc
        if t.tag_ == "MD" or t.text.lower() in {"will", "going", "tomorrow", "next", "future", "soon"}
    )

    if past_count > future_count and past_count > 0:
        flags["Rumination"] = ["past events / regret focus"]
        return "Past Focus (Rumination)", flags
    elif future_count > past_count and future_count > 0:
        flags["Anticipatory Anxiety"] = ["future uncertainty / worry focus"]
        return "Future Focus (Anticipatory Anxiety)", flags
    else:
        return "Present Grounded", flags


def generate_even_if(
    category: str,
    flags: dict[str, list[str]],
    venting_compound: float = 0.0,
    temporal_focus: str = "Present Grounded",
) -> str:
    """
    Construct an emotion-aware, pragmatic "Even if…" resilience statement.
    """
    if venting_compound <= -0.5:
        return (
            "Even when my anxiety feels overwhelming and I fear falling behind, "
            "I choose to offer myself grace and recognize the courage it takes to keep trying."
        )

    if temporal_focus == "Past Focus (Rumination)" or "Rumination" in flags:
        return (
            "Even if past attempts didn't turn out as expected, "
            "I release old regrets and trust that every day brings a fresh opportunity to grow."
        )
    elif temporal_focus == "Future Focus (Anticipatory Anxiety)" or "Anticipatory Anxiety" in flags:
        return (
            "Even if future uncertainties cause temporary worry, "
            "I remain anchored in the present moment and focus on what I can control today."
        )

    if "Rigid/Modal" in flags:
        return (
            "Even if things don't go perfectly as planned, "
            "I adapt with grace and keep moving forward."
        )
    if "Negative Framing" in flags:
        return (
            "Even if I experience setbacks, I forgive myself "
            "and return to my positive habits."
        )
    if "Absolutist" in flags:
        return (
            "Even if my progress isn't perfect or absolute, "
            "every small step still counts and compounds over time."
        )

    primary_category = category.split(" & ")[0].strip()
    category_fallbacks = {
        "Health, Wellness & Lifestyle": "Even if progress feels slow, I trust that my body is healing and getting stronger.",
        "Financial Goals & Wealth": "Even if the results aren't immediate, I trust in the compound effect of my daily efforts.",
        "Career & Professional Growth": "Even if professional growth takes time, every effort builds my capability.",
        "Relationships & Social Connections": "Even if misunderstandings arise, I stay open and communicate with love.",
        "Education & Academic Milestones": "Even if concepts feel difficult today, consistent study brings understanding.",
        "Spiritual Growth & Alignment": "Even if life feels noisy, I remain aligned with my higher purpose.",
    }
    return category_fallbacks.get(
        primary_category,
        "Even if the path feels uncertain, I move forward with trust in my own capacity to grow.",
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 5 – NLG (MASSIVE NUANCED CATEGORY MODULAR ASSEMBLY DICTIONARY)    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# RULE 4: NEUTRAL BRIDGE THOUGHTS (For Negative Sentiment)
BRIDGE_COMPONENTS = {
    "openers": [
        "I am learning to focus on",
        "I am practicing patience with my",
        "Step by step, I am exploring how to improve my",
        "I am open to the possibility of healing my",
        "I am slowly building better habits around my"
    ],
    "closers": [
        "and that is enough for today.",
        "and I give myself grace in this process.",
        "without needing to be perfect.",
        "one small step at a time."
    ]
}

# 4-PART SMART FORMULA DICTIONARY (Targeting 10-15 words average)
SMART_AFFIRMATION_COMPONENTS = {
    "action_verbs": [
        "I am confidently executing",
        "I consistently master",
        "I am actively building",
        "I calmly navigate",
        "I am intentionally focusing on"
    ],
    "emotional_benefits": {
        "Default": [
            "to cultivate deep inner peace.",
            "which brings me immense joy.",
            "to build absolute trust in myself."
        ],
        "Career & Professional Growth": [
            "to secure my profound professional success.",
            "which fuels my deep sense of purpose.",
            "to unlock new career breakthroughs."
        ],
        "Financial Goals & Wealth": [
            "to create absolute financial freedom.",
            "which grounds me in total abundance.",
            "to build lasting generational prosperity."
        ],
        "Health, Wellness & Lifestyle": [
            "to experience vibrant daily vitality.",
            "which honors my body's true needs.",
            "to feel energized and physically strong."
        ],
        "Relationships & Social Connections": [
            "to cultivate deep, meaningful bonds.",
            "which brings mutual warmth and harmony.",
            "to foster unconditional love and trust."
        ],
        "Environment & Material Upgrades": [
            "to design a serene, inspiring sanctuary.",
            "which nurtures my daily peace of mind.",
            "to elevate my physical surroundings."
        ],
        "Mindset & Emotional Well-Being": [
            "to anchor unshakable inner calm.",
            "which strengthens my emotional stability.",
            "to cultivate radical self-compassion."
        ],
        "Personal Habits & Time Management": [
            "to build unbroken daily momentum.",
            "which gives me complete control of my day.",
            "to compound small actions into greatness."
        ],
        "Creativity, Hobbies & Learning": [
            "to express my unique creative vision.",
            "which expands my artistic boundaries.",
            "to experience pure creative flow."
        ],
        "Social Impact & Contribution": [
            "to create a lasting positive footprint.",
            "which serves the greater good.",
            "to touch lives with genuine kindness."
        ],
        "Education & Academic Milestones": [
            "to achieve total intellectual mastery.",
            "which opens doors to endless opportunities.",
            "to excel in my academic pursuits."
        ],
        "Independence & Freedom": [
            "to live strictly on my own terms.",
            "which unlocks total personal autonomy.",
            "to experience true life sovereignty."
        ],
        "Spiritual Growth & Alignment": [
            "to align with my highest spiritual self.",
            "which grounds me in divine trust.",
            "to experience deep soul presence."
        ],
        "Major Life & Family Milestones": [
            "to build a strong legacy of love.",
            "which brings deep joy to my family.",
            "to honor every cherished milestone."
        ],
        "Legal & Administrative Resolution": [
            "to bring complete order and resolution.",
            "which restores my total peace of mind.",
            "to handle my affairs with clarity."
        ],
        "Intellectual & Cognitive Development": [
            "to sharpen my mental clarity every day.",
            "which expands my cognitive capacity.",
            "to absorb complex insights effortlessly."
        ],
        "Athletic & Physical Performance": [
            "to unlock peak physical endurance.",
            "which elevates my athletic strength.",
            "to shatter my personal records."
        ],
        "Eco-Conscious & Sustainable Living": [
            "to live in deep harmony with the earth.",
            "which honors my ecological footprint.",
            "to build a sustainable future."
        ],
        "Long-Term Relationship Longevity": [
            "to deepen an unshakeable lifelong bond.",
            "which enriches our shared partnership.",
            "to grow closer together every single day."
        ],
        "Crisis Recovery & Fresh Starts": [
            "to build a brilliant new beginning.",
            "which restores my inner resilience.",
            "to transform challenges into strength."
        ],
        "Digital Detox & Tech Balance": [
            "to reclaim full sovereignty over my focus.",
            "which cultivates deep presence in life.",
            "to restore my mental space."
        ],
        "Mindful Nutrition & Culinary Arts": [
            "to nourish my body with wholesome fuel.",
            "which vitalizes my physical health.",
            "to enjoy conscious, healthy dining."
        ],
        "Travel, Exploration & Adventure": [
            "to expand my horizons with wonder.",
            "which enriches my soul with new experiences.",
            "to embrace exciting global discovery."
        ],
        "Parenting & Child Development": [
            "to guide my family with patient love.",
            "which creates a safe, nurturing sanctuary.",
            "to instill resilience and confidence."
        ],
        "Sleep, Rest & Recovery Science": [
            "to invite deep, restorative nightly sleep.",
            "which recharges my energy for tomorrow.",
            "to honor my body's need for rest."
        ],
        "Generational Legacy & Ancestral Healing": [
            "to build a lasting multi-generational legacy.",
            "which secures prosperity for my family.",
            "to anchor strength in my bloodline."
        ],
        "Shadow Work & Deep Unconscious Integration": [
            "to embrace total self-acceptance and wholeness.",
            "which transmutes fear into wisdom.",
            "to shine awareness into my life."
        ],
        "Civic Leadership & Systemic Change": [
            "to drive meaningful positive systemic change.",
            "which elevates my community with justice.",
            "to lead with brave integrity."
        ],
        "Radical Detachment & Monk Mode": [
            "to cultivate deep, uninterrupted focus.",
            "which grounds me in quiet discipline.",
            "to experience freedom from distraction."
        ],
        "Interpersonal Severing & Deep Closure": [
            "to honor sacred personal peace and boundaries.",
            "which allows me to move forward with grace.",
            "to step into full emotional clarity."
        ],
        "Advanced Esoteric & Metaphysical Mastery": [
            "to align with high universal energy.",
            "which manifests rapid quantum progress.",
            "to master conscious creation."
        ],
        "Neurodivergence & Executive Function": [
            "to work in harmony with my natural energy.",
            "which honors how my mind operates.",
            "to navigate tasks with ease and grace."
        ],
        "Everyday Adulting & Life Admin": [
            "to clear mental clutter with easy efficiency.",
            "which brings steady order to my day.",
            "to handle my responsibilities smoothly."
        ],
        "Early Parenthood & Childcare": [
            "to remain a calm, loving anchor.",
            "which nurtures my child with presence.",
            "to navigate this season with grace."
        ],
        "Midlife Transitions & Empty Nesting": [
            "to welcome this exciting new chapter of life.",
            "which prioritizes my personal fulfillment.",
            "to embrace fresh possibilities."
        ],
        "Tech, Gear & Digital Lifestyle": [
            "to optimize my setup for peak performance.",
            "which creates an inspiring workspace.",
            "to streamline my digital lifestyle."
        ],
        "Micro-Hobbies & Crafting Mastery": [
            "to experience mindful joy in creating.",
            "which satisfies my artistic detail.",
            "to take pride in my handiwork."
        ],
        "Elder Care & Family Stewardship": [
            "to honor my family while fiercely protecting my own peace.", 
            "which brings me comfort during this deeply difficult season.",
            "knowing I am doing the absolute best I can."
        ],
        "Workplace Dynamics & Colleague Boundaries": [
            "to maintain absolute professional peace and distance.", 
            "which protects my energy from corporate stress.",
            "knowing my self-worth is entirely separate from my job."
        ],
        "Chronic Illness & Invisible Disability": [
            "to honor my body’s unique daily limits with profound grace.", 
            "which allows me to thrive safely at my own pace.",
            "knowing that rest is a productive and necessary medical requirement."
        ],
        "Substance Recovery & Vice Cessation": [
            "to reclaim absolute control over my life and my future.", 
            "which heals my body and mind every single day.",
            "to build a clean, peaceful reality that I never want to escape from."
        ],
        "Event Planning & Milestone Hosting": [
            "to ensure a beautifully memorable experience.", 
            "which allows me to actually enjoy the celebration I created.",
            "knowing that perfection is not required for people to have a good time."
        ],
        "Homeownership & Property Maintenance": [
            "to build a safe, secure, and functioning sanctuary.", 
            "which grounds me deeply in my physical space.",
            "knowing that I am highly capable of managing my environment."
        ],
        "Adult Friendship & Platonic Intimacy": [
            "to cultivate deep, meaningful, and effortless platonic love.", 
            "which brings absolute joy and profound connection to my daily life.",
            "knowing I am deeply worthy of a supportive chosen family."
        ],
        "Intuitive Eating & Diet Culture Recovery": [
            "to rebuild absolute trust and neutrality with my incredible body.", 
            "which liberates my mind from the exhausting cycle of food guilt.",
            "knowing that nourishment is a beautiful, necessary right."
        ],
        "Hard Conversations & Conflict Navigation": [
            "to fiercely advocate for my needs with absolute clarity and calm.", 
            "which clears the air and builds much deeper, honest connections.",
            "knowing my voice deserves to be heard without apology."
        ],
        "Guilt-Free Rest & Hustle Culture Detox": [
            "to fully absorb the beautiful, healing power of doing absolutely nothing.", 
            "which honors my profound need for deep, uninterrupted rest.",
            "knowing my worth is entirely disconnected from my productivity."
        ],
        "Career Re-entry & Late-in-Life Pivots": [
            "to fearlessly embrace this exciting, brand-new chapter of my journey.", 
            "which proves that it is never, ever too late to reinvent myself.",
            "knowing my past life experience makes me uniquely powerful in this new space."
        ],
        "Nervous System Regulation & Somatic Healing": [
            "to gently guide my beautiful body back to a state of profound safety.", 
            "which physically melts the stress out of my muscles and mind.",
            "knowing I have the absolute power to calm my own biological storm."
        ],
        "Modern Dating & Romantic Vulnerability": [
            "to open my heart fully without fear of the outcome.", 
            "which allows me to date with absolute confidence and boundaries.",
            "knowing my worth is never determined by someone else's rejection."
        ],
        "Academic Pressure & High-Stakes Testing": [
            "to prove my absolute mastery of this challenging material.", 
            "which brings me one massive step closer to my ultimate career vision.",
            "knowing my intelligence and preparation are more than enough."
        ],
        "Sleep Optimization & Circadian Health": [
            "to grant my brain and body the profound rest they deserve.", 
            "which allows me to wake up with vibrant, natural energy.",
            "knowing that peaceful sleep is the foundation of my entire life."
        ],
        "Driving Confidence & Commute Anxiety": [
            "to navigate the roads with absolute calm, safety, and control.", 
            "which gives me the profound freedom to go wherever I choose.",
            "knowing I am a highly capable, deeply focused, and safe driver."
        ],
        "Language Acquisition & Accent Confidence": [
            "to connect deeply with entirely new cultures and people.", 
            "which expands my world and builds immense cognitive flexibility.",
            "knowing that perfection is entirely unnecessary for true connection."
        ],
        "Physical Rehabilitation & Physiotherapy": [
            "to honor the slow, beautiful process of my body's healing.", 
            "which actively restores my strength and pain-free mobility.",
            "knowing that every small stretch is a massive step forward."
        ],
        "Solo Living & Domestic Independence": [
            "to cultivate a sanctuary of profound peace and self-reliance.", 
            "which proves I am deeply capable of taking care of myself.",
            "knowing that my own company is beautiful and entirely enough."
        ],
        "Time Management & Punctuality Discipline": [
            "to respect my own energy and the boundaries of others.", 
            "which replaces chaotic rushing with absolute, grounded calm.",
            "knowing that I am in total control of my daily schedule."
        ],
        "Financial Literacy & Basic Investing": [
            "to build a quiet, unshakeable foundation for my future.", 
            "which replaces financial fear with absolute, educated empowerment.",
            "knowing my money is working tirelessly to secure my freedom."
        ],
        "Digital Declutter & Cybersecurity Hygiene": [
            "to reclaim my digital peace and protect my private identity.", 
            "which clears the mental static from my daily screen time.",
            "knowing I am entirely secure and organized in the modern world."
        ],
        "Personal Safety & Situational Awareness": [
            "to navigate my environment with fierce, grounded confidence.", 
            "which protects my peace and honors my right to take up space.",
            "knowing I am highly capable of defending my own boundaries."
        ],
        "Retirement & Golden Years Transition": [
            "to step joyfully into this beautiful, unwritten chapter of freedom.", 
            "which honors the decades of hard work I have already completed.",
            "knowing my identity is vast, profound, and entirely mine to shape."
        ],
        "Freelancing & Solopreneurship Survival": [
            "to fiercely protect the value of my time and creative energy.", 
            "which replaces financial anxiety with absolute, unwavering confidence.",
            "knowing my expertise is deeply valuable and worthy of compensation."
        ],
        "Immigration, Visas & Bureaucracy": [
            "to navigate this complex system with unshakeable patience and grace.", 
            "which grounds me in deep peace while I wait for my approval.",
            "knowing my future is secure and I belong exactly where I choose to be."
        ],
        "Shared Living & Roommate Dynamics": [
            "to cultivate a deeply respectful and peaceful shared environment.", 
            "which protects my daily peace and enforces my healthy boundaries.",
            "knowing I deserve to feel completely comfortable in my own home."
        ],
        "Pre-Marital & Financial Merging": [
            "to build an unbreakable foundation of trust and transparency.", 
            "which unites us deeply in our shared vision for the future.",
            "knowing that honest conversations create the strongest partnerships."
        ],
        "Urban Commuting & Public Transit": [
            "to protect my internal peace amidst the chaos of the city.", 
            "which allows me to arrive at my destination grounded and calm.",
            "knowing I can easily regulate my energy in any crowded space."
        ],
        "Independent Publishing & Creative Launch": [
            "to step fearlessly into the light and share my gifts with the world.", 
            "which transforms my deep vulnerability into my greatest strength.",
            "knowing my work deserves to be seen, celebrated, and heard."
        ],
        "Value & Long-Term Investing": [
            "to secure a steadily growing, resilient portfolio.", 
            "which grounds my financial decisions in pure logic and patience.",
            "knowing true wealth is built over decades, not days."
        ],
        "Short-Form Video Production": [
            "to share my creative vision with absolute clarity.", 
            "which allows my content to reach and inspire the right audience.",
            "knowing my editing skills improve with every single upload."
        ],
        "Nature & Wildlife Ecotourism": [
            "to reconnect deeply with the quiet rhythm of the natural world.", 
            "which completely resets my nervous system.",
            "knowing that exploring the earth is a beautiful privilege."
        ],
        "Independent Software Development": [
            "to transform complex problems into elegant, usable solutions.", 
            "which proves my capability to build real-world tools.",
            "knowing my code has the power to help others."
        ],
        "Digital Community Moderation": [
            "to fiercely protect the peace and safety of my digital space.", 
            "which allows genuine connection to thrive.",
            "knowing I control the energy I allow into my life."
        ],
        "Personal Vehicle Maintenance": [
            "to ensure safe, reliable travels wherever I choose to go.", 
            "which gives me profound independence and peace of mind.",
            "knowing I am fully responsible for my own journey."
        ],
        "Tenant & Landlord Navigation": [
            "to fiercely protect the comfort and peace of my living space.", 
            "which gives me a profound sense of stability and control.",
            "knowing I deserve to feel completely at home wherever I am."
        ],
        "Holistic Home Organization": [
            "to create a clear, peaceful, and beautifully functioning sanctuary.", 
            "which instantly clears the anxious clutter from my mind.",
            "knowing my physical space directly reflects my inner peace."
        ],
        "Expert Mentorship & Methodology Study": [
            "to rapidly accelerate my mastery and real-world success.", 
            "which builds unshakeable confidence in my strategic decisions.",
            "knowing I am learning directly from the absolute best."
        ],
        "Mindful Media Consumption": [
            "to deeply enrich my mind and protect my attention span.", 
            "which reconnects me with true, high-quality inspiration.",
            "knowing my time and focus are my most valuable assets."
        ],
        "Extracurricular Parenting": [
            "to support my family's growth while protecting my own energy.", 
            "which creates a beautiful, balanced rhythm for our household.",
            "knowing I am doing an incredible job raising well-rounded humans."
        ],
        "Subscription & Expense Auditing": [
            "to take absolute, meticulous control of my financial resources.", 
            "which plugs every leak and maximizes my hard-earned money.",
            "knowing every dollar I save buys back my future freedom."
        ]
    }
}

# LEGACY AFFIRMATION_COMPONENTS BACKWARD COMPATIBILITY
AFFIRMATION_COMPONENTS = {
    "Default": {
        "openers": ["I am calmly mastering my"],
        "closers": ["right now."]
    }
}


def _get_wordnet_synonym(word: str, pos_type: str = "v") -> str:
    """
    Phase 5 (NLG): Query NLTK WordNet for dynamic, context-appropriate synonyms.
    """
    synsets = wordnet.synsets(word, pos=pos_type)
    if not synsets:
        return word

    candidates: list[str] = []
    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name != word.lower() and len(name.split()) == 1:
                candidates.append(name)

    if candidates:
        return random.choice(list(set(candidates)))
    return word


def _clean_noun_chunk(noun_str: str) -> str:
    """
    NLG Grammar: Strips leading determiners ('the', 'my', 'a', 'an', 'our', 'your')
    and cleans gerund phrases before template slot insertion.
    """
    words = noun_str.strip().split()
    if not words:
        return "goals"

    determiners = {"the", "my", "a", "an", "our", "your", "his", "her", "their"}
    while words and words[0].lower() in determiners:
        words.pop(0)

    cleaned = " ".join(words) if words else noun_str.strip()
    return cleaned if cleaned else "goals"


def _extract_measurable_detail(
    doc: spacy.tokens.Doc | spacy.tokens.Span | None = None,
    intent_noun: str = ""
) -> str:
    """
    Extract a specific, measurable detail or noun phrase using spaCy NER,
    noun chunks, and S-V-O dependency parsing.
    """
    if doc:
        # 1. Look for specific named entities (ORG, MONEY, DATE, QUANTITY, GPE, WORK_OF_ART, LAW)
        entities = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "MONEY", "DATE", "QUANTITY", "GPE", "WORK_OF_ART", "EVENT", "LAW"}]
        if entities:
            clean_ent = _clean_noun_chunk(entities[0])
            if clean_ent:
                return f"my {clean_ent}"

        # 2. Look for explicit noun chunks in doc
        noun_chunks = [nc.text for nc in doc.noun_chunks if nc.text.lower() not in {"i", "me", "my", "myself", "we", "us", "you", "your", "they", "them"}]
        if noun_chunks:
            clean_nc = _clean_noun_chunk(noun_chunks[0])
            if clean_nc and len(clean_nc.split()) <= 4:
                return f"my {clean_nc}"

    clean_noun = _clean_noun_chunk(intent_noun) if intent_noun else "daily priorities"
    return f"my {clean_noun}"


def generate_affirmation(
    intent_verb: str,
    intent_noun: str,
    category: str | list[str],
    flags: dict[str, list[str]],
    vader_compound: float = 0.0,
    doc: spacy.tokens.Doc | spacy.tokens.Span | None = None,
) -> str:
    """
    Construct a 4-part SMART Affirmation naturally averaging 10-15 words:
    [Subject + Positive Action Verb] + [Specific Detail / Measurable Noun] + [Emotional Benefit]
    """
    specific_detail = _extract_measurable_detail(doc=doc, intent_noun=intent_noun)

    # RULE 4: NEUTRAL BRIDGE THOUGHTS (For Negative Sentiment)
    if vader_compound < -0.05:
        opener = random.choice(BRIDGE_COMPONENTS["openers"])
        closer = random.choice(BRIDGE_COMPONENTS["closers"])
        clean_detail = specific_detail[3:] if specific_detail.startswith("my ") else specific_detail
        if opener.endswith("my") or opener.endswith("on"):
            return f"{opener} {clean_detail} {closer}"
        return f"{opener} my {clean_detail} {closer}"

    # Determine primary category
    if isinstance(category, str):
        categories = [c.strip() for c in category.split(",") if c.strip()]
    else:
        categories = list(category) if category else ["Default"]

    primary_cat = categories[0] if categories else "Default"

    # Fetch emotional benefits for category (or Default fallback)
    benefits_list = SMART_AFFIRMATION_COMPONENTS["emotional_benefits"].get(
        primary_cat, SMART_AFFIRMATION_COMPONENTS["emotional_benefits"]["Default"]
    )

    action_verb = random.choice(SMART_AFFIRMATION_COMPONENTS["action_verbs"])
    emotional_benefit = random.choice(benefits_list)

    # 4-PART SMART FORMULA ASSEMBLY
    affirmation = f"{action_verb} {specific_detail} {emotional_benefit}"

    if "Negative Framing" in flags:
        reframe = "I protect my focus purposefully."
        affirmation = f"{affirmation} {reframe}"

    return affirmation


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║            SEMANTIC ENTITY INJECTION (IF-THEN & MICRO-ACTIONS)           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_if_then(category: str, entities: list[dict[str, str]]) -> str:
    """
    Generate IF-THEN behavioural plan, injecting extracted real-world entities (Semantic Phase).
    """
    date_ent = next((e["text"] for e in entities if e["label"] in {"DATE", "TIME"}), None)
    org_ent = next((e["text"] for e in entities if e["label"] in {"ORG", "GPE", "LAW"}), None)

    if date_ent:
        return (
            f"IF I feel overwhelmed before {date_ent}, "
            "THEN I pause, step back, and focus entirely on completing the single next priority step."
        )
    elif org_ent:
        return (
            f"IF I experience distractions regarding {org_ent}, "
            "THEN I clear my desk and commit to 15 minutes of uninterrupted focus."
        )

    plans = {
        "Health, Wellness & Lifestyle": "IF I feel tempted to skip my healthy routine, THEN I take the smallest positive step.",
        "Financial Goals & Wealth": "IF I feel the urge to make an impulsive decision, THEN I pause for 10 minutes and re-evaluate my long-term strategy.",
        "Education & Academic Milestones": "IF I feel overwhelmed studying for exams, THEN I use a 25-minute Pomodoro timer.",
        "Legal & Administrative Resolution": "IF I feel anxious about paperwork, THEN I review one single document at a time.",
    }
    return plans.get(category, "IF I feel unmotivated, THEN I take three deep breaths and commit to 5 minutes of focused effort.")


def generate_micro_action(category: str, entities: list[dict[str, str]]) -> str:
    """
    Generate a 2-minute micro-action task, injecting extracted entities (Semantic Phase).
    """
    org_ent = next((e["text"] for e in entities if e["label"] in {"ORG", "GPE", "WORK_OF_ART", "LAW"}), None)
    date_ent = next((e["text"] for e in entities if e["label"] in {"DATE", "TIME"}), None)

    if org_ent:
        return f"Spend 2 minutes outlining your main objective for {org_ent} on a piece of paper."
    elif date_ent:
        return f"Write down one priority task to complete by {date_ent}."

    actions = {
        "Health, Wellness & Lifestyle": "Drink one full glass of water and do a 2-minute stretch.",
        "Financial Goals & Wealth": "Log into your account to check your asset allocation or track a single expense.",
        "Education & Academic Milestones": "Open your study notes and review a single key concept right now.",
        "Legal & Administrative Resolution": "Organize your essential documents into a single dedicated folder.",
    }
    return actions.get(category, "Put your phone away and sit in silence for exactly 2 minutes to reset your focus.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║        TEXTRANK SENTENCE SUMMARIZATION & INTENT ISOLATION                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _extract_sentences_textrank(
    doc: spacy.tokens.Doc,
) -> tuple[spacy.tokens.Span | spacy.tokens.Doc, list[spacy.tokens.Span]]:
    """
    Extract single most mathematically important sentence as Goal Sentence via PyTextRank.
    Treat remaining sentences as Venting Sentences.
    """
    sentences = list(doc.sents)
    if not sentences:
        return doc, []
    if len(sentences) == 1:
        return sentences[0], []

    summary_sentences = list(doc._.textrank.summary(limit_sentences=1))
    if summary_sentences:
        goal_sent = summary_sentences[0]
        venting_sents = [s for s in sentences if s.text.strip() != goal_sent.text.strip()]
        return goal_sent, venting_sents

    goal_sent = sentences[-1]
    venting_sents = sentences[:-1]
    return goal_sent, venting_sents


def _detect_patterns(
    raw_tokens: list[str], lemmas: list[str], vader_compound: float = 0.0
) -> dict[str, list[str]]:
    """
    Phase 4 (Pragmatic): Context-Aware CBT Pattern Detection.
    ONLY flags Absolutist thinking ('always', 'never') if sentiment compound score < -0.1.
    """
    search_set = {t.lower() for t in raw_tokens} | set(lemmas)
    patterns = {}
    for name, triggers in PROBLEMATIC_PATTERNS.items():
        matched = [w for w in triggers if w in search_set]
        if matched:
            if name == "Absolutist" and vader_compound >= -0.1:
                continue
            patterns[name] = matched
    return patterns


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  CLINICAL CBT COGNITIVE REAPPRAISAL ENGINE               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

REAPPRAISAL_TEMPLATES = {
    "Catastrophizing": {
        "validation": "It is completely normal to feel overwhelmed when thinking about the worst-case scenario regarding {noun}.",
        "reframe": "However, just because my brain is projecting fear does not mean it is a guaranteed reality.",
        "efficacy": "I am fully capable of handling whatever actually happens, one step at a time."
    },
    "All-or-Nothing": {
        "validation": "It is easy to feel like things are entirely ruined when dealing with {noun}.",
        "reframe": "In reality, life operates on a spectrum. One setback does not erase all of my previous progress.",
        "efficacy": "I choose to focus on the nuanced middle ground, where true growth happens."
    },
    "Personalization": {
        "validation": "I am carrying a heavy emotional burden by taking all the blame for {noun}.",
        "reframe": "I must recognize that many external factors outside of my control contributed to this situation.",
        "efficacy": "I release the responsibility that isn't mine, and focus only on what I can control today."
    },
    "Fortune Telling": {
        "validation": "My anxiety is trying to protect me by predicting a negative outcome for my {noun}.",
        "reframe": "The truth is, the future is unwritten. My past failures do not dictate my future results.",
        "efficacy": "I choose to stay grounded in the present moment, where my actual power lives."
    },
    "Absolutist": {
        "validation": "It is easy to feel like things are entirely black-and-white when dealing with {noun}.",
        "reframe": "In reality, life operates on a spectrum. One setback does not erase all of my previous progress.",
        "efficacy": "I choose to focus on the nuanced middle ground, where true growth happens."
    },
    "Negative Framing": {
        "validation": "It is completely understandable to focus on obstacles when struggling with {noun}.",
        "reframe": "However, highlighting past failures hides the strength and lessons I have gained.",
        "efficacy": "I protect my focus and channel my energy purposefully into solutions."
    },
    "Rigid/Modal": {
        "validation": "Setting strict expectations around {noun} can feel necessary, but creates immense pressure.",
        "reframe": "Holding plans lightly allows me to adapt without losing my momentum.",
        "efficacy": "I adapt with grace and keep moving forward with confidence."
    }
}


def apply_cognitive_reappraisal(
    intent_noun: str,
    distortion_type: str,
    sentiment_score: float = 0.0
) -> dict[str, str]:
    """
    Clinical CBT Cognitive Reappraisal Engine.
    Structures output into Validation -> Reappraisal -> Self-Efficacy format.
    Triggered only when cognitive distortions are present.
    """
    clean_noun = _clean_noun_chunk(intent_noun) if intent_noun else "my goals"

    key_map = {
        "Absolutist": "All-or-Nothing",
        "Rumination": "Fortune Telling",
    }
    target_key = key_map.get(distortion_type, distortion_type)
    template = REAPPRAISAL_TEMPLATES.get(target_key, REAPPRAISAL_TEMPLATES["All-or-Nothing"])

    val = template["validation"].format(noun=clean_noun)
    ref = template["reframe"].format(noun=clean_noun)
    eff = template["efficacy"].format(noun=clean_noun)

    full_text = f"{val} {ref} {eff}"

    return {
        "distortion_type": distortion_type,
        "validation": val,
        "reframe": ref,
        "efficacy": eff,
        "full_text": full_text
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         PUBLIC API                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def process_text(text: str) -> dict:
    """
    Run full 5-stage Computational Linguistics Pipeline with Scikit-Learn ML
    36-Category Manifestation Intent Classifier.
    """
    # ── Phase 1: Lexical (Spell Correction, Tokenization & Lemmatization) ───
    corrected_text, spelling_corrections = _correct_spelling(text)

    doc = nlp(corrected_text)

    raw_tokens = [t.text for t in doc if not t.is_space]
    all_lemmas = [
        t.lemma_.lower() for t in doc
        if not t.is_stop and not t.is_punct and not t.is_space and t.lemma_.strip()
    ]

    # ── Phase 2: Syntactic (S-V-O & Locus of Control) ───────────────────────
    locus_of_control = _analyze_locus_of_control(doc)

    # ── Phase 3: Semantic (PyTextRank, Scikit-Learn ML Classifier & NER) ────
    goal_sent, venting_sents = _extract_sentences_textrank(doc)
    extracted_entities = _extract_entities(doc)

    # Predict multi-label intents using SentenceTransformer embeddings + Calibrated MultiOutputClassifier
    detected_categories, top_intents_with_probabilities, detected_intents_debug = _predict_multilabel_intents(goal_sent.text.strip())
    primary_category = detected_categories[0] if detected_categories else "Default"
    intent_verb, intent_noun = _extract_intent(goal_sent)

    pos_tags = [
        {"word": t.text, "lemma": t.lemma_.lower(), "pos": t.pos_}
        for t in goal_sent if not t.is_space
    ]

    # ── Phase 4: Pragmatic (Temporal Focus, VADER & Patterns) ────────────────
    vader_raw = sia.polarity_scores(doc.text)
    overall_compound = vader_raw.get("compound", 0.0)

    if venting_sents:
        venting_text = " ".join(s.text for s in venting_sents)
        vader_raw = sia.polarity_scores(venting_text)
        venting_compound = vader_raw.get("compound", 0.0)
    else:
        venting_compound = overall_compound

    # Context-Aware CBT: Pass VADER compound score into pattern detector
    flags = _detect_patterns(raw_tokens, all_lemmas, vader_compound=overall_compound)
    temporal_focus, flags = _analyze_temporal_focus(doc, flags)

    # ── Phase 5: NLG (Modular Affirmation Assembly & Entity Injection) ───────
    affirmation = generate_affirmation(
        intent_verb=intent_verb,
        intent_noun=intent_noun,
        category=detected_categories,
        flags=flags,
        vader_compound=overall_compound,
        doc=goal_sent,
    )

    even_if = generate_even_if(
        category=primary_category,
        flags=flags,
        venting_compound=venting_compound,
        temporal_focus=temporal_focus,
    )

    combined_affirmation = f"{affirmation} {even_if}"

    if_then = generate_if_then(category=primary_category, entities=extracted_entities)
    micro_action = generate_micro_action(category=primary_category, entities=extracted_entities)

    # Clinical CBT Cognitive Reappraisal Engine integration
    reappraisal_payload = None
    if flags:
        primary_distortion = list(flags.keys())[0]
        reappraisal_payload = apply_cognitive_reappraisal(
            intent_noun=intent_noun,
            distortion_type=primary_distortion,
            sentiment_score=overall_compound
        )

    # ── Telemetry Payload (5 Formal Explicit NLP Phases with ML Output) ──────
    nlp_debug = {
        "Phase_1_Lexical_Analysis": {
            "description": "Tokenization, spelling correction, and lemmatization (root words).",
            "spelling_corrections": spelling_corrections,
            "raw_tokens": raw_tokens,
            "lemmas": all_lemmas,
        },
        "Phase_2_Syntactic_Analysis": {
            "description": "Grammar checking, POS tagging, and dependency parsing.",
            "pos_tags": pos_tags,
            "svo_triplets": {
                "intent_verb": intent_verb,
                "intent_noun": intent_noun,
                "locus_of_control": locus_of_control,
            },
        },
        "Phase_3_Semantic_Analysis": {
            "description": "Extracting literal meaning, categorizing via Scikit-Learn Multi-Label TF-IDF ML model, and NER.",
            "ml_predicted_intent": primary_category,
            "detected_intents": detected_intents_debug,
            "top_intents_with_probabilities": top_intents_with_probabilities,
            "named_entities": extracted_entities,
        },
        "Phase_4_Discourse_Integration": {
            "description": "Sentence relations, TextRank sentence isolation, and VADER sentiment analysis.",
            "goal_sentence": goal_sent.text.strip(),
            "venting_sentences": [s.text.strip() for s in venting_sents],
            "vader_raw": vader_raw,
        },
        "Phase_5_Pragmatic_Analysis": {
            "description": "Contextual intent, temporal focus (rumination vs anticipatory anxiety), and cognitive pattern flags.",
            "temporal_focus": temporal_focus,
            "cognitive_flags": flags,
            "cognitive_reappraisal": reappraisal_payload,
        },
    }

    response = {
        "original_goal": text,
        "generated_affirmation": combined_affirmation,
        "implementation_intention": if_then,
        "micro_action": micro_action,
        "nlp_debug": nlp_debug,
    }

    if reappraisal_payload:
        response["cognitive_reappraisal"] = reappraisal_payload

    return response
