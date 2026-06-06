# --- Scene Prompt Builder ---
from .emotion_scene_elements import EMOTION_SCENE_ELEMENTS, COMPLEX_WORD_TO_PRIMARY_EMOTION, get_scene_elements_for_word, get_details_for_complex_word

def pastelize_hex(hex_color: str) -> str:
    """Convert a hex color to a pastel version (simple average with white)."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # Blend with white (simple average)
    r = int((r + 255) / 2)
    g = int((g + 255) / 2)
    b = int((b + 255) / 2)
    return f"#{r:02x}{g:02x}{b:02x}"

def build_scene_prompt(subject_desc: str, palette: list[str], emotion_words: list[str], scene_elements: dict = None, complex_details: dict = None) -> str:
    """
    Build the full scene prompt for the image generator.
    - subject_desc: description of the subject (from photo upload)
    - palette: list of top 5 hex color codes from the photo
    - emotion_words: list of detected complex emotion words
    - scene_elements: dict of scene elements (optional, can be built from emotion_words)
    - complex_details: dict of complex word details (optional)
    """
    import random
    # Pastelize palette and assign to scene regions
    pastel_palette = [pastelize_hex(c) for c in palette[:5]]
    palette_names = [f"#{c.lstrip('#').upper()}" for c in palette[:5]]
    region_names = ["background", "foreground", "left", "right", "accent"]

    # Scene lighting, palette, and composition (concise, literal)
    lighting = "Soft natural daylight, gentle pastel colors, subtle vintage textures, calm and slightly nostalgic mood. Layered composition with overlapping forms and a sense of depth."
    subject = f"The subject is shown exactly as in the uploaded photo: {subject_desc}."

    # Fallback objects for emotional symbolism if none are found
    fallback_objects = [
        "mask", "mirror", "hourglass", "veil", "laurel wreath", "musical instrument", "book", "candle", "key", "locket", "scroll", "quill", "lantern", "ribbon", "coin", "feather", "rose", "handkerchief", "gemstone", "chess piece", "letter", "ring", "shell", "star", "apple", "cup", "bottle"
    ]

    # Material and placement options (shuffled for each call)
    material_options = ["polished ceramic", "brushed metal", "smooth glass", "natural wood", "soft fabric", "sculpted stone", "painted clay", "etched crystal"]
    placement_options = [
        "on a small pedestal at the center foreground",
        "on a low table to the left",
        "on a shelf to the right",
        "hanging from a thin wire in the background",
        "resting on a folded cloth near the subject",
        "leaning against a textured wall",
        "partially hidden behind a translucent screen",
        "arranged in a cluster with other objects"
    ]
    random.shuffle(material_options)
    random.shuffle(placement_options)
    random.shuffle(fallback_objects)

    # --- Prompt Outline ---
    prompt_outline = {
        "lighting": lighting,
        "subject": subject,
        "colors": [],
        "scene_items": [],
        "motifs": [],
    }

    # 1. Colors section: map each palette color to an emotion
    for idx, word in enumerate(emotion_words):
        color = pastel_palette[idx % len(pastel_palette)] if pastel_palette else "#cccccc"
        color_name = palette_names[idx % len(palette_names)] if palette_names else color
        prompt_outline["colors"].append(f"Emotion '{word}' is represented by color {color_name} ({color}).")

    # 2. Scene items and motifs: explicit, colored, placed, and detailed
    used_symbols = set()
    fallback_idx = 0
    for idx, word in enumerate(emotion_words):
        details = get_details_for_complex_word(word)
        color = pastel_palette[idx % len(pastel_palette)] if pastel_palette else "#cccccc"
        color_name = palette_names[idx % len(palette_names)] if palette_names else color
        found = False
        # Scene items: unique, visually specific, colored, and placed
        if details:
            # Randomly pick a symbol, motif, and art for variety
            symbols = [s for s in details.get("symbols", []) if s not in used_symbols and s not in ["mountains", "sky", "boulder", "crystal", "field", "landscape", "clouds", "sunrise", "sunset", "forest", "pond", "river", "sea", "ocean", "mountain", "valley", "skyline", "horizon"]]
            if symbols:
                symbol = random.choice(symbols)
                material = material_options[idx % len(material_options)]
                placement = placement_options[idx % len(placement_options)]
                prompt_outline["scene_items"].append(
                    f"A {symbol} made of {material}, colored {color_name} ({color}), is {placement}, symbolizing {word}. The object is rendered with sharp edges, clear texture, and realistic lighting.")
                used_symbols.add(symbol)
                found = True
            # Motifs: explicit, colored, and placed
            tropes = [t for t in details.get("tropes", []) if not any(x in t.lower() for x in ["mountain", "sky", "landscape", "valley", "horizon", "field", "sea", "ocean", "forest"])]
            if tropes:
                trope = random.choice(tropes)
                prompt_outline["motifs"].append(
                    f"Theatrical motif: '{trope}' is staged as a prop or gesture, colored {color_name} ({color}), placed to the left or right of the subject, with visible material and clear placement.")
                found = True
            # Art style: as a visual treatment in the background
            arts = details.get("art", [])
            if arts:
                art = random.choice(arts)
                prompt_outline["motifs"].append(
                    f"Background visual style: {art}, applied with visible brushwork or pattern, matching the palette and lighting.")
                found = True
        if not found:
            fallback_obj = fallback_objects[fallback_idx % len(fallback_objects)]
            material = material_options[idx % len(material_options)]
            placement = placement_options[idx % len(placement_options)]
            prompt_outline["scene_items"].append(
                f"A {fallback_obj} made of glass, colored {color_name} ({color}), is {placement}, to represent {word}. The object is rendered with sharp detail, clear material, and realistic placement.")
            fallback_idx += 1

    # Compose final prompt: literal, itemized, visually specific, and explicit
    prompt = (
        prompt_outline["lighting"] + "\n"
        + prompt_outline["subject"] + "\n"
        + "\n".join(prompt_outline["colors"]) + "\n"
        + "\n".join(prompt_outline["scene_items"]) + "\n"
        + "\n".join(prompt_outline["motifs"]) + "\n"
    )
    return prompt
import time
import json
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

import logging
import io
import hashlib
import requests
from typing import Optional
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import os
from pydantic import BaseModel
from typing import Any, Literal
from functools import lru_cache
"""from sentence_transformers import SentenceTransformer"""
import numpy as np
import cv2
from sklearn.cluster import KMeans
detected_clothing_palette = None
import re
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random

app = FastAPI(title="Feeling AI API", version="0.1.0")
logger = logging.getLogger(__name__)

FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_local_env_file(env_path: Optional[Path] = None) -> None:
    path = env_path or Path(__file__).with_name(".env")
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env_file()
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "").strip()
STYLE_REFERENCES_DIR = Path(__file__).resolve().parent.parent / "style-references"
# Disable background-reference conditioning to prioritize prompt adherence.
USE_BACKGROUND_REFERENCE_IMAGE = False


class TextRequest(BaseModel):
    text: str


class GenerateEmotionImageRequest(BaseModel):
    text_analysis: dict[str, Any]
    palette: Optional[list[str]] = None
    clothing_palette: Optional[list[str]] = None
    full_image_palette: Optional[list[str]] = None
    palette_mode: Literal["clothing", "full"] = "clothing"
    clothing_style_profile: Optional[dict[str, Any]] = None
    style_notes: Optional[str] = None


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "from",
    "by",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "that",
    "this",
    "as",
    "i",
    "you",
    "he",
    "she",
    "they",
    "we",
    "my",
    "your",
    "our",
    "their",
    "me",
    "him",
    "her",
    "them",
    "so",
    "not",
    "too",
    "very",
}

# Map split compounds to their merged spoken form for better matching.
COMPOUND_JOIN_MAP = {
    ("any", "more"): "anymore",
}

EMOTION_LEXICON = {
    "joy": {
        "joy",
        "happy",
        "happiness",
        "giddy",
        "joyous",
        "pleased",
        "pleasant",
        "cute",
        "kind",
        "grateful",
        "forgiving",
        "ready",
        "smile",
        "laugh",
        "glow",
        "dance",
    },
    "sadness": {
        "sad",
        "sadness",
        "melancholy",
        "lost",
        "cry",
        "crying",
        "tear",
        "tears",
        "crushed",
        "lonely",
        "forgotten",
        "nothingness",
        "helplessness",
        "weak",
        "poor",
        "shaken",
        "ache",
        "grief",
        "loss",
    },
    "melancholy": {"melancholy", "somber", "wistful", "bleak", "empty", "grey"},
    "anger": {"anger", "rage", "fury", "impending", "encrouching", "burn", "fight"},
    "calm": {"calm", "still", "quiet", "controlled", "aware", "alert", "clean"},
    "fear": {
        "fear",
        "afraid",
        "disorented",
        "disoriented",
        "confused",
        "helplessness",
        "impending",
        "unknown",
    },
    "shame": {"shameful", "foolish", "fake", "filthy", "dumb", "imperfect"},
    "confidence": {"brave", "powerful", "certain", "important", "deserving", "real", "egotistical"},
    "uncertainty": {"indecisive", "impressionable", "pointless", "distant", "detached"},
    "boredom": {"bored", "boredom", "tedious", "monotonous", "dull", "numb"},
    "detachment": {"detached", "distant", "disembodied", "numb", "unabiding"},
    "awe": {"fabled", "faited", "destined", "grandure", "sacred", "wonder"},
    "belonging": {"togetherness", "together", "parental", "leadership", "clingy"},
    "order": {"organized", "orgnized", "unorganized", "unorginized", "simple", "complicated"},
    "presence": {"present", "aware", "alert", "different", "fabled", "faited", "destined"},
    "value": {"precious", "famous", "perfect", "unimportant", "grandure", "plaine"},
    "instability": {"flounder", "disembodied", "feverish", "grey", "unabiding", "retorical"},
    "growth": {"constructive", "emboldened", "emoldened", "forgiving", "ready", "grateful"},
}

EMOTION_THEME_LABELS = {
    "joy": "joy",
    "sadness": "grief",
    "melancholy": "melancholy",
    "anger": "rage",
    "calm": "stillness",
    "fear": "dread",
    "shame": "shame",
    "confidence": "power",
    "uncertainty": "uncertainty",
    "boredom": "boredom",
    "detachment": "detachment",
    "awe": "awe",
    "belonging": "belonging",
    "order": "order",
    "presence": "presence",
    "value": "value",
    "instability": "instability",
    "growth": "growth",
    # User-provided feeling aliases
    "happiness": "joy",
    "constructive": "growth",
    "disorented": "dread",
    "forgotten": "grief",
    "shameful": "shame",
    "present": "presence",
    "kind": "joy",
    "cute": "joy",
    "brave": "power",
    "precious": "value",
    "flounder": "instability",
    "clingy": "belonging",
    "detached": "detachment",
    "powerful": "power",
    "weak": "grief",
    "giddy": "joy",
    "joyous": "joy",
    "crushed": "grief",
    "disembodied": "instability",
    "retorical": "instability",
    "pointless": "uncertainty",
    "famous": "value",
    "clean": "stillness",
    "pleased": "joy",
    "unorginized": "order",
    "orgnized": "order",
    "confused": "dread",
    "certain": "power",
    "important": "power",
    "distant": "detachment",
    "foolish": "shame",
    "poor": "grief",
    "simple": "order",
    "complicated": "order",
    "unimportant": "value",
    "impressionable": "uncertainty",
    "perfect": "value",
    "imperfect": "shame",
    "indecisive": "uncertainty",
    "unabiding": "instability",
    "dumb": "shame",
    "parental": "belonging",
    "leadership": "belonging",
    "filthy": "shame",
    "pleasant": "joy",
    "encrouching": "rage",
    "impending": "dread",
    "fake": "shame",
    "real": "power",
    "different": "presence",
    "lonely": "grief",
    "togetherness": "belonging",
    "grateful": "growth",
    "forgiving": "growth",
    "ready": "growth",
    "emoldened": "growth",
    "shaken": "grief",
    "alert": "presence",
    "aware": "presence",
    "fabled": "awe",
    "faited": "awe",
    "destined": "awe",
    "egotistical": "power",
    "nothingness": "grief",
    "helplessness": "grief",
    "grandure": "awe",
    "plaine": "value",
    "feverish": "instability",
    "grey": "melancholy",
    "controlled": "stillness",
    "deserving": "power",
    "lost": "detachment",
    "crying": "grief",
    "tears": "grief",
    "youth": "melancholy",
}

THEME_NOISE_WORDS = STOPWORDS | {
    "how",
    "what",
    "when",
    "where",
    "why",
    "which",
    "who",
    "whom",
    "just",
    "really",
    "very",
    "always",
    "every",
    "everything",
    "everyone",
    "everybody",
    "something",
    "someone",
    "thing",
    "things",
    "want",
    "wants",
    "need",
    "needs",
    "get",
    "gets",
    "got",
    "make",
    "makes",
    "made",
    "take",
    "takes",
    "taken",
    "say",
    "says",
    "said",
    "tell",
    "tells",
    "told",
    "know",
    "knows",
    "knew",
    "think",
    "thinks",
    "thinking",
    "feel",
    "feels",
    "feeling",
    "like",
    "just",
    "keep",
    "keeps",
    "kept",
    "go",
    "goes",
    "going",
    "gone",
    "come",
    "comes",
    "came",
    "see",
    "sees",
    "seen",
    "short",
    "long",
    "something",
    "everything",
    "anything",
    "nothing",
    "somebody",
    "nobody",
}

PAIRING_WEIGHTS = {
    "exact": 1.0,
    "similar": 0.8,
    "phonetic": 0.7,
    "rhyme": 0.6,
    "contextual": 0.9,
}

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

THEME_ANCHORS = {
    "grief": "loss mourning absence farewell sorrow",
    "mortality": "death dying finitude end of life",
    "suffering": "pain anguish torment suffering misery",
    "desire": "wanting longing craving yearning",
    "isolation": "alone apart distance loneliness separation",
    "devotion": "loyalty care attachment closeness keeping",
    "inheritance": "family ancestry lineage descendants",
    "identity": "self mirror face voice becoming",
    "transcendence": "god magic sacred divinity revelation",
    "conflict": "fight struggle violence tension breakage",
    "healing": "repair rest peace recovery softness",
    "hope": "future possibility light promise renewal",
    "shame": "guilt shame disgrace secrecy",
    "belonging": "together home community inclusion",
    "freedom": "release liberation escape open space",
    "memory": "remember echo past recall persistence",
    "creation": "build make world story invention",
    "love": "love tenderness affection intimacy",
    "silence": "quiet hush stillness absence of speech",
    "sacrifice": "giving up offering surrender cost",
    "fear": "dread anxiety threat unknown",
    "responsibility": "duty burden obligation care",
    "meaning": "purpose truth significance understanding",
    "change": "becoming transformation shift transition",
}

THEME_TO_EMOTION_HINTS = {
    "grief": {"sadness": 1.2, "melancholy": 0.6, "shame": 0.3},
    "mortality": {"sadness": 0.9, "fear": 0.8},
    "suffering": {"sadness": 1.1, "fear": 0.5},
    "desire": {"joy": 0.7, "uncertainty": 0.5},
    "isolation": {"sadness": 0.9, "detachment": 0.8, "boredom": 0.4, "fear": 0.4},
    "devotion": {"belonging": 0.9, "joy": 0.5},
    "inheritance": {"belonging": 0.8, "presence": 0.4},
    "identity": {"presence": 0.8, "uncertainty": 0.4, "detachment": 0.3},
    "transcendence": {"growth": 0.7, "awe": 0.9, "joy": 0.4},
    "conflict": {"anger": 1.1, "fear": 0.5},
    "healing": {"calm": 0.9, "growth": 0.7},
    "hope": {"joy": 0.9, "growth": 0.8},
    "shame": {"shame": 1.4, "sadness": 0.5},
    "belonging": {"belonging": 1.0, "joy": 0.6},
    "freedom": {"confidence": 0.9, "growth": 0.5},
    "memory": {"sadness": 0.4, "melancholy": 0.7, "presence": 0.7},
    "creation": {"growth": 0.9, "confidence": 0.4},
    "love": {"joy": 1.2, "belonging": 0.7},
    "silence": {"calm": 0.6, "boredom": 0.6, "melancholy": 0.5, "sadness": 0.3},
    "sacrifice": {"sadness": 0.6, "growth": 0.6},
    "fear": {"fear": 1.3},
    "responsibility": {"confidence": 0.5, "uncertainty": 0.5},
    "meaning": {"presence": 0.8, "growth": 0.6, "awe": 0.4},
    "change": {"uncertainty": 0.6, "growth": 0.7},
}


SUBJECT_MATTER_FEELING_PATTERNS: dict[str, dict[str, Any]] = {
    "understanding": {
        "phrases": ["understand", "understanding", "help me understand"],
        "emotions": ["presence", "growth"],
    },
    "indifference": {
        "phrases": ["indifference", "without opinion", "unconscious of feeling"],
        "emotions": ["detachment", "boredom"],
    },
    "opinion": {
        "phrases": ["opinion", "without opinion"],
        "emotions": ["presence", "uncertainty"],
    },
    "truth": {
        "phrases": ["truth", "fact", "fact or a lie"],
        "emotions": ["presence", "uncertainty"],
    },
    "knowingness": {
        "phrases": ["knowing", "knowing nothing", "know"],
        "emotions": ["presence", "uncertainty"],
    },
    "self-deception": {
        "phrases": ["pretend", "faking", "fake", "lie", "lying to"],
        "emotions": ["shame", "uncertainty", "detachment"],
    },
    "distraction": {
        "phrases": ["distraction", "another distraction"],
        "emotions": ["boredom", "detachment"],
    },
    "meaninglessness": {
        "phrases": ["nothing worth", "worth dying on", "worth something"],
        "emotions": ["uncertainty", "melancholy", "boredom"],
    },
    "gratitude": {
        "phrases": ["gratitude", "grateful"],
        "emotions": ["growth", "joy", "value"],
    },
    "ingratitude": {
        "phrases": ["ungrateful", "ungratitude", "not grateful"],
        "emotions": ["detachment", "instability", "shame"],
    },
    "wisdom-burden": {
        "phrases": ["wisdom", "means nothing", "entire lifetime"],
        "emotions": ["presence", "melancholy", "instability"],
    },
    "mortality devotion": {
        "phrases": ["past lives", "marked by death", "death in", "living for nothing other than death"],
        "emotions": ["melancholy", "awe", "presence"],
    },
    "sacrificial heroism": {
        "phrases": ["heroism", "death for one another", "one life for all", "all for one"],
        "emotions": ["growth", "belonging", "awe"],
    },
    "sacred faith": {
        "phrases": ["faith", "higher power", "comes naturally"],
        "emotions": ["awe", "growth", "presence"],
    },
    "collective love": {
        "phrases": ["love for one another", "one for all", "all for one"],
        "emotions": ["belonging", "joy", "growth"],
    },
    "purpose struggle": {
        "phrases": ["lack of purpose", "give life purpose", "purpose"],
        "emotions": ["uncertainty", "growth", "presence"],
    },
    "forgiveness ethic": {
        "phrases": ["forgive life", "forgive", "one another"],
        "emotions": ["growth", "joy", "belonging"],
    },
    "directionless change": {
        "phrases": ["i am lost", "not even close", "one more step", "walking"],
        "emotions": ["uncertainty", "detachment", "melancholy"],
    },
    "social invisibility": {
        "phrases": ["no one sees me", "no one cares", "nobody but you"],
        "emotions": ["sadness", "detachment", "shame"],
    },
    "private grief": {
        "phrases": ["why am i crying", "these tears", "tears are for"],
        "emotions": ["sadness", "melancholy", "presence"],
    },
    "ironic defense": {
        "phrases": ["what a joke", "cannot be serious", "sensitive humour", "drama"],
        "emotions": ["instability", "shame", "detachment"],
    },
    "youth mourning": {
        "phrases": ["where did my youth go", "much longer", "youth"],
        "emotions": ["melancholy", "sadness", "fear"],
    },
    "change desire": {
        "phrases": ["life is only change", "has to change", "something has to happen"],
        "emotions": ["growth", "uncertainty", "instability"],
    },
    "meaning search": {
        "phrases": ["that means something", "looking for", "gives you an idea"],
        "emotions": ["presence", "growth", "uncertainty"],
    },
    "paradox tension": {
        "phrases": ["everything right", "so wrong", "not surprised"],
        "emotions": ["instability", "uncertainty", "detachment"],
    },
    "fated arrival": {
        "phrases": ["for you to be here", "didn't happen for you to be here", "for you to be 'here'"],
        "emotions": ["awe", "presence", "growth"],
    },
    "unashamed impulse": {
        "phrases": ["not ashamed", "idea"],
        "emotions": ["confidence", "growth", "presence"],
    },
    "existential disbelief": {
        "phrases": ["inconsolable", "won't believe", "anything true"],
        "emotions": ["melancholy", "uncertainty", "instability"],
    },
    "sacred gratitude": {
        "phrases": ["gratitude as our religion", "gratitude", "religion"],
        "emotions": ["awe", "growth", "value"],
    },
    "cosmic smallness": {
        "phrases": ["next to a giant", "we are nothing", "giant"],
        "emotions": ["awe", "melancholy", "presence"],
    },
    "future pull": {
        "phrases": ["point to the future", "one year closer", "future"],
        "emotions": ["uncertainty", "growth", "presence"],
    },
    "meaning imperative": {
        "phrases": ["make it matter", "means something", "matter"],
        "emotions": ["presence", "growth", "value"],
    },
    "bereavement spiral": {
        "phrases": ["miss you", "see you go", "crying to myself", "tears are for"],
        "emotions": ["sadness", "melancholy", "detachment"],
    },
    "mortality fixation": {
        "phrases": ["short life", "thinking about death", "kill myself", "torn limb"],
        "emotions": ["fear", "sadness", "instability"],
    },
    "surrender paradox": {
        "phrases": ["let go of everything", "have it all", "don't want it"],
        "emotions": ["uncertainty", "growth", "presence"],
    },
    "ancestral mysticism": {
        "phrases": ["dead relatives", "talking to god", "family tree", "magic"],
        "emotions": ["awe", "melancholy", "presence"],
    },
    "dream residue": {
        "phrases": ["used to", "chasing a dream", "dancing like"],
        "emotions": ["melancholy", "growth", "presence"],
    },
    "ancestral continuity": {
        "phrases": ["with everyone before me", "only everyone before me", "before me"],
        "emotions": ["presence", "belonging", "melancholy"],
    },
    "regret paradox": {
        "phrases": ["grateful to regret", "never regret", "regret"],
        "emotions": ["instability", "growth", "uncertainty"],
    },
    "worldbuilding mandate": {
        "phrases": ["build a world", "world to be built over again", "make a new rule"],
        "emotions": ["growth", "presence", "confidence"],
    },
    "sacred humanism": {
        "phrases": ["human condition", "we're gods", "magical", "other than life"],
        "emotions": ["awe", "presence", "growth"],
    },
    "custodial burden": {
        "phrases": ["in your hands", "get everyone killed", "you need to stop thinking"],
        "emotions": ["fear", "confidence", "instability"],
    },
    "waiting": {
        "phrases": ["until", "not yet", "too old", "forever"],
        "emotions": ["uncertainty", "boredom", "detachment"],
    },
    "time-anxiety": {
        "phrases": ["night and day", "too old to", "forever", "difference is night and day"],
        "emotions": ["fear", "uncertainty", "melancholy"],
    },
}


NEGATING_PREFIXES = ("un", "dis", "non", "anti")
ABSTRACT_FEELING_SUFFIXES = ("tion", "sion", "ness", "ity", "ship", "dom", "tude")
CONNECTOR_TO_CONCEPT = {
    "until": "waiting",
    "till": "waiting",
    "while": "waiting",
    "when": "anticipation",
    "before": "anticipation",
    "after": "aftermath",
    "but": "inner-conflict",
    "however": "inner-conflict",
    "yet": "inner-conflict",
    "though": "inner-conflict",
    "rather": "avoidance",
    "instead": "avoidance",
    "if": "uncertainty",
    "unless": "uncertainty",
    "because": "justification",
    "therefore": "consequence",
    "so": "consequence",
}
CONNECTOR_CONCEPT_EMOTIONS = {
    "waiting": ["uncertainty", "boredom", "detachment"],
    "time-anxiety": ["fear", "uncertainty", "melancholy"],
    "anticipation": ["uncertainty", "growth"],
    "aftermath": ["melancholy", "presence"],
    "inner-conflict": ["instability", "uncertainty", "shame"],
    "avoidance": ["detachment", "uncertainty"],
    "justification": ["uncertainty", "presence"],
    "consequence": ["fear", "presence"],
}
TEMPORAL_PRESSURE_WORDS = {
    "old",
    "forever",
    "lifetime",
    "time",
    "day",
    "night",
    "late",
    "aging",
    "age",
}
GENERIC_ADAPTIVE_NOISE = {
    "entire",
    "under",
    "learn",
    "life",
    "acting",
    "speak",
    "took",
    "come",
    "stop",
    "mess",
    "other",
    "wise",
    "until",
    "forever",
    "everywhere",
    "anywhere",
    "nobody",
    "really",
    "real",
    "explanation",
    "condition",
    "mean",
    "means",
    "said",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _normalize_token(token: str) -> str:
    token = token.lower().strip("' ")

    # Preserve abstract pronouns/nouns ending with 'thing' for concept detection.
    if token.endswith("thing"):
        return token

    suffixes = ("ingly", "edly", "ingly", "ing", "edly", "ed", "ly", "es")
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]

    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        return token[:-1]

    return token


def _soundex(word: str) -> str:
    if not word:
        return ""

    mapping = {
        "b": "1",
        "f": "1",
        "p": "1",
        "v": "1",
        "c": "2",
        "g": "2",
        "j": "2",
        "k": "2",
        "q": "2",
        "s": "2",
        "x": "2",
        "z": "2",
        "d": "3",
        "t": "3",
        "l": "4",
        "m": "5",
        "n": "5",
        "r": "6",
    }

    first = word[0].upper()
    tail_codes: list[str] = []
    previous = mapping.get(word[0], "")

    for ch in word[1:]:
        code = mapping.get(ch, "")
        if code and code != previous:
            tail_codes.append(code)
        previous = code

    return (first + "".join(tail_codes) + "000")[:4]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _sentence_word_set(sentence: str) -> list[str]:
    words = [_normalize_token(w) for w in _tokenize(sentence)]
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def _vowel_group_count(word: str) -> int:
    """Count groups of consecutive vowels."""
    normalized = word.lower()
    groups = re.findall(r"[aeiouy]+", normalized)
    return len(groups)


def _longest_common_subsequence_length(left: str, right: str) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(left), len(right)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if left[i - 1] == right[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _bigram_dice_similarity(left: str, right: str) -> float:
    """Compute Dice similarity between bigrams."""
    if len(left) < 2 or len(right) < 2:
        return 0.0
    grams_a = [left[i : i + 2] for i in range(len(left) - 1)]
    grams_b = [right[i : i + 2] for i in range(len(right) - 1)]
    
    count_a = Counter(grams_a)
    overlap = 0
    for gram in grams_b:
        if count_a[gram] > 0:
            overlap += 1
            count_a[gram] -= 1
    
    return (2 * overlap) / (len(grams_a) + len(grams_b)) if (len(grams_a) + len(grams_b)) > 0 else 0.0


def _has_adjacent_letter_overlap(left: str, right: str) -> bool:
    """True when words share at least one adjacent 2-letter sequence."""
    if len(left) < 2 or len(right) < 2:
        return False

    left_bigrams = {left[i : i + 2] for i in range(len(left) - 1)}
    right_bigrams = {right[i : i + 2] for i in range(len(right) - 1)}
    return bool(left_bigrams & right_bigrams)


IDEA_CLUSTERS = [
    {"song", "music", "dance", "dancing", "rhythm"},
    {"death", "dying", "dead", "mourning", "grief", "loss"},
    {"family", "tree", "historic", "ancestors", "relatives"},
    {"magic", "god", "sacred", "soul", "spirit"},
    {"world", "build", "built", "story", "create", "creation"},
    {"want", "desire", "wish", "longing", "crave"},
]


def _is_idea_related(left: str, right: str) -> bool:
    """Check if two words are in the same idea cluster."""
    for cluster in IDEA_CLUSTERS:
        if left in cluster and right in cluster:
            return True
    return False


def _sentence_keywords(words: list[str], top_n: int = 4) -> list[str]:
    counts = Counter(word for word in words if word not in THEME_NOISE_WORDS)
    return [word for word, _ in counts.most_common(top_n)]


def _sentence_emotion_hits(words: list[str]) -> dict[str, int]:
    counts = Counter(_normalize_token(word) for word in words)
    lookup = _emotion_lookup_map()
    emotion_counts: Counter[str] = Counter()

    for token, count in counts.items():
        for emotion in lookup.get(token, set()):
            emotion_counts[emotion] += count

    return {emotion: int(score) for emotion, score in emotion_counts.items() if score > 0}


def _sentence_devices(sentence: str, words: list[str]) -> list[str]:
    devices: list[str] = []
    if len(words) >= 4:
        initials = Counter(word[0] for word in words)
        if any(count >= 3 for count in initials.values()):
            devices.append("alliteration")
    lowered = sentence.lower()
    if re.search(r"\blike\s+a?n?\s+\w+", lowered) or re.search(
        r"\bas\s+\w+\s+as\s+\w+", lowered
    ):
        devices.append("simile")
    return devices


def _build_sentence_profiles(text: str) -> list[dict[str, Any]]:
    sentences = _split_sentences(text)
    profiles: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        words = _sentence_word_set(sentence)
        profiles.append(
            {
                "index": index,
                "sentence": sentence,
                "words": words,
                "keywords": _sentence_keywords(words),
                "emotion_hits": _sentence_emotion_hits(words),
                "devices": _sentence_devices(sentence, words),
            }
        )
    return profiles



# Disabled: Model loading for OOM debug
def _get_sentence_embedding_model():
    return None



# Disabled: Model loading for OOM debug
def _get_theme_anchor_embeddings():
    return None



# Disabled: Model loading for OOM debug
def _sentence_embedding_similarity(sentences: list[str]):
    return None


def _build_token_contexts(sentences: list[str]) -> dict[str, Counter[str]]:
    token_contexts: dict[str, Counter[str]] = {}
    sentence_words = [_sentence_word_set(sentence) for sentence in sentences]

    for index, words in enumerate(sentence_words):
        local_context = set(words)
        if index > 0:
            local_context.update(sentence_words[index - 1])
        if index + 1 < len(sentence_words):
            local_context.update(sentence_words[index + 1])

        for word in words:
            token_contexts.setdefault(word, Counter())
            related_words = [item for item in local_context if item != word]
            token_contexts[word].update(related_words)

    return token_contexts


def _contextual_similarity(
    left: str, right: str, token_contexts: dict[str, Counter[str]]
) -> float:
    left_context = set(token_contexts.get(left, {}).keys())
    right_context = set(token_contexts.get(right, {}).keys())
    if not left_context or not right_context:
        return 0.0

    overlap = left_context & right_context
    union = left_context | right_context
    if not overlap or not union:
        return 0.0

    return len(overlap) / len(union)


def _classify_pairing(
    left: str, right: str, token_contexts: dict[str, Counter[str]]
) -> Optional[str]:
    if left == right:
        return "exact"

    if len(left) >= 4 and len(right) >= 4:
        ratio = SequenceMatcher(a=left, b=right).ratio()
        if ratio >= 0.82:
            return "similar"

    if len(left) >= 3 and len(right) >= 3 and _soundex(left) == _soundex(right):
        return "phonetic"

    if len(left) >= 4 and len(right) >= 4 and left[-3:] == right[-3:]:
        return "rhyme"

    if _contextual_similarity(left, right, token_contexts) >= 0.18:
        return "contextual"

    return None


def _build_sentence_associations(text: str) -> list[dict[str, Any]]:
    sentences = _split_sentences(text)
    sentence_words = [_sentence_word_set(sentence) for sentence in sentences]
    token_contexts = _build_token_contexts(sentences)

    associations: list[dict[str, Any]] = []
    for i, j in combinations(range(len(sentences)), 2):
        pairings: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for left in sentence_words[i]:
            for right in sentence_words[j]:
                relation = _classify_pairing(left, right, token_contexts)
                if not relation:
                    continue

                key = tuple(sorted((left, right)))
                if key in seen:
                    continue
                seen.add(key)

                pairings.append(
                    {
                        "left": left,
                        "right": right,
                        "relation": relation,
                        "weight": PAIRING_WEIGHTS[relation],
                    }
                )

        if pairings:
            pairings = sorted(pairings, key=lambda item: item["weight"], reverse=True)[:12]
            edge_strength = round(sum(item["weight"] for item in pairings), 3)
            associations.append(
                {
                    "sentence_a_index": i,
                    "sentence_b_index": j,
                    "sentence_a": sentences[i],
                    "sentence_b": sentences[j],
                    "edge_strength": edge_strength,
                    "pairings": pairings,
                }
            )

    return associations


def _build_semantic_sentence_links(
    sentence_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sentences = [profile["sentence"] for profile in sentence_profiles]
    if len(sentences) < 2:
        return []

    token_contexts = _build_token_contexts(sentences)
    embedding_similarities = _sentence_embedding_similarity(sentences)

    word_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    word_matrix = word_vectorizer.fit_transform(sentences)
    word_similarities = cosine_similarity(word_matrix)

    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    char_matrix = char_vectorizer.fit_transform(sentences)
    char_similarities = cosine_similarity(char_matrix)

    links: list[dict[str, Any]] = []
    for i, j in combinations(range(len(sentences)), 2):
        profile_a = sentence_profiles[i]
        profile_b = sentence_profiles[j]

        keyword_overlap = sorted(
            set(profile_a["keywords"]) & set(profile_b["keywords"])
        )
        emotion_overlap = sorted(
            set(profile_a["emotion_hits"].keys()) & set(profile_b["emotion_hits"].keys())
        )
        device_overlap = sorted(set(profile_a["devices"]) & set(profile_b["devices"]))
        contextual_keyword_links = []
        for left in profile_a["keywords"]:
            for right in profile_b["keywords"]:
                similarity = _contextual_similarity(left, right, token_contexts)
                if similarity >= 0.12:
                    contextual_keyword_links.append((left, right, round(similarity, 3)))

        lexical_semantic_similarity = float(
            (word_similarities[i][j] * 0.7) + (char_similarities[i][j] * 0.3)
        )
        embedding_similarity = (
            float(embedding_similarities[i][j]) if embedding_similarities is not None else 0.0
        )
        raw_similarity = max(lexical_semantic_similarity, embedding_similarity)
        support_bonus = 0.0
        if keyword_overlap:
            support_bonus += 0.08
        if emotion_overlap:
            support_bonus += 0.06
        if device_overlap:
            support_bonus += 0.05
        if contextual_keyword_links:
            support_bonus += min(0.18, len(contextual_keyword_links) * 0.03)

        meaning_strength = round(min(1.0, raw_similarity + support_bonus), 3)
        if meaning_strength < 0.12:
            continue

        reasons: list[str] = []
        if embedding_similarity >= 0.26:
            reasons.append("embedding meaning match")
        if lexical_semantic_similarity >= 0.12:
            reasons.append("sentence meaning overlap")
        if keyword_overlap:
            reasons.append(f"shared keywords: {', '.join(keyword_overlap[:3])}")
        if contextual_keyword_links:
            examples = [f"{left}~{right}" for left, right, _ in contextual_keyword_links[:3]]
            reasons.append(f"keyword resonance: {', '.join(examples)}")
        if emotion_overlap:
            reasons.append(f"shared emotion cues: {', '.join(emotion_overlap[:2])}")
        if device_overlap:
            reasons.append(f"shared devices: {', '.join(device_overlap[:2])}")

        links.append(
            {
                "sentence_a_index": i,
                "sentence_b_index": j,
                "sentence_a": profile_a["sentence"],
                "sentence_b": profile_b["sentence"],
                "meaning_strength": meaning_strength,
                "raw_similarity": round(raw_similarity, 3),
                "embedding_similarity": round(embedding_similarity, 3),
                "lexical_semantic_similarity": round(lexical_semantic_similarity, 3),
                "reasons": reasons,
            }
        )

    return sorted(links, key=lambda item: item["meaning_strength"], reverse=True)


def _merge_sentence_evidence(
    lexical_associations: list[dict[str, Any]],
    semantic_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[int, int], dict[str, Any]] = {}

    for edge in lexical_associations:
        key = (edge["sentence_a_index"], edge["sentence_b_index"])
        merged[key] = {
            "sentence_a_index": edge["sentence_a_index"],
            "sentence_b_index": edge["sentence_b_index"],
            "sentence_a": edge["sentence_a"],
            "sentence_b": edge["sentence_b"],
            "lexical_strength": edge["edge_strength"],
            "meaning_strength": 0.0,
            "combined_strength": round(edge["edge_strength"], 3),
            "pairings": edge["pairings"],
            "meaning_reasons": [],
        }

    for link in semantic_links:
        key = (link["sentence_a_index"], link["sentence_b_index"])
        if key not in merged:
            merged[key] = {
                "sentence_a_index": link["sentence_a_index"],
                "sentence_b_index": link["sentence_b_index"],
                "sentence_a": link["sentence_a"],
                "sentence_b": link["sentence_b"],
                "lexical_strength": 0.0,
                "meaning_strength": link["meaning_strength"],
                "combined_strength": round(link["meaning_strength"] * 2.2, 3),
                "pairings": [],
                "meaning_reasons": link["reasons"],
            }
            continue

        merged[key]["meaning_strength"] = link["meaning_strength"]
        merged[key]["meaning_reasons"] = link["reasons"]
        merged[key]["combined_strength"] = round(
            merged[key]["lexical_strength"] + (link["meaning_strength"] * 2.2), 3
        )

    return sorted(merged.values(), key=lambda item: item["combined_strength"], reverse=True)


def _connected_sentence_groups(
    sentence_count: int, combined_edges: list[dict[str, Any]]
) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(sentence_count)}
    for edge in combined_edges:
        lexical_strength = edge.get("lexical_strength", 0.0)
        meaning_strength = edge.get("meaning_strength", 0.0)
        if lexical_strength <= 0 and meaning_strength < 0.22:
            continue
        if edge["combined_strength"] < 1.4:
            continue
        a = edge["sentence_a_index"]
        b = edge["sentence_b_index"]
        adjacency[a].add(b)
        adjacency[b].add(a)

    visited: set[int] = set()
    groups: list[list[int]] = []
    for start in range(sentence_count):
        if start in visited or not adjacency[start]:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(adjacency[node] - visited)
        if len(component) >= 2:
            groups.append(sorted(component))
    return groups


def _build_word_network(pairings: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    node_weights: Counter[str] = Counter()
    adjacency: dict[str, Counter[str]] = {}

    for pair in pairings:
        left = pair["left"]
        right = pair["right"]
        weight = pair["weight"]
        node_weights[left] += weight
        node_weights[right] += weight
        adjacency.setdefault(left, Counter())[right] += weight
        adjacency.setdefault(right, Counter())[left] += weight

    return node_weights, adjacency


def _count_syllables(word: str) -> int:
    """Count syllables as number of vowel groups."""
    word = word.lower()
    if not word:
        return 1

    # Basic silent-e heuristic (e.g., made -> mad) for closer spoken rhythm.
    if len(word) > 2 and word.endswith("e") and not word.endswith(("le", "ye")):
        word = word[:-1]

    vowels = "aeiouy"
    count = 0
    previous_was_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    return max(1, count)


def _get_phonetic_ending(word: str) -> str:
    """Extract phonetic ending for rhyme detection (last 2-3 chars)."""
    word = word.lower()
    if len(word) >= 3:
        return word[-3:]
    return word


def _words_rhyme(word_a: str, word_b: str) -> bool:
    """Check if words rhyme by sharing a contiguous letter fragment (3+ chars)."""
    if word_a == word_b:
        return False
    return bool(_shared_rhyme_fragment(word_a, word_b))


def _shared_rhyme_fragment(word_a: str, word_b: str, min_size: int = 3, max_size: int = 6) -> str:
    """Return the longest shared contiguous fragment for rhyme underlining."""
    if word_a == word_b:
        return ""

    left = word_a.lower()
    right = word_b.lower()

    max_fragment = min(len(left), len(right), max_size)
    if max_fragment < min_size:
        return ""

    for size in range(max_fragment, min_size - 1, -1):
        for start in range(0, len(left) - size + 1):
            fragment = left[start : start + size]
            if fragment in right:
                return fragment
    return ""


def _same_syllables_and_similar_letters(word_a: str, word_b: str) -> bool:
    """Check if words have same syllable count and similar letters (rhythm match)."""
    if len(word_a) < 4 or len(word_b) < 4:
        return False
    
    syl_a = _count_syllables(word_a)
    syl_b = _count_syllables(word_b)

    # Allow a one-syllable gap to handle coarse syllable estimates.
    if abs(syl_a - syl_b) > 1:
        return False
    
    # Keep word lengths reasonably close for rhythm-based similarity.
    if abs(len(word_a) - len(word_b)) > 3:
        return False

    # Require stronger letter-shape overlap to avoid noisy pairings.
    shared_letters = len(set(word_a.lower()) & set(word_b.lower()))
    min_len = min(len(word_a), len(word_b))
    shared_ratio = shared_letters / min_len if min_len else 0.0

    return shared_ratio >= 0.6 and _has_adjacent_letter_overlap(word_a.lower(), word_b.lower())


def _same_phonetic_signature(word_a: str, word_b: str) -> bool:
    """Check if words have the same Soundex (phonetic signature)."""
    if word_a == word_b:
        return False
    return _soundex(word_a) == _soundex(word_b)


def _compute_six_rule_connections(text: str) -> list[dict[str, Any]]:
    """Compute word connections based on the 6 rules."""
    sentences = _split_sentences(text)
    all_tokens = []
    
    # Tokenize all sentences with token-level tracking
    for sent_idx, sentence in enumerate(sentences):
        words = _tokenize(sentence)
        normalized_words = [_normalize_token(word) for word in words]

        for idx, word in enumerate(words):
            normalized = normalized_words[idx]
            if normalized and normalized not in STOPWORDS and len(normalized) >= 3:
                all_tokens.append({
                    "word": word,
                    "normalized": normalized,
                    "sentence_index": sent_idx,
                    "stem": _normalize_token(word),
                    "sound": _soundex(normalized),
                })

            # Add synthetic compound tokens like "any more" -> "anymore".
            if idx + 1 < len(words):
                left = normalized_words[idx]
                right = normalized_words[idx + 1]
                merged = COMPOUND_JOIN_MAP.get((left, right))
                if merged:
                    all_tokens.append({
                        "word": f"{words[idx]} {words[idx + 1]}",
                        "normalized": merged,
                        "normalized_parts": [left, right],
                        "sentence_index": sent_idx,
                        "stem": merged,
                        "sound": _soundex(merged),
                    })
        
    
    connections = []
    processed_pairs = set()
    
    # Generate pairwise connections between tokens in different sentences
    for i in range(len(all_tokens)):
        for j in range(i + 1, len(all_tokens)):
            token_a = all_tokens[i]
            token_b = all_tokens[j]
            
            sentence_distance = abs(token_a["sentence_index"] - token_b["sentence_index"])

            # Only allow links between words that are at most 2 sentences apart.
            # This includes words in the same sentence.
            if sentence_distance > 2:
                continue
            
            left_key = (token_a["sentence_index"], token_a["normalized"])
            right_key = (token_b["sentence_index"], token_b["normalized"])
            pair_key = (left_key, right_key) if left_key <= right_key else (right_key, left_key)
            
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            norm_a = token_a["normalized"]
            norm_b = token_b["normalized"]
            
            matched_rule = None
            
            # Rule 1: Exact token match
            if norm_a == norm_b:
                matched_rule = 1
            
            # Rule 2: Stem match
            elif norm_a == norm_b:  # Already the same after normalization
                matched_rule = 2
            
            # Rule 5: Rhyming (dark red underline) - similar phonetic endings (check first for priority)
            elif _words_rhyme(norm_a, norm_b):
                matched_rule = 5
            
            # Rule 4: Rhythm match (blue) - same syllables + similar letters
            elif _same_syllables_and_similar_letters(norm_a, norm_b):
                matched_rule = 4
            
            # Rule 3: Phonetic signature match (light red) - same Soundex
            elif _same_phonetic_signature(norm_a, norm_b):
                matched_rule = 3
            
            # Rule 6: Idea cluster membership
            elif _is_idea_related(norm_a, norm_b):
                matched_rule = 6
            
            if matched_rule:
                connection = {
                    "rule": matched_rule,
                    "sentence_a_index": token_a["sentence_index"],
                    "normalized_a": norm_a,
                    "word_a": token_a["word"],
                    "sentence_b_index": token_b["sentence_index"],
                    "normalized_b": norm_b,
                    "word_b": token_b["word"],
                }

                compound_parts = token_a.get("normalized_parts") or token_b.get("normalized_parts")
                if matched_rule == 1 and not compound_parts:
                    # Exact repeats are not highlightable.
                    continue

                if matched_rule == 1 and compound_parts:
                    # Joined/split compounds (e.g., anymore <-> any more) are
                    # highlighted as sound similarity, not as an exact-match highlight.
                    connection["highlight_type"] = "sound_similarity"
                elif matched_rule == 3:
                    connection["highlight_type"] = "phonetic_signature"
                elif matched_rule == 4:
                    connection["highlight_type"] = "sound_similarity"
                elif matched_rule == 5:
                    connection["highlight_type"] = "rhyme"
                    fragment = _shared_rhyme_fragment(norm_a, norm_b)
                    if fragment:
                        connection["rhyme_fragment"] = fragment
                elif compound_parts:
                    fragments = [part for part in compound_parts if len(part) >= 3]
                    if fragments:
                        connection["rhyme_fragments"] = fragments

                if token_a.get("normalized_parts"):
                    connection["normalized_parts_a"] = token_a["normalized_parts"]
                if token_b.get("normalized_parts"):
                    connection["normalized_parts_b"] = token_b["normalized_parts"]

                connections.append(connection)
    
    return connections[:200]


def _content_theme_candidates(
    node_weights: Counter[str], sentence_profiles: list[dict[str, Any]], group: list[int]
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for word, weight in node_weights.items():
        normalized = _normalize_token(word)
        if normalized in THEME_NOISE_WORDS:
            continue
        counts[normalized] += weight * 1.4

    for idx in group:
        profile = sentence_profiles[idx]
        for word in profile["words"]:
            normalized = _normalize_token(word)
            if normalized in THEME_NOISE_WORDS:
                continue
            counts[normalized] += 1.0
        for word in profile["keywords"]:
            normalized = _normalize_token(word)
            if normalized in THEME_NOISE_WORDS:
                continue
            counts[normalized] += 0.5

    return counts


def _emotion_theme_candidates(
    sentence_profiles: list[dict[str, Any]], group: list[int]
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for idx in group:
        counts.update(sentence_profiles[idx]["emotion_hits"])
    return counts


def _abstract_theme_label(
    node_weights: Counter[str],
    sentence_profiles: list[dict[str, Any]],
    group: list[int],
) -> str:
    content_counts = _content_theme_candidates(node_weights, sentence_profiles, group)
    emotion_counts = _emotion_theme_candidates(sentence_profiles, group)

    group_text = " ".join(sentence_profiles[idx]["sentence"] for idx in group)
    anchor_labels: list[str] = []
    anchor_bundle = _get_theme_anchor_embeddings()
    model = _get_sentence_embedding_model()
    if anchor_bundle is not None and model is not None and group_text.strip():
        labels, anchor_embeddings = anchor_bundle
        group_embedding = model.encode([group_text], normalize_embeddings=True)
        anchor_scores = cosine_similarity(group_embedding, anchor_embeddings)[0]
        ranked_anchors = sorted(
            zip(labels, anchor_scores), key=lambda item: item[1], reverse=True
        )
        anchor_labels = [label for label, score in ranked_anchors if score >= 0.18][:3]

    content_words = [
        word
        for word, _ in content_counts.most_common(6)
        if word not in THEME_NOISE_WORDS and len(word) > 2
    ]
    emotion_labels = [
        EMOTION_THEME_LABELS.get(emotion, emotion)
        for emotion, _ in emotion_counts.most_common()
        if emotion_counts[emotion] > 0
    ]

    if not content_words and not emotion_labels:
        return "emergent pattern"

    if anchor_labels:
        anchor_components: list[str] = []
        for label in anchor_labels:
            if label not in anchor_components:
                anchor_components.append(label)
            if len(anchor_components) >= 3:
                break
        if anchor_components:
            return " / ".join(anchor_components[:3])

    components: list[str] = []
    for label in emotion_labels:
        if label not in components:
            components.append(label)
        if len(components) >= 2:
            break

    for word in content_words:
        if word not in components:
            components.append(word)
        if len(components) >= 3:
            break

    if not components:
        return "emergent pattern"

    return " / ".join(components[:3])


def _theme_interpretation(
    node_weights: Counter[str],
    relation_counts: Counter[str],
    linked_sentences: int,
    meaning_reasons: list[str],
) -> str:
    core_words = [word for word, _ in node_weights.most_common(3)]
    relation_words = [name for name, _ in relation_counts.most_common(2)]

    if not core_words:
        return "No strong cross-sentence concept thread was detected."

    semantic_clause = ""
    if meaning_reasons:
        semantic_clause = f" Semantic support comes from {'; '.join(meaning_reasons[:2])}."

    return (
        f"A recurring concept thread ties {', '.join(core_words)} across "
        f"{linked_sentences} sentences through {', '.join(relation_words)} echoes, "
        "which suggests those images are accumulating into a shared meaning rather than appearing once."
        f"{semantic_clause}"
    )


def _discover_layered_themes(text: str) -> dict[str, Any]:
    sentence_profiles = _build_sentence_profiles(text)
    sentences = [profile["sentence"] for profile in sentence_profiles]
    lexical_associations = _build_sentence_associations(text)
    semantic_links = _build_semantic_sentence_links(sentence_profiles)
    combined_edges = _merge_sentence_evidence(lexical_associations, semantic_links)
    groups = _connected_sentence_groups(len(sentences), combined_edges)

    layered_themes: list[dict[str, Any]] = []
    for group in groups:
        group_set = set(group)
        group_edges = [
            edge
            for edge in combined_edges
            if edge["sentence_a_index"] in group_set and edge["sentence_b_index"] in group_set
        ]
        pairings = [pair for edge in group_edges for pair in edge.get("pairings", [])]
        if not pairings:
            continue

        node_weights, adjacency = _build_word_network(pairings)
        relation_counts = Counter(pair["relation"] for pair in pairings)
        relation_diversity = len(relation_counts)
        meaning_reasons = []
        for edge in group_edges:
            meaning_reasons.extend(edge.get("meaning_reasons", []))
        meaning_reason_counts = Counter(meaning_reasons)

        avg_edge_strength = (
            sum(edge["combined_strength"] for edge in group_edges) / max(len(group_edges), 1)
        )
        avg_meaning_strength = (
            sum(edge.get("meaning_strength", 0.0) for edge in group_edges)
            / max(len(group_edges), 1)
        )
        sentence_support = len(group) / max(len(sentences), 1)
        confidence = min(
            0.99,
            round(
                (sentence_support * 0.45)
                + (min(avg_edge_strength / 3.2, 1.0) * 0.3)
                + (min(relation_diversity / 5.0, 1.0) * 0.15)
                + (min(avg_meaning_strength / 0.5, 1.0) * 0.1),
                3,
            ),
        )

        sentence_stage = [
            {
                "index": sentence_profiles[idx]["index"],
                "sentence": sentence_profiles[idx]["sentence"],
                "keywords": sentence_profiles[idx]["keywords"],
                "emotion_hits": sentence_profiles[idx]["emotion_hits"],
                "devices": sentence_profiles[idx]["devices"],
            }
            for idx in group[:4]
        ]

        layered_themes.append(
            {
                "theme": _abstract_theme_label(node_weights, sentence_profiles, group),
                "confidence": confidence,
                "sentence_indices": group,
                "evidence_sentences": [sentences[idx] for idx in group[:4]],
                "core_words": [word for word, _ in node_weights.most_common(5)],
                "dominant_relations": dict(relation_counts.most_common(3)),
                "key_pairings": pairings[:8],
                "interpretation": _theme_interpretation(
                    node_weights,
                    relation_counts,
                    len(group),
                    [reason for reason, _ in meaning_reason_counts.most_common(3)],
                ),
                "analysis_steps": {
                    "sentence_findings": sentence_stage,
                    "lexical_links": [
                        {
                            "sentences": [edge["sentence_a_index"], edge["sentence_b_index"]],
                            "strength": edge.get("lexical_strength", 0.0),
                            "pairings": edge.get("pairings", [])[:5],
                        }
                        for edge in group_edges
                        if edge.get("lexical_strength", 0.0) > 0
                    ][:6],
                    "semantic_support": [
                        {
                            "sentences": [edge["sentence_a_index"], edge["sentence_b_index"]],
                            "strength": edge.get("meaning_strength", 0.0),
                            "reasons": edge.get("meaning_reasons", []),
                        }
                        for edge in group_edges
                        if edge.get("meaning_strength", 0.0) > 0
                    ][:6],
                },
                "triangulation": {
                    "linked_sentences": len(group),
                    "association_edges": len(group_edges),
                    "avg_edge_strength": round(avg_edge_strength, 3),
                    "avg_meaning_strength": round(avg_meaning_strength, 3),
                },
            }
        )

    layered_themes.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "layered_themes": layered_themes[:6],
        "sentence_profiles": sentence_profiles,
        "sentence_associations": lexical_associations[:20],
        "semantic_links": semantic_links[:20],
        "combined_edges": combined_edges[:20],
    }


def _top_keywords(tokens: list[str], top_n: int = 8) -> list[str]:
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def _emotion_scores(tokens: list[str]) -> dict[str, float]:
    normalized_tokens = [_normalize_token(token) for token in tokens]
    counts = Counter(token for token in normalized_tokens if token)
    total = max(sum(counts.values()), 1)
    lookup = _emotion_lookup_map()
    hit_counts: Counter[str] = Counter()

    for token, count in counts.items():
        for emotion in lookup.get(token, set()):
            hit_counts[emotion] += count

    return {
        emotion: round(float(hit_counts.get(emotion, 0)) / total, 4)
        for emotion in EMOTION_LEXICON
    }


def _theme_label_to_emotions_map() -> dict[str, set[str]]:
    """Map a theme label (e.g. 'power') back to canonical emotion categories."""
    mapping: dict[str, set[str]] = {}
    for emotion in EMOTION_LEXICON:
        label = EMOTION_THEME_LABELS.get(emotion, emotion)
        normalized_label = _normalize_token(label)
        if not normalized_label:
            continue
        mapping.setdefault(normalized_label, set()).add(emotion)
    return mapping


def _emotion_lookup_map() -> dict[str, set[str]]:
    """Map normalized tokens to all matching emotion labels."""
    lookup: dict[str, set[str]] = {}
    for emotion, words in EMOTION_LEXICON.items():
        for word in words:
            normalized = _normalize_token(word)
            if not normalized:
                continue
            lookup.setdefault(normalized, set()).add(emotion)

    # Let theme-label aliases (including user-provided feeling words) feed emotion scoring.
    label_to_emotions = _theme_label_to_emotions_map()
    for token, label in EMOTION_THEME_LABELS.items():
        normalized_token = _normalize_token(token)
        normalized_label = _normalize_token(label)
        if not normalized_token or not normalized_label:
            continue
        for emotion in label_to_emotions.get(normalized_label, set()):
            lookup.setdefault(normalized_token, set()).add(emotion)

    return lookup


def _theme_label_lookup_map() -> dict[str, set[str]]:
    """Map normalized tokens to their configured theme labels."""
    lookup: dict[str, set[str]] = {}
    for token, label in EMOTION_THEME_LABELS.items():
        normalized = _normalize_token(token)
        if not normalized:
            continue
        lookup.setdefault(normalized, set()).add(label)
    return lookup


def _matched_feeling_terms(tokens: list[str]) -> list[dict[str, Any]]:
    """Return raw feeling words ranked by emotional relevance and frequency."""
    counts = Counter(tokens)
    emotion_lookup = _emotion_lookup_map()
    theme_lookup = _theme_label_lookup_map()
    ranked_matches: list[dict[str, Any]] = []

    for term, count in counts.items():
        normalized = _normalize_token(term)
        if not normalized:
            continue

        emotions = sorted(emotion_lookup.get(normalized, set()))
        theme_labels = set(theme_lookup.get(normalized, set()))
        for emotion in emotions:
            theme_labels.add(EMOTION_THEME_LABELS.get(emotion, emotion))

        if not emotions and not theme_labels:
            continue

        # Blend semantic relevance (mapped emotions/labels) with repetition frequency.
        emotional_relevance = round((len(emotions) * 1.0) + (len(theme_labels) * 0.35), 4)
        frequency_score = round(float(count), 4)
        ranking_score = round((emotional_relevance * 0.65) + (frequency_score * 0.35), 4)

        ranked_matches.append(
            {
                "term": term,
                "normalized": normalized,
                "count": int(count),
                "emotions": emotions,
                "theme_labels": sorted(theme_labels),
                "emotional_relevance": emotional_relevance,
                "ranking_score": ranking_score,
            }
        )

    ranked_matches.sort(key=lambda item: (item["ranking_score"], item["count"], item["term"]), reverse=True)
    return ranked_matches[:40]


def _dominant_subject_emotions(
    matched_feelings: list[dict[str, Any]],
    blended_emotion_scores: dict[str, float],
    concept_feelings: Optional[list[dict[str, Any]]] = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return top in-text feeling terms, boosted by mapped emotion strength."""
    ranked: list[dict[str, Any]] = []

    for concept in concept_feelings or []:
        ranked.append(
            {
                "feeling": concept.get("feeling", ""),
                "emotion": concept.get("feeling", ""),
                "score": float(concept.get("score", 0.0)),
                "count": int(concept.get("count", 0)),
                "mapped_emotions": concept.get("mapped_emotions", []),
                "theme_labels": concept.get("theme_labels", []),
                "evidence": concept.get("evidence", []),
            }
        )

    for item in matched_feelings:
        mapped_emotions = item.get("emotions", [])
        mapped_strength = 0.0
        if mapped_emotions:
            mapped_strength = sum(blended_emotion_scores.get(emotion, 0.0) for emotion in mapped_emotions)
            mapped_strength /= max(len(mapped_emotions), 1)

        subject_score = round((item.get("ranking_score", 0.0) * 0.75) + (mapped_strength * 10.0 * 0.25), 4)
        ranked.append(
            {
                "feeling": item.get("term", ""),
                "emotion": item.get("term", ""),
                "score": subject_score,
                "count": item.get("count", 0),
                "mapped_emotions": mapped_emotions,
                "theme_labels": item.get("theme_labels", []),
                "evidence": [item.get("term", "")],
            }
        )

    merged: dict[str, dict[str, Any]] = {}
    for entry in ranked:
        key = str(entry.get("feeling") or entry.get("emotion") or "").strip().lower()
        if not key:
            continue
        if key not in merged:
            merged[key] = entry
            continue
        merged[key]["score"] = max(float(merged[key].get("score", 0.0)), float(entry.get("score", 0.0)))
        merged[key]["count"] = int(merged[key].get("count", 0)) + int(entry.get("count", 0))
        merged[key]["mapped_emotions"] = sorted(
            set(merged[key].get("mapped_emotions", [])) | set(entry.get("mapped_emotions", []))
        )
        merged[key]["theme_labels"] = sorted(
            set(merged[key].get("theme_labels", [])) | set(entry.get("theme_labels", []))
        )
        merged[key]["evidence"] = sorted(
            set(merged[key].get("evidence", [])) | set(entry.get("evidence", []))
        )

    out = list(merged.values())
    out.sort(key=lambda entry: (entry["score"], entry["count"], entry["emotion"]), reverse=True)
    return out[:limit]


def _extract_subject_matter_feelings(
    text: str,
    blended_emotion_scores: dict[str, float],
) -> list[dict[str, Any]]:
    normalized_tokens = [_normalize_token(token) for token in _tokenize(text)]
    normalized_text = f" {' '.join(token for token in normalized_tokens if token)} "
    extracted: list[dict[str, Any]] = []

    for feeling, spec in SUBJECT_MATTER_FEELING_PATTERNS.items():
        phrases = spec.get("phrases", [])
        mapped_emotions = [str(item) for item in spec.get("emotions", [])]
        hits = 0
        evidence: list[str] = []
        for phrase in phrases:
            normalized_phrase = " ".join(_normalize_token(tok) for tok in _tokenize(str(phrase)) if _normalize_token(tok))
            if not normalized_phrase:
                continue
            if f" {normalized_phrase} " in normalized_text:
                hits += 1
                evidence.append(str(phrase))

        if hits <= 0:
            continue

        mapped_strength = 0.0
        if mapped_emotions:
            mapped_strength = sum(blended_emotion_scores.get(emotion, 0.0) for emotion in mapped_emotions)
            mapped_strength /= max(len(mapped_emotions), 1)

        theme_labels = [EMOTION_THEME_LABELS.get(emotion, emotion) for emotion in mapped_emotions]
        score = round((hits * 1.15) + (mapped_strength * 10.0 * 0.35), 4)
        extracted.append(
            {
                "feeling": feeling,
                "score": score,
                "count": hits,
                "mapped_emotions": mapped_emotions,
                "theme_labels": sorted(set(theme_labels)),
                "evidence": sorted(set(evidence)),
            }
        )

    extracted.sort(key=lambda item: (item["score"], item["count"], item["feeling"]), reverse=True)
    return extracted[:12]


def _extract_text_specific_feelings(
    text: str,
    tokens: list[str],
    matched_feelings: list[dict[str, Any]],
    blended_emotion_scores: dict[str, float],
) -> list[dict[str, Any]]:
    """Mine text-unique feeling terms so each passage yields more specific dominant feelings."""
    counts = Counter(_normalize_token(token) for token in tokens)
    counts = Counter(
        {
            token: count
            for token, count in counts.items()
            if token and token not in STOPWORDS and token not in THEME_NOISE_WORDS
        }
    )
    emotion_lookup = _emotion_lookup_map()
    theme_lookup = _theme_label_lookup_map()
    known_terms = {str(item.get("normalized", "")).strip() for item in matched_feelings}

    hyphen_terms = [term.lower() for term in re.findall(r"\b[a-zA-Z]+-[a-zA-Z]+\b", text)]
    for hyphen_term in hyphen_terms:
        normalized_hyphen = "-".join(_normalize_token(part) for part in hyphen_term.split("-"))
        if normalized_hyphen:
            counts[normalized_hyphen] += 1

    extracted: list[dict[str, Any]] = []
    for token, count in counts.items():
        if len(token) < 5:
            continue
        if token in GENERIC_ADAPTIVE_NOISE:
            continue

        is_negated = token.startswith(NEGATING_PREFIXES)
        has_abstract_suffix = token.endswith(ABSTRACT_FEELING_SUFFIXES)
        in_known = token in known_terms
        mapped_emotions = sorted(emotion_lookup.get(token, set()))
        direct_theme_labels = set(theme_lookup.get(token, set()))

        if not (in_known or has_abstract_suffix or is_negated or mapped_emotions or direct_theme_labels):
            continue

        mapped_strength = 0.0
        if mapped_emotions:
            mapped_strength = sum(blended_emotion_scores.get(emotion, 0.0) for emotion in mapped_emotions)
            mapped_strength /= max(len(mapped_emotions), 1)

        theme_labels = set(direct_theme_labels)
        for emotion in mapped_emotions:
            theme_labels.add(EMOTION_THEME_LABELS.get(emotion, emotion))

        specificity_score = 1.0
        specificity_score += min(1.2, max(0.0, (len(token) - 5) * 0.07))
        if is_negated:
            specificity_score += 0.45
        if has_abstract_suffix:
            specificity_score += 0.3
        if count >= 2:
            specificity_score += 0.35

        score = round((count * 0.9) + (specificity_score * 0.9) + (mapped_strength * 10.0 * 0.2), 4)
        extracted.append(
            {
                "feeling": token,
                "score": score,
                "count": int(count),
                "mapped_emotions": mapped_emotions,
                "theme_labels": sorted(theme_labels),
                "evidence": [token],
            }
        )

    extracted.sort(key=lambda item: (item["score"], item["count"], item["feeling"]), reverse=True)
    return extracted[:12]


def _extract_connector_context_feelings(
    text: str,
    blended_emotion_scores: dict[str, float],
) -> list[dict[str, Any]]:
    """Translate connector words into context-aware feeling concepts."""
    raw_tokens = _tokenize(text)
    normalized = [_normalize_token(token) for token in raw_tokens]
    concept_counts: Counter[str] = Counter()
    concept_evidence: dict[str, set[str]] = {}

    for idx, token in enumerate(normalized):
        concept = CONNECTOR_TO_CONCEPT.get(token)
        if not concept:
            continue

        left = max(0, idx - 3)
        right = min(len(normalized), idx + 4)
        window = [item for item in normalized[left:right] if item]

        # Temporal connectors with time-pressure context become time-anxiety.
        if concept == "waiting" and any(word in TEMPORAL_PRESSURE_WORDS for word in window):
            concept = "time-anxiety"

        concept_counts[concept] += 1
        concept_evidence.setdefault(concept, set()).add(token)

    extracted: list[dict[str, Any]] = []
    for concept, count in concept_counts.items():
        mapped_emotions = CONNECTOR_CONCEPT_EMOTIONS.get(concept, ["uncertainty"])
        mapped_strength = sum(blended_emotion_scores.get(emotion, 0.0) for emotion in mapped_emotions)
        mapped_strength /= max(len(mapped_emotions), 1)
        theme_labels = [EMOTION_THEME_LABELS.get(emotion, emotion) for emotion in mapped_emotions]
        score = round((count * 1.05) + (mapped_strength * 10.0 * 0.35), 4)
        extracted.append(
            {
                "feeling": concept,
                "score": score,
                "count": int(count),
                "mapped_emotions": mapped_emotions,
                "theme_labels": sorted(set(theme_labels)),
                "evidence": sorted(concept_evidence.get(concept, set())),
            }
        )

    extracted.sort(key=lambda item: (item["score"], item["count"], item["feeling"]), reverse=True)
    return extracted[:8]


def _rule_triangulated_emotion_scores(
    rule_connections: list[dict[str, Any]],
) -> dict[str, float]:
    """Infer emotion strength from six-rule links with triangulation bonuses."""
    if not rule_connections:
        return {emotion: 0.0 for emotion in EMOTION_LEXICON}

    lookup = _emotion_lookup_map()
    rule_weights = {
        1: 1.15,  # exact token match
        2: 1.05,  # stem/variation match
        3: 0.95,  # sound/rhythm
        4: 0.9,   # bigram similarity
        5: 0.85,  # similar letters
        6: 1.2,   # idea cluster relation
    }

    emotion_signal: Counter[str] = Counter()
    emotion_rule_support: dict[str, set[int]] = {emotion: set() for emotion in EMOTION_LEXICON}
    emotion_sentence_support: dict[str, set[int]] = {
        emotion: set() for emotion in EMOTION_LEXICON
    }
    emotion_connection_count: Counter[str] = Counter()

    for connection in rule_connections:
        left_word = _normalize_token(connection.get("normalized_a") or connection.get("word_a", ""))
        right_word = _normalize_token(connection.get("normalized_b") or connection.get("word_b", ""))
        if not left_word or not right_word:
            continue

        left_emotions = lookup.get(left_word, set())
        right_emotions = lookup.get(right_word, set())
        if not left_emotions and not right_emotions:
            continue

        rule_id = int(connection.get("rule", 0) or 0)
        rule_weight = rule_weights.get(rule_id, 1.0)
        sentence_a = int(connection.get("sentence_a_index", -1))
        sentence_b = int(connection.get("sentence_b_index", -1))

        shared_emotions = left_emotions & right_emotions
        for emotion in shared_emotions:
            emotion_signal[emotion] += 1.4 * rule_weight
            emotion_rule_support[emotion].add(rule_id)
            if sentence_a >= 0:
                emotion_sentence_support[emotion].add(sentence_a)
            if sentence_b >= 0:
                emotion_sentence_support[emotion].add(sentence_b)
            emotion_connection_count[emotion] += 1

        linked_emotions = (left_emotions | right_emotions) - shared_emotions
        for emotion in linked_emotions:
            emotion_signal[emotion] += 0.7 * rule_weight
            emotion_rule_support[emotion].add(rule_id)
            if sentence_a >= 0:
                emotion_sentence_support[emotion].add(sentence_a)
            if sentence_b >= 0:
                emotion_sentence_support[emotion].add(sentence_b)
            emotion_connection_count[emotion] += 1

    triangulated_scores: dict[str, float] = {}
    for emotion in EMOTION_LEXICON:
        base = float(emotion_signal[emotion])
        if base <= 0:
            triangulated_scores[emotion] = 0.0
            continue

        rule_diversity = len(emotion_rule_support[emotion])
        sentence_diversity = len(emotion_sentence_support[emotion])
        connection_count = int(emotion_connection_count[emotion])

        bonus = 1.0
        if rule_diversity >= 2:
            bonus += 0.25
        if sentence_diversity >= 3:
            bonus += 0.35
        if connection_count >= 3:
            bonus += 0.2

        triangulated_scores[emotion] = base * bonus

    total = sum(triangulated_scores.values())
    if total <= 0:
        return {emotion: 0.0 for emotion in EMOTION_LEXICON}

    return {
        emotion: round(triangulated_scores[emotion] / total, 4)
        for emotion in EMOTION_LEXICON
    }


def _theme_informed_emotion_scores(layered_themes: list[dict[str, Any]]) -> dict[str, float]:
    """Infer emotion strength from discovered themes and their triangulated evidence."""
    if not layered_themes:
        return {emotion: 0.0 for emotion in EMOTION_LEXICON}

    lookup = _emotion_lookup_map()
    signal: Counter[str] = Counter()

    for theme_item in layered_themes:
        confidence = float(theme_item.get("confidence", 0.0) or 0.0)
        triangulation = theme_item.get("triangulation", {})
        linked_sentences = int(triangulation.get("linked_sentences", 0) or 0)
        association_edges = int(triangulation.get("association_edges", 0) or 0)

        theme_weight = max(0.15, confidence)
        theme_weight *= 1.0 + min(0.45, (linked_sentences * 0.08) + (association_edges * 0.02))

        theme_tokens: list[str] = []
        theme_tokens.extend(_tokenize(str(theme_item.get("theme", ""))))
        theme_tokens.extend(_tokenize(str(theme_item.get("interpretation", ""))))
        theme_tokens.extend(theme_item.get("core_words", []) or [])

        for sentence_step in (theme_item.get("analysis_steps", {}) or {}).get("sentence_findings", []):
            theme_tokens.extend(sentence_step.get("keywords", []) or [])

        for raw_token in theme_tokens:
            token = _normalize_token(raw_token)
            if not token:
                continue

            if token in THEME_TO_EMOTION_HINTS:
                for emotion, weight in THEME_TO_EMOTION_HINTS[token].items():
                    signal[emotion] += weight * theme_weight

            for emotion in lookup.get(token, set()):
                signal[emotion] += 0.65 * theme_weight

    total = float(sum(signal.values()))
    if total <= 0:
        return {emotion: 0.0 for emotion in EMOTION_LEXICON}

    return {
        emotion: round(float(signal.get(emotion, 0.0)) / total, 4)
        for emotion in EMOTION_LEXICON
    }


def _blend_emotion_scores(
    lexical_scores: dict[str, float],
    rule_scores: dict[str, float],
    theme_scores: dict[str, float],
) -> dict[str, float]:
    """Blend lexical, six-rule, and theme-driven emotion signals."""
    blended: dict[str, float] = {}
    for emotion in EMOTION_LEXICON:
        lexical = lexical_scores.get(emotion, 0.0)
        triangulated = rule_scores.get(emotion, 0.0)
        theme = theme_scores.get(emotion, 0.0)
        blended[emotion] = round(
            (lexical * 0.3) + (triangulated * 0.3) + (theme * 0.4),
            4,
        )
    return blended


def _detect_alliteration(text: str) -> list[str]:
    examples: list[str] = []
    for sentence in re.split(r"[.!?\n]+", text):
        words = [w for w in _tokenize(sentence) if len(w) > 2]
        if len(words) < 4:
            continue
        initials = [w[0] for w in words]
        initial_counts = Counter(initials)
        if any(v >= 3 for v in initial_counts.values()):
            examples.append(sentence.strip())
    return examples[:4]


def _detect_similes(text: str) -> list[str]:
    simile_patterns = [
        r"\blike\s+a?n?\s+\w+",
        r"\bas\s+\w+\s+as\s+\w+",
    ]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    matches: list[str] = []
    for line in lines:
        if any(re.search(pattern, line.lower()) for pattern in simile_patterns):
            matches.append(line)
    return matches[:4]


def _line_endings_for_rhyme(text: str) -> list[str]:
    endings: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        words = _tokenize(line)
        if words:
            endings.append(words[-1])
    return endings


def _rhyme_pairs(text: str) -> list[dict[str, str]]:
    endings = _line_endings_for_rhyme(text)
    pairs: list[dict[str, str]] = []
    for i in range(len(endings)):
        for j in range(i + 1, len(endings)):
            left = endings[i]
            right = endings[j]
            if left != right and left[-3:] == right[-3:]:
                pairs.append({"a": left, "b": right})
            if len(pairs) >= 5:
                return pairs
    return pairs


def _sentence_structure_stats(text: str) -> dict[str, Any]:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = [len(_tokenize(s)) for s in sentences] if sentences else [0]
    return {
        "sentence_count": len(sentences),
        "avg_words_per_sentence": round(float(np.mean(lengths)), 2),
        "shortest_sentence_words": int(np.min(lengths)),
        "longest_sentence_words": int(np.max(lengths)),
    }


def _compact_theme_phrase(raw: str) -> str:
    words = [
        _normalize_token(word)
        for word in _tokenize(raw)
        if _normalize_token(word) and _normalize_token(word) not in THEME_NOISE_WORDS
    ]
    if not words:
        return ""
    return " ".join(words[:2])


def _theme_token_set(theme_name: str) -> set[str]:
    return {token for token in (_normalize_token(item) for item in theme_name.split()) if token}


IDEA_THEME_PATTERNS: dict[str, set[str]] = {
    "truth fracture": {"truth", "fact", "lie", "exist"},
    "self deception": {"pretend", "fake", "faking", "otherwise"},
    "meaning collapse": {"nothing", "worth", "pointless", "empty"},
    "identity fracture": {"nobody", "self", "different", "detached"},
    "ancestral continuity": {"before", "everyone", "alone", "lineage", "family"},
    "regret paradox": {"regret", "grateful", "never", "right", "blade"},
    "worldbuilding mandate": {"build", "world", "rule", "story", "musical"},
    "sacred humanism": {"human", "gods", "magical", "life", "condition"},
    "custodial burden": {"hands", "skull", "killed", "stop", "thinking"},
    "bereavement spiral": {"miss", "crying", "tears", "go", "goodbye", "end"},
    "mortality fixation": {"death", "dead", "short", "kill", "torn", "limb"},
    "ancestral mysticism": {"relatives", "god", "family", "tree", "magic", "life"},
    "surrender paradox": {"let", "go", "everything", "have", "want", "all"},
    "dream residue": {"dancing", "used", "dream", "chasing", "memory"},
    "moral dissonance": {"gratitude", "ungrateful", "ungratitude", "wisdom"},
    "temporal burden": {"forever", "lifetime", "aging", "age", "old", "late"},
    "waiting state": {"until", "while", "before", "after", "time", "delay"},
    "future pull": {"future", "closer", "next", "happen", "toward"},
    "value fracture": {"right", "wrong", "ashamed", "forgive", "guilt"},
    "purpose fracture": {"purpose", "meaning", "truth", "exist", "lie"},
    "unseen distance": {"care", "alone", "nobody", "another", "seen"},
    "identity erosion": {"lost", "youth", "self", "nobody", "detached"},
    "restless change": {"change", "happen", "stuck", "close", "step", "waiting"},
    "ironic shielding": {"serious", "joke", "drama", "humour", "sensitive"},
    "social dislocation": {"anywhere", "everywhere", "together", "alone"},
    "restless stasis": {"bored", "boredom", "waiting", "stuck"},
}


def _build_main_idea_themes(
    dominant_feelings: list[dict[str, Any]],
    subject_matter_feelings: list[dict[str, Any]],
    connector_context_feelings: list[dict[str, Any]],
    adaptive_feelings: list[dict[str, Any]],
    layered_themes: list[dict[str, Any]],
    keywords: list[str],
    text: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    dominant_tokens = {
        _normalize_token(str(item.get("feeling") or item.get("emotion", "")))
        for item in dominant_feelings
    }

    signal_tokens: Counter[str] = Counter()
    for raw in _tokenize(text):
        token = _normalize_token(raw)
        if token and token not in THEME_NOISE_WORDS:
            signal_tokens[token] += 1

    for item in layered_themes[:8]:
        confidence = float(item.get("confidence", 0.0) or 0.0)
        for source in [str(item.get("theme", "")), str(item.get("interpretation", ""))]:
            for raw in _tokenize(source):
                token = _normalize_token(raw)
                if token and token not in THEME_NOISE_WORDS:
                    signal_tokens[token] += max(1, int(round(confidence * 6)))
        for word in item.get("core_words", []) or []:
            token = _normalize_token(str(word))
            if token and token not in THEME_NOISE_WORDS:
                signal_tokens[token] += max(1, int(round(confidence * 5)))

    for idx, word in enumerate(keywords[:8]):
        token = _normalize_token(word)
        if token and token not in THEME_NOISE_WORDS:
            signal_tokens[token] += max(1, 6 - idx)

    pattern_scores: list[tuple[str, float]] = []
    for label, pattern_tokens in IDEA_THEME_PATTERNS.items():
        overlap = pattern_tokens & set(signal_tokens.keys())
        if not overlap:
            continue
        score = sum(signal_tokens[token] for token in overlap)
        pattern_scores.append((label, float(score)))

    pattern_scores.sort(key=lambda item: item[1], reverse=True)

    selected: list[tuple[str, float]] = []
    for label, score in pattern_scores:
        label_tokens = _theme_token_set(label)
        if label_tokens & dominant_tokens:
            continue
        too_similar = False
        for existing_label, _ in selected:
            overlap = _theme_token_set(existing_label) & label_tokens
            if overlap and (len(overlap) / max(1, min(len(_theme_token_set(existing_label)), len(label_tokens)))) >= 0.6:
                too_similar = True
                break
        if too_similar:
            continue
        selected.append((label, score))
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        fallback_terms = [
            token
            for token, _ in signal_tokens.most_common(12)
            if token not in dominant_tokens and token not in THEME_NOISE_WORDS and len(token) > 3
        ]
        for token in fallback_terms:
            label = _compact_theme_phrase(token)
            if not label:
                continue
            if any(label == existing for existing, _ in selected):
                continue
            selected.append((label, max(1.0, float(signal_tokens[token]))))
            if len(selected) >= limit:
                break

    if not selected:
        selected = [("core tension", 1.0)]

    top_score = max(score for _, score in selected)
    return [
        {
            "theme": label,
            "confidence": round(max(0.35, min(0.95, 0.35 + (0.6 * (score / max(top_score, 0.001))))), 2),
        }
        for label, score in selected[:limit]
    ]


def analyze_text_content(text: str) -> dict[str, Any]:
    tokens = _tokenize(text)
    keywords = _top_keywords(tokens)
    lexical_emotions = _emotion_scores(tokens)
    matched_feelings = _matched_feeling_terms(tokens)

    layered = _discover_layered_themes(text)
    layered_themes = layered["layered_themes"]

    rule_connections = _compute_six_rule_connections(text)
    triangulated_rule_emotions = _rule_triangulated_emotion_scores(rule_connections)
    theme_informed_emotions = _theme_informed_emotion_scores(layered_themes)
    emotions = _blend_emotion_scores(
        lexical_emotions,
        triangulated_rule_emotions,
        theme_informed_emotions,
    )
    concept_feelings = _extract_subject_matter_feelings(text, emotions)
    connector_feelings = _extract_connector_context_feelings(text, emotions)
    if concept_feelings and connector_feelings:
        for item in connector_feelings:
            item["score"] = round(float(item.get("score", 0.0)) * 0.62, 4)
    adaptive_feelings = _extract_text_specific_feelings(text, tokens, matched_feelings, emotions)
    dominant_feelings = _dominant_subject_emotions(
        matched_feelings,
        emotions,
        concept_feelings=(concept_feelings + connector_feelings + adaptive_feelings),
        limit=5,
    )

    if not dominant_feelings:
        dominant_feelings = [
            {
                "feeling": emotion,
                "emotion": emotion,
                "score": score,
                "count": 0,
                "mapped_emotions": [emotion],
                "theme_labels": [],
                "evidence": [],
            }
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

    if len(dominant_feelings) < 5:
        existing_keys = {
            (_normalize_token(str(item.get("feeling") or item.get("emotion", ""))) or "")
            for item in dominant_feelings
        }

        pad_candidates: list[dict[str, Any]] = []
        for source in (concept_feelings, connector_feelings, adaptive_feelings):
            pad_candidates.extend(source)

        for item in pad_candidates:
            label = str(item.get("feeling") or item.get("emotion", "")).strip()
            key = _normalize_token(label)
            if not label or not key or key in existing_keys:
                continue
            dominant_feelings.append(
                {
                    "feeling": label,
                    "emotion": label,
                    "score": float(item.get("score", 0.0)),
                    "count": int(item.get("count", 1)),
                    "mapped_emotions": list(item.get("mapped_emotions", [])),
                    "theme_labels": list(item.get("theme_labels", [])),
                    "evidence": list(item.get("evidence", [])),
                }
            )
            existing_keys.add(key)
            if len(dominant_feelings) >= 5:
                break

    if len(dominant_feelings) < 5:
        for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            label = str(emotion).strip()
            key = _normalize_token(label)
            if not label or not key:
                continue
            if any(_normalize_token(str(item.get("feeling") or item.get("emotion", ""))) == key for item in dominant_feelings):
                continue
            dominant_feelings.append(
                {
                    "feeling": label,
                    "emotion": label,
                    "score": float(score),
                    "count": 0,
                    "mapped_emotions": [emotion],
                    "theme_labels": [EMOTION_THEME_LABELS.get(emotion, emotion)],
                    "evidence": [],
                }
            )
            if len(dominant_feelings) >= 5:
                break

    dominant_feelings = dominant_feelings[:5]

    theme_map = _build_main_idea_themes(
        dominant_feelings,
        concept_feelings,
        connector_feelings,
        adaptive_feelings,
        layered_themes,
        keywords,
        text,
        limit=3,
    )

    literary_devices = {
        "alliteration_examples": _detect_alliteration(text),
        "simile_examples": _detect_similes(text),
        "rhyme_pairs": _rhyme_pairs(text),
    }

    return {
        "keywords": keywords,
        "themes": theme_map,
        "layered_themes": layered_themes,
        "sentence_profiles": layered["sentence_profiles"],
        "sentence_associations": layered["sentence_associations"],
        "semantic_links": layered["semantic_links"],
        "combined_edges": layered["combined_edges"],
        "rule_connections": rule_connections,
        "emotion_scores": emotions,
        "dominant_feelings": dominant_feelings,
        "dominant_emotions": dominant_feelings,
        "subject_matter_feelings": concept_feelings,
        "connector_context_feelings": connector_feelings,
        "adaptive_feelings": adaptive_feelings,
        "matched_feelings": matched_feelings,
        "sentence_structure": _sentence_structure_stats(text),
        "literary_devices": literary_devices,
        "visual_style_tags": [
            item.get("feeling") or item.get("emotion", "")
            for item in dominant_feelings
            if item.get("score", 0) > 0
        ],
    }


def _rgb_to_hex(color: np.ndarray) -> str:
    r, g, b = [int(x) for x in color]
    return f"#{r:02X}{g:02X}{b:02X}"


PALETTE_MIN_SATURATION = 52
PALETTE_SOFT_MIN_SATURATION = 38
PALETTE_MIN_COLOR_DISTANCE = 26.0
PALETTE_MIN_VALUE = 28
PALETTE_MAX_VALUE = 245
PALETTE_MAX_SAMPLE_PIXELS = 22000
PALETTE_NEUTRAL_CHANNEL_SPREAD = 22
PALETTE_NEAR_BLACK_VALUE = 48
PALETTE_NEAR_WHITE_VALUE = 222
PALETTE_NEAR_EXTREME_SAT = 58


def _rgb_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def _is_neutral_or_extreme_hsv(h: float, s: float, v: float, rgb: np.ndarray) -> bool:
    # Reject gray-like colors by requiring enough RGB channel spread (colorfulness/chroma proxy).
    spread = int(np.max(rgb)) - int(np.min(rgb))
    if spread < PALETTE_NEUTRAL_CHANNEL_SPREAD:
        return True

    # Reject near-black and near-white colors unless they are strongly saturated.
    if v <= PALETTE_NEAR_BLACK_VALUE and s < PALETTE_NEAR_EXTREME_SAT:
        return True
    if v >= PALETTE_NEAR_WHITE_VALUE and s < PALETTE_NEAR_EXTREME_SAT:
        return True

    return False


def extract_color_palette(image_bytes: bytes, n_colors: int = 5) -> list[str]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image. Please choose JPG, PNG, or WEBP.",
        ) from exc

    np_img = np.array(img)

    # Downsample to keep k-means fast for web usage.
    small = cv2.resize(np_img, (240, 240), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3)
    return _fit_palette_from_pixels(pixels, n_colors=n_colors)


def _fit_palette_from_pixels(pixels: np.ndarray, n_colors: int = 5) -> list[str]:
    if pixels.size == 0:
        return []

    rgb_pixels = pixels.reshape(-1, 3).astype(np.uint8)
    hsv_pixels = cv2.cvtColor(rgb_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    sat = hsv_pixels[:, 1].astype(np.float32)
    val = hsv_pixels[:, 2].astype(np.float32)

    # Hard reject neutrals: gray/black/white are excluded from palette analysis.
    strict_mask = (sat >= PALETTE_MIN_SATURATION) & (val >= PALETTE_MIN_VALUE) & (val <= PALETTE_MAX_VALUE)
    candidate_rgb = rgb_pixels[strict_mask]

    # Controlled fallback if strict mask is too small.
    if candidate_rgb.shape[0] < max(400, n_colors * 120):
        soft_mask = (sat >= PALETTE_SOFT_MIN_SATURATION) & (val >= PALETTE_MIN_VALUE) & (val <= PALETTE_MAX_VALUE)
        candidate_rgb = rgb_pixels[soft_mask]

    if candidate_rgb.shape[0] == 0:
        return []

    if candidate_rgb.shape[0] > PALETTE_MAX_SAMPLE_PIXELS:
        idx = np.random.default_rng(42).choice(candidate_rgb.shape[0], size=PALETTE_MAX_SAMPLE_PIXELS, replace=False)
        candidate_rgb = candidate_rgb[idx]

    cluster_count = int(min(max(n_colors * 6, 12), candidate_rgb.shape[0], 28))
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
    model.fit(candidate_rgb)

    centers = model.cluster_centers_.astype(np.uint8)
    labels = model.labels_
    counts = np.bincount(labels, minlength=cluster_count).astype(np.float32)

    hsv_centers = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    center_sat = hsv_centers[:, 1].astype(np.float32)
    center_val = hsv_centers[:, 2].astype(np.float32)
    center_hue = hsv_centers[:, 0].astype(np.float32)

    valid_centers = (
        (center_sat >= PALETTE_SOFT_MIN_SATURATION)
        & (center_val >= PALETTE_MIN_VALUE)
        & (center_val <= PALETTE_MAX_VALUE)
    )

    for i in range(cluster_count):
        if _is_neutral_or_extreme_hsv(
            float(center_hue[i]),
            float(center_sat[i]),
            float(center_val[i]),
            centers[i],
        ):
            valid_centers[i] = False

    # Saturation-led ranking: prioritize the most saturated representative colors.
    sat_boost = (center_sat / 255.0) ** 1.45
    weighted_score = counts * (0.15 + sat_boost)
    weighted_score[~valid_centers] = -1.0
    order = np.argsort(-weighted_score)

    selected_idx: list[int] = []

    def _hue_circular_distance(h1: float, h2: float) -> float:
        diff = abs(h1 - h2)
        return min(diff, 180.0 - diff)

    def _try_select(min_sat: float, min_dist: float, min_hue_dist: float) -> None:
        for idx in order:
            if len(selected_idx) >= n_colors:
                return
            if float(center_sat[idx]) < min_sat:
                continue
            if not bool(valid_centers[idx]):
                continue
            if any(
                _rgb_distance(centers[idx], centers[j]) < min_dist
                or _hue_circular_distance(float(center_hue[idx]), float(center_hue[j])) < min_hue_dist
                for j in selected_idx
            ):
                continue
            selected_idx.append(int(idx))

    _try_select(PALETTE_MIN_SATURATION, PALETTE_MIN_COLOR_DISTANCE, 11.0)
    if len(selected_idx) < n_colors:
        _try_select(PALETTE_SOFT_MIN_SATURATION, PALETTE_MIN_COLOR_DISTANCE * 0.72, 7.5)
    if len(selected_idx) < n_colors:
        _try_select(28.0, PALETTE_MIN_COLOR_DISTANCE * 0.52, 5.0)

    # Final fill from highest saturation centers (still excluding near-neutrals).
    if len(selected_idx) < n_colors:
        for idx in np.argsort(-center_sat):
            if len(selected_idx) >= n_colors:
                break
            if idx in selected_idx:
                continue
            if float(center_sat[idx]) < 24.0:
                continue
            if not bool(valid_centers[idx]):
                continue
            if any(_rgb_distance(centers[idx], centers[j]) < 8.0 for j in selected_idx):
                continue
            selected_idx.append(int(idx))

    return [_rgb_to_hex(centers[idx]) for idx in selected_idx[:n_colors]]


def extract_clothing_palette(image_bytes: bytes, n_colors: int = 5) -> list[str]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image. Please choose JPG, PNG, or WEBP.",
        ) from exc

    np_img = np.array(img)
    small = cv2.resize(np_img, (320, 320), interpolation=cv2.INTER_AREA)
    h, w, _ = small.shape

    # Heuristic portrait torso crop: avoids top sky/background and side walls.
    x0, x1 = int(w * 0.18), int(w * 0.82)
    y0, y1 = int(h * 0.30), int(h * 0.96)
    roi = small[y0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)

    # Conservative skin masking to avoid face/arms dominating the clothing palette.
    skin_ycrcb = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    skin_hsv_low = cv2.inRange(hsv, (0, 20, 40), (25, 255, 255))
    skin_hsv_high = cv2.inRange(hsv, (160, 20, 40), (180, 255, 255))
    skin_hsv = cv2.bitwise_or(skin_hsv_low, skin_hsv_high)
    skin_mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)

    # Ignore near-white highlights and near-black shadows.
    value = hsv[:, :, 2]
    not_extreme = (value > 18) & (value < 245)
    non_skin = skin_mask == 0
    mask = non_skin & not_extreme

    pixels = roi[mask]
    if pixels.shape[0] < 600:
        # Fallback to center-lower area if skin masking is too aggressive.
        fallback = small[int(h * 0.40):int(h * 0.95), int(w * 0.24):int(w * 0.76)]
        pixels = fallback.reshape(-1, 3)

    return _fit_palette_from_pixels(pixels.reshape(-1, 3), n_colors=n_colors)


def _scene_guidance_for_clothing(style_label: str) -> str:
    mapping = {
        "uniform_professional": "operational transit architecture, airfield-adjacent terminals, disciplined structural geometry",
        "formal_business": "street-level office facade, finance district plaza, clean architectural lines",
        "rustic_workwear": "barn-side wall, rural workshop edge, weathered timber and plaster",
        "expressive_street": "urban creative block, curated mural-adjacent facade, artistic storefront rhythm",
        "evening_editorial": "minimal gallery exterior, theater district frontage, moody refined stone surfaces",
        "minimal_monochrome": "quiet design-district frontage, restrained geometric facades, minimalist tonal surfaces",
        "utility_structured": "workshop-adjacent service corridors, industrial civic edges, utilitarian modular textures",
        "vibrant_layered": "experimental art block frontage, layered color-field facades, expressive mixed-material surfaces",
        "soft_organic": "sunlit neighborhood frontage, soft plaster-and-limewash textures, gentle handcrafted details",
        "casual_everyday": "residential mixed-use street facade, soft patched walls, lived-in urban texture",
    }
    return mapping.get(style_label, mapping["casual_everyday"])


def _infer_color_family(mean_sat: float, mean_val: float, red_ratio: float, blue_ratio: float) -> str:
    if mean_sat < 40:
        return "neutral"
    if red_ratio > max(0.16, blue_ratio + 0.05):
        return "warm"
    if blue_ratio > max(0.18, red_ratio + 0.04):
        return "cool"
    if mean_val > 165 and mean_sat < 75:
        return "pastel"
    return "balanced"


def _infer_texture_profile(edge_density: float, colorfulness: float) -> str:
    if edge_density > 0.145:
        return "structured"
    if colorfulness > 88:
        return "layered"
    if edge_density < 0.085 and colorfulness < 55:
        return "minimal"
    return "soft"


def analyze_clothing_style(image_bytes: bytes) -> dict[str, Any]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image. Please choose JPG, PNG, or WEBP.",
        ) from exc

    np_img = np.array(img)
    small = cv2.resize(np_img, (320, 320), interpolation=cv2.INTER_AREA)
    h, w, _ = small.shape

    x0, x1 = int(w * 0.18), int(w * 0.82)
    y0, y1 = int(h * 0.30), int(h * 0.96)
    roi = small[y0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    skin_ycrcb = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    skin_hsv_low = cv2.inRange(hsv, (0, 20, 40), (25, 255, 255))
    skin_hsv_high = cv2.inRange(hsv, (160, 20, 40), (180, 255, 255))
    skin_hsv = cv2.bitwise_or(skin_hsv_low, skin_hsv_high)
    skin_mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)

    value = hsv[:, :, 2]
    non_skin = skin_mask == 0
    not_extreme = (value > 18) & (value < 245)
    mask = non_skin & not_extreme
    masked_pixels = roi[mask]

    if masked_pixels.shape[0] < 500:
        masked_pixels = roi.reshape(-1, 3)

    hsv_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    mean_sat = float(np.mean(hsv_pixels[:, 1]))
    mean_val = float(np.mean(hsv_pixels[:, 2]))

    r = masked_pixels[:, 0].astype(np.float32)
    g = masked_pixels[:, 1].astype(np.float32)
    b = masked_pixels[:, 2].astype(np.float32)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    colorfulness = float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))

    edges = cv2.Canny(gray, 80, 170)
    edge_density = float(np.mean(edges > 0))

    red_ratio = float(np.mean((hsv_pixels[:, 0] <= 10) | (hsv_pixels[:, 0] >= 170)))
    blue_ratio = float(np.mean((hsv_pixels[:, 0] >= 90) & (hsv_pixels[:, 0] <= 135)))
    color_family = _infer_color_family(mean_sat, mean_val, red_ratio, blue_ratio)
    texture_profile = _infer_texture_profile(edge_density, colorfulness)

    label = "casual_everyday"
    confidence = 0.52

    if mean_sat < 40 and colorfulness < 45 and edge_density < 0.095:
        label = "minimal_monochrome"
        confidence = 0.7
    elif edge_density > 0.155 and mean_sat < 75 and colorfulness < 75:
        label = "utility_structured"
        confidence = 0.69
    elif colorfulness > 92 and mean_sat > 95:
        label = "vibrant_layered"
        confidence = 0.71
    elif mean_val > 165 and 45 <= mean_sat <= 88 and edge_density < 0.1:
        label = "soft_organic"
        confidence = 0.66
    elif mean_sat < 60 and blue_ratio > 0.24 and edge_density > 0.11:
        label = "uniform_professional"
        confidence = 0.74
    elif mean_sat < 55 and mean_val < 145:
        label = "formal_business"
        confidence = 0.72
    elif red_ratio > 0.14 and blue_ratio > 0.14 and edge_density > 0.12:
        label = "rustic_workwear"
        confidence = 0.66
    elif colorfulness > 75 and mean_sat > 90:
        label = "expressive_street"
        confidence = 0.68
    elif mean_sat < 65 and mean_val < 120 and edge_density > 0.14:
        label = "evening_editorial"
        confidence = 0.61

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "scene_guidance": _scene_guidance_for_clothing(label),
        "color_family": color_family,
        "texture_profile": texture_profile,
        "metrics": {
            "mean_saturation": round(mean_sat, 2),
            "mean_value": round(mean_val, 2),
            "edge_density": round(edge_density, 4),
            "colorfulness": round(colorfulness, 2),
            "red_ratio": round(red_ratio, 3),
            "blue_ratio": round(blue_ratio, 3),
        },
    }


def build_background_prompt(text_analysis: dict[str, Any], palette: list[str]) -> str:
    themes = [item["theme"] for item in text_analysis.get("themes", [])[:3]]
    emotions = _top_dominant_feelings(text_analysis, limit=2)
    return (
        "Create a cinematic background inspired by "
        f"themes {themes}, emotions {emotions}, and palette {palette}. "
        "Match lighting and color harmony to the subject's clothes."
    )


def _top_terms(items: list[dict[str, Any]], field: str, limit: int = 3) -> list[str]:
    terms: list[str] = []
    for item in items:
        value = str(item.get(field, "")).strip()
        if not value or value in terms:
            continue
        terms.append(value)
        if len(terms) >= limit:
            break
    return terms


def _top_dominant_feelings(text_analysis: dict[str, Any], limit: int = 3) -> list[str]:
    items = sorted(
        text_analysis.get("dominant_feelings", []),
        key=lambda item: item.get("score", 0),
        reverse=True,
    )
    terms = _top_terms(items, "feeling", limit=limit)
    if terms:
        return terms

    legacy_items = sorted(
        text_analysis.get("dominant_emotions", []),
        key=lambda item: item.get("score", 0),
        reverse=True,
    )
    return _top_terms(legacy_items, "emotion", limit=limit)


def _normalized_palette(palette: Optional[list[str]], limit: int = 6) -> list[str]:
    if not palette:
        return []

    out: list[str] = []
    for item in palette:
        value = str(item).strip().upper()
        if not re.fullmatch(r"#[0-9A-F]{6}", value):
            continue
        if value in out:
            continue
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.strip().lstrip("#")
    if len(value) != 6:
        return (127, 127, 127)
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


COLOR_NAME_SWATCHES: list[tuple[str, tuple[int, int, int]]] = [
    ("soft red", (209, 99, 110)),
    ("coral", (227, 127, 111)),
    ("peach", (232, 170, 126)),
    ("apricot", (227, 182, 120)),
    ("amber", (217, 177, 90)),
    ("mustard", (184, 156, 75)),
    ("sand", (206, 187, 151)),
    ("beige", (206, 183, 157)),
    ("taupe", (162, 142, 124)),
    ("olive", (146, 153, 92)),
    ("sage", (145, 177, 140)),
    ("mint", (143, 201, 178)),
    ("teal", (86, 164, 154)),
    ("aqua", (120, 189, 191)),
    ("sky blue", (129, 174, 216)),
    ("steel blue", (102, 133, 171)),
    ("indigo", (103, 104, 168)),
    ("lavender", (174, 154, 211)),
    ("mauve", (168, 136, 164)),
    ("rose", (200, 139, 156)),
    ("charcoal", (81, 87, 97)),
]


def _closest_color_name(rgb: tuple[int, int, int]) -> str:
    pixel = np.array(rgb, dtype=np.float32)
    best_name = "accent color"
    best_distance = float("inf")
    for name, swatch in COLOR_NAME_SWATCHES:
        distance = float(np.linalg.norm(pixel - np.array(swatch, dtype=np.float32)))
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


def _palette_named_colors(palette: Optional[list[str]], limit: int = 5) -> list[str]:
    values = _normalized_palette(palette, limit=limit)
    named: list[str] = []
    for color in values:
        name = _closest_color_name(_hex_to_rgb(color))
        named.append(f"{name} ({color})")
    return named


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    arr = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def _hsv_to_rgb(hsv: tuple[int, int, int]) -> tuple[int, int, int]:
    arr = np.array([[[hsv[0], hsv[1], hsv[2]]]], dtype=np.uint8)
    rgb = cv2.cvtColor(arr, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _pastelize_rgb(rgb: tuple[int, int, int], sat_scale: float = 0.58, val_floor: int = 176, white_mix: float = 0.34) -> tuple[int, int, int]:
    h, s, v = _rgb_to_hsv(rgb)
    s_new = int(max(22, min(145, s * sat_scale)))
    v_new = int(max(val_floor, min(252, v)))
    base = np.array(_hsv_to_rgb((h, s_new, v_new)), dtype=np.float32)
    mixed = (1.0 - white_mix) * base + white_mix * np.array([255.0, 255.0, 255.0], dtype=np.float32)
    clipped = np.clip(mixed, 0, 255).astype(np.uint8)
    return int(clipped[0]), int(clipped[1]), int(clipped[2])


def _build_clothing_harmonized_pastel_palette(
    clothing_palette: Optional[list[str]],
    full_palette: Optional[list[str]] = None,
    reference_palette: Optional[list[str]] = None,
    limit: int = 10,
) -> list[str]:
    anchors = _normalized_palette(clothing_palette, limit=8)
    full_colors = _normalized_palette(full_palette, limit=8)
    reference_colors = _normalized_palette(reference_palette, limit=8)

    if not anchors:
        fallback = _merge_prompt_palette(full_colors, reference_colors, limit=limit)
        if not fallback:
            return []
        pastel_fallback: list[str] = []
        for color in fallback:
            pastel = _rgb_to_hex(np.array(_pastelize_rgb(_hex_to_rgb(color)), dtype=np.uint8))
            if pastel not in pastel_fallback:
                pastel_fallback.append(pastel)
            if len(pastel_fallback) >= limit:
                break
        return pastel_fallback

    hue_offsets = [18, -22, 30, -35, 46, -52, 64, -70]
    palette_out: list[str] = []

    for idx, anchor_hex in enumerate(anchors):
        r, g, b = _hex_to_rgb(anchor_hex)
        h, s, v = _rgb_to_hsv((r, g, b))

        # Keep one anchor-derived pastel and one varied companion pastel for each clothing hue.
        anchor_pastel = _rgb_to_hex(np.array(_pastelize_rgb((r, g, b)), dtype=np.uint8))
        if anchor_pastel not in palette_out:
            palette_out.append(anchor_pastel)

        shift = hue_offsets[idx % len(hue_offsets)]
        companion_h = (h + shift) % 180
        companion_s = int(max(28, min(118, s * 0.62 + 18)))
        companion_v = int(max(186, min(252, v * 0.92 + 36)))
        companion_rgb = _hsv_to_rgb((companion_h, companion_s, companion_v))
        companion_pastel = _rgb_to_hex(np.array(_pastelize_rgb(companion_rgb, sat_scale=0.66, white_mix=0.29), dtype=np.uint8))
        if companion_pastel not in palette_out:
            palette_out.append(companion_pastel)

        if len(palette_out) >= limit:
            return palette_out[:limit]

    for extra in full_colors + reference_colors:
        pastel = _rgb_to_hex(np.array(_pastelize_rgb(_hex_to_rgb(extra), sat_scale=0.6), dtype=np.uint8))
        if pastel not in palette_out:
            palette_out.append(pastel)
        if len(palette_out) >= limit:
            break

    return palette_out[:limit]


def _palette_collage_guidance(palette_values: list[str]) -> str:
    if not palette_values:
        return (
            "Use a balanced collage color hierarchy: one base neutral family, one mid-tone bridge, "
            "and one restrained accent family across separate wall fragments."
        )

    rgb_values = [_hex_to_rgb(color) for color in palette_values]
    brightness = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in rgb_values]
    sorted_pairs = sorted(zip(palette_values, brightness), key=lambda item: item[1])

    dark = [item[0] for item in sorted_pairs[:2]]
    light = [item[0] for item in sorted_pairs[-2:]]
    mids = [item[0] for item in sorted_pairs[2:-2]] if len(sorted_pairs) > 4 else [item[0] for item in sorted_pairs]
    accent = palette_values[-1]

    return (
        f"Apply palette hierarchy from subject + reference colors: deep contrast accents {', '.join(dark)}, "
        f"mid-tone sections {', '.join(mids[:3])}, light lift areas {', '.join(light)}, "
        f"and controlled accent notes using {accent}. "
        "Distribute these across different collage panels instead of a single uniform wall wash. "
        "Favor a pastel multi-color editorial mix and avoid monochrome, muddy, or single-hue dominance. "
        "Use at least five distinct color families across panels, with clothing-derived hues as the primary driver."
    )


def _emotion_collage_guidance(emotions: list[str], themes: list[str]) -> str:
    if not emotions:
        return (
            "Build symbolic collage cues from the text themes: mixed geometric linework, subtle icon-like marks, "
            "and varied pattern density per wall section."
        )

    emotion_to_style = {
        "joy": "brighter shape clusters, cleaner line rhythms, lighter material transitions",
        "sadness": "soft desaturation, gentle matte textures, calmer low-contrast transitions",
        "anger": "bold geometry, sharper diagonal marks, higher contrast panel transitions",
        "fear": "tense narrow lines, compressed pattern spacing, high-contrast seams",
        "love": "curved motifs, warmer overlays, connected pattern flows",
        "surprise": "asymmetric cut panels, abrupt texture changes, offset shape blocks",
        "neutral": "balanced panel rhythm, restrained texture shifts, moderate detail density",
        "trust": "stable vertical/horizontal grids, reliable material joins, clean structural seams",
        "anticipation": "directional line movement, repeating cues, progressive panel transitions",
        "disgust": "irregular pattern boundaries, contrasted matte-gloss sections, abstract tension marks",
    }

    style_parts: list[str] = []
    for emotion in emotions[:3]:
        key = emotion.lower().strip()
        if key in emotion_to_style:
            style_parts.append(f"{emotion}: {emotion_to_style[key]}")

    theme_phrase = ", ".join(themes[:3]) if themes else "the core text themes"
    if not style_parts:
        return (
            f"Use theme-derived wall sections for {theme_phrase}: varied symbols, textures, and line systems, "
            "without repeating one surface treatment."
        )

    return (
        f"Use internal-analysis wall-language directives ({'; '.join(style_parts)}). "
        f"Embed motifs connected to themes {theme_phrase} as abstract symbols and pattern fragments on selected panels."
    )


def _build_guided_scene_structure(
    text_analysis: dict[str, Any],
    clothing_style_profile: Optional[dict[str, Any]],
    palette: Optional[list[str]],
    variation_token: Optional[str] = None,
) -> str:
    emotions = _top_dominant_feelings(text_analysis, limit=3)
    themes = _top_terms(text_analysis.get("themes", []), "theme", limit=3)
    palette_values = _normalized_palette(palette, limit=8)
    metrics = (clothing_style_profile or {}).get("metrics", {}) if clothing_style_profile else {}
    texture_profile = str((clothing_style_profile or {}).get("texture_profile", "soft")).strip() or "soft"
    color_family = str((clothing_style_profile or {}).get("color_family", "balanced")).strip() or "balanced"

    mean_sat = float(metrics.get("mean_saturation", 65.0) or 65.0)
    edge_density = float(metrics.get("edge_density", 0.11) or 0.11)
    colorfulness = float(metrics.get("colorfulness", 70.0) or 70.0)

    if any(e in {"anger", "fear", "instability"} for e in emotions):
        mood_axis = "high-tension"
    elif any(e in {"sadness", "melancholy", "detachment"} for e in emotions):
        mood_axis = "quiet-reflective"
    elif any(e in {"joy", "growth", "confidence", "awe"} for e in emotions):
        mood_axis = "uplifting-dynamic"
    else:
        mood_axis = "balanced-editorial"

    layout_templates = [
        {
            "name": "staggered-panels",
            "zones": "left 35% vertical panel stack, center 40% mixed facade seam band, right 25% inset tiles and shutters",
            "linework": "alternating vertical and horizontal rhythm on a flat frontal plane",
        },
        {
            "name": "tiered-facade",
            "zones": "upper 30% roofline strip, middle 45% multi-material wall collage, lower 25% plinth band with subtle signage traces",
            "linework": "strong horizontal strata with short vertical connectors",
        },
        {
            "name": "asymmetric-grid",
            "zones": "left 20% narrow texture spine, center 55% primary collage field, right 25% framed accent modules",
            "linework": "offset modular grid with irregular cell widths",
        },
        {
            "name": "front-elevation-bands",
            "zones": "top 22% parapet and clerestory band, middle 56% primary facade articulation, bottom 22% ground-level plinth and entry modules",
            "linework": "parallel horizontal and vertical seam bands with no vanishing-point convergence",
        },
    ]

    material_map = {
        "structured": "painted metal ribs, concrete sections, glazed block inserts, narrow brick accents",
        "layered": "painted plaster, mural overlays, ceramic tile patches, mixed masonry fragments",
        "minimal": "limewash plaster planes, restrained stone bands, smooth concrete sheets, minimal brick",
        "soft": "stucco textures, brushed plaster layers, tile trims, occasional recolored brick",
    }

    motif_map = {
        "high-tension": "compressed spacing, sharper seam turns, denser symbol clusters near panel boundaries",
        "quiet-reflective": "larger calm fields, soft transitions, sparse motifs with low-contrast cadence",
        "uplifting-dynamic": "open spacing, rising shape clusters, brighter motif pockets across multiple zones",
        "balanced-editorial": "even motif distribution, medium contrast transitions, clear panel hierarchy",
    }

    palette_hint = ", ".join(palette_values[:6]) if palette_values else "soft sage, dusty blush, powder blue, chalk cream"
    theme_hint = ", ".join(themes) if themes else "core text themes"
    emotion_hint = ", ".join(emotions) if emotions else "dominant internal signal"

    seed_parts = [
        mood_axis,
        texture_profile,
        color_family,
        f"{mean_sat:.2f}",
        f"{edge_density:.4f}",
        f"{colorfulness:.2f}",
        "|".join(emotions),
        "|".join(themes),
        "|".join(palette_values),
        variation_token or "",
    ]
    seed_text = "::".join(seed_parts)
    template_idx = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16) % len(layout_templates)
    template = layout_templates[template_idx]

    brick_cap = "max 12% brick visibility" if mean_sat >= 55 else "max 18% brick visibility"
    color_mode = "pastel multi-hue" if color_family != "neutral" else "pastel neutral-plus-accent"
    detail_density = "high" if edge_density > 0.14 or colorfulness > 86 else "medium"

    return ""


TEXT_COLOR_HEX_MAP: dict[str, str] = {
    "red": "#D66A6A",
    "crimson": "#C45B6A",
    "scarlet": "#D96A63",
    "pink": "#E4A0B7",
    "rose": "#DFA2B0",
    "peach": "#E8B08C",
    "orange": "#E4AE7E",
    "gold": "#DCC788",
    "yellow": "#E7D28B",
    "beige": "#DCC8AD",
    "brown": "#B89B84",
    "tan": "#C9AB8A",
    "green": "#93C1A0",
    "sage": "#9FBDA8",
    "mint": "#A8D6C2",
    "olive": "#AAB489",
    "teal": "#7EBAB1",
    "cyan": "#8ECAD1",
    "blue": "#8FAFD9",
    "navy": "#7E91C1",
    "indigo": "#8B8FC8",
    "purple": "#B4A1CC",
    "violet": "#BDA4CF",
    "lavender": "#C9BBE1",
    "magenta": "#CFA0C6",
    "white": "#E8E4DE",
    "gray": "#B6B8BE",
    "grey": "#B6B8BE",
    "black": "#6E727A",
}


TEXT_OBJECT_COLOR_HINTS: dict[str, list[str]] = {
    "sky": ["#A9C7EA"],
    "ocean": ["#86B5DA", "#8DD2D1"],
    "sea": ["#86B5DA", "#8DD2D1"],
    "forest": ["#96BE97", "#A9C28B"],
    "grass": ["#9FC98E"],
    "leaf": ["#A7C68D"],
    "flower": ["#DFA8C6", "#E7C29A", "#BCAED9"],
    "sun": ["#EAD18B", "#F0B28D"],
    "sunset": ["#EAA79D", "#D9A7C4", "#EBC89D"],
    "sunrise": ["#E8B9A7", "#F0D0A3", "#B7C5E8"],
    "fire": ["#E49A76", "#EBC58D"],
    "ice": ["#BCD6E8", "#C7E4E6"],
    "stone": ["#B9B0A6", "#C8BFB4"],
    "sand": ["#D9C4A6", "#E4D6BE"],
    "wood": ["#C2A58B", "#B89276"],
    "brick": ["#C89C90", "#CFAF9F"],
    "metal": ["#B8BEC7", "#AEB7C0"],
}


EMOTION_DETAIL_LIBRARY: dict[str, dict[str, Any]] = {
    "mat_patch_plaster": {
        "id": "mat_patch_plaster",
        "text": "patched matte plaster planes with subtle tonal shifts",
        "category": "material",
        "tags": ["plaster", "patch", "matte"],
        "conflict_tags": ["mirror_gloss"],
    },
    "mat_limewash_band": {
        "id": "mat_limewash_band",
        "text": "limewash wall bands with soft chalky grain",
        "category": "material",
        "tags": ["limewash", "chalk", "soft"],
        "conflict_tags": [],
    },
    "mat_rubble_stone": {
        "id": "mat_rubble_stone",
        "text": "limestone rubble wall courses with irregular hand-laid joints",
        "category": "material",
        "tags": ["stone", "rubble", "coursing"],
        "conflict_tags": ["mirror_gloss"],
    },
    "mat_whitewashed_industrial": {
        "id": "mat_whitewashed_industrial",
        "text": "whitewashed industrial frontage with patched render over masonry",
        "category": "material",
        "tags": ["whitewash", "industrial", "masonry"],
        "conflict_tags": [],
    },
    "mat_corrugated_metal": {
        "id": "mat_corrugated_metal",
        "text": "corrugated metal cladding strips mixed with plaster fields",
        "category": "material",
        "tags": ["corrugated", "metal", "cladding"],
        "conflict_tags": ["soft_limewash_only"],
    },
    "mat_stone_storefront": {
        "id": "mat_stone_storefront",
        "text": "weathered stone storefront skin with rough masonry patchwork",
        "category": "material",
        "tags": ["stone", "storefront", "weathered"],
        "conflict_tags": ["mirror_gloss"],
    },
    "mat_galvanized_patch": {
        "id": "mat_galvanized_patch",
        "text": "patched galvanized sheet surfaces integrated into lower facade bands",
        "category": "material",
        "tags": ["galvanized", "sheet", "patch"],
        "conflict_tags": [],
    },
    "open_narrow_gate": {
        "id": "open_narrow_gate",
        "text": "a narrow steel pedestrian gate inset into the facade",
        "category": "opening",
        "tags": ["gate", "steel", "entry"],
        "conflict_tags": [],
    },
    "open_recessed_entry": {
        "id": "open_recessed_entry",
        "text": "a recessed entry bay with simple framed door geometry",
        "category": "opening",
        "tags": ["recessed", "entry", "frame"],
        "conflict_tags": [],
    },
    "open_central_service_gate": {
        "id": "open_central_service_gate",
        "text": "a central metal service gate anchoring the ground-level composition",
        "category": "opening",
        "tags": ["service", "gate", "ground-level"],
        "conflict_tags": [],
    },
    "open_double_metal_door": {
        "id": "open_double_metal_door",
        "text": "patched double metal doors with utilitarian panel geometry",
        "category": "opening",
        "tags": ["double-door", "metal", "utility"],
        "conflict_tags": [],
    },
    "open_black_grid_gate": {
        "id": "open_black_grid_gate",
        "text": "a matte black steel gate with strict grid subdivision",
        "category": "opening",
        "tags": ["black", "grid", "gate"],
        "conflict_tags": ["ornate_timber"],
    },
    "open_carved_timber_entry": {
        "id": "open_carved_timber_entry",
        "text": "a carved timber entry surround with frosted transom glazing",
        "category": "opening",
        "tags": ["timber", "entry", "heritage"],
        "conflict_tags": ["strict_industrial"],
    },
    "open_single_punched_window": {
        "id": "open_single_punched_window",
        "text": "a single punched window opening with deep reveal",
        "category": "opening",
        "tags": ["window", "punched", "reveal"],
        "conflict_tags": [],
    },
    "age_hairline_cracks": {
        "id": "age_hairline_cracks",
        "text": "fine hairline render cracks and light weather staining",
        "category": "aging",
        "tags": ["crack", "weathered", "patina"],
        "conflict_tags": ["fresh_pristine"],
    },
    "age_soft_patina": {
        "id": "age_soft_patina",
        "text": "soft patina streaks around seams and drainage edges",
        "category": "aging",
        "tags": ["patina", "streak", "weathered"],
        "conflict_tags": ["fresh_pristine"],
    },
    "age_moss_capstone": {
        "id": "age_moss_capstone",
        "text": "moss-darkened capstone edges and damp weather marks",
        "category": "aging",
        "tags": ["moss", "capstone", "damp"],
        "conflict_tags": ["fresh_pristine"],
    },
    "age_rust_streaks": {
        "id": "age_rust_streaks",
        "text": "rust streaking beneath metal seams and fastening points",
        "category": "aging",
        "tags": ["rust", "metal", "streak"],
        "conflict_tags": ["fresh_pristine"],
    },
    "age_peeling_paint": {
        "id": "age_peeling_paint",
        "text": "peeling paint layers exposing older substrate tones",
        "category": "aging",
        "tags": ["peeling", "paint", "layered-age"],
        "conflict_tags": ["fresh_pristine"],
    },
    "age_repair_scars": {
        "id": "age_repair_scars",
        "text": "visible repair scars and patched render seams",
        "category": "aging",
        "tags": ["repair", "scar", "patch"],
        "conflict_tags": ["fresh_pristine"],
    },
    "ctx_ordered_wires": {
        "id": "ctx_ordered_wires",
        "text": "lightly visible service conduits and cables routed in orderly runs",
        "category": "context",
        "tags": ["utility", "cable", "ordered"],
        "conflict_tags": ["dense_cable_tangle"],
    },
    "ctx_dense_wires": {
        "id": "ctx_dense_wires",
        "text": "dense overhead utility wiring crossing multiple facade zones",
        "category": "context",
        "tags": ["utility", "cable", "dense_cable_tangle"],
        "conflict_tags": ["ordered"],
    },
    "ctx_rooftop_water_tank": {
        "id": "ctx_rooftop_water_tank",
        "text": "a rooftop water tank silhouette set behind the parapet",
        "category": "context",
        "tags": ["rooftop", "water-tank", "silhouette"],
        "conflict_tags": [],
    },
    "ctx_rooftop_fence_vents": {
        "id": "ctx_rooftop_fence_vents",
        "text": "rooftop fence rails and vent stacks punctuating the skyline edge",
        "category": "context",
        "tags": ["rooftop", "fence", "vents"],
        "conflict_tags": [],
    },
    "ctx_service_boxes": {
        "id": "ctx_service_boxes",
        "text": "small service boxes and meter plates integrated near entries",
        "category": "context",
        "tags": ["service", "meter", "utility"],
        "conflict_tags": [],
    },
    "ctx_ground_plinth": {
        "id": "ctx_ground_plinth",
        "text": "a durable ground-level plinth band with subtle street wear",
        "category": "context",
        "tags": ["plinth", "ground", "street"],
        "conflict_tags": [],
    },
    "acc_tile_patch": {
        "id": "acc_tile_patch",
        "text": "small ceramic tile patch accents embedded between wall fields",
        "category": "accent",
        "tags": ["tile", "patch", "accent"],
        "conflict_tags": [],
    },
    "acc_gate_grille": {
        "id": "acc_gate_grille",
        "text": "simple ornamental grille motifs on gate surfaces",
        "category": "accent",
        "tags": ["grille", "gate", "ornament"],
        "conflict_tags": [],
    },
    "acc_checker_mural": {
        "id": "acc_checker_mural",
        "text": "a restrained checker mural insert as a localized wall accent",
        "category": "accent",
        "tags": ["checker", "mural", "graphic"],
        "conflict_tags": ["full-facade-mural"],
    },
    "acc_horse_stencil": {
        "id": "acc_horse_stencil",
        "text": "a hand-painted horse stencil motif on one gate panel",
        "category": "accent",
        "tags": ["stencil", "gate", "graphic"],
        "conflict_tags": [],
    },
    "acc_hand_marked_gate": {
        "id": "acc_hand_marked_gate",
        "text": "hand-marked gate surfaces with subtle textural marks",
        "category": "accent",
        "tags": ["hand-marked", "gate", "surface"],
        "conflict_tags": [],
    },
    "acc_poster_remnants": {
        "id": "acc_poster_remnants",
        "text": "faded poster remnants and paint ghosts in limited patches",
        "category": "accent",
        "tags": ["poster", "faded", "ghost"],
        "conflict_tags": [],
    },
    "rhythm_soft_horizontal": {
        "id": "rhythm_soft_horizontal",
        "text": "calm horizontal facade bands with wider spacing",
        "category": "rhythm",
        "tags": ["horizontal", "calm", "spaced"],
        "conflict_tags": ["compressed"],
    },
    "rhythm_compressed_vertical": {
        "id": "rhythm_compressed_vertical",
        "text": "compressed vertical seams and tighter panel cadence",
        "category": "rhythm",
        "tags": ["vertical", "compressed", "tense"],
        "conflict_tags": ["spaced"],
    },
    "rhythm_stone_coursing": {
        "id": "rhythm_stone_coursing",
        "text": "tight horizontal stone coursing that stabilizes the facade base",
        "category": "rhythm",
        "tags": ["stone", "horizontal", "coursing"],
        "conflict_tags": [],
    },
    "rhythm_asym_window_grid": {
        "id": "rhythm_asym_window_grid",
        "text": "an asymmetrical punched-window grid with uneven bay spacing",
        "category": "rhythm",
        "tags": ["window", "grid", "asymmetric"],
        "conflict_tags": [],
    },
    "rhythm_panel_mosaic": {
        "id": "rhythm_panel_mosaic",
        "text": "a mixed panel mosaic rhythm across 6-10 facade sections",
        "category": "rhythm",
        "tags": ["panel", "mosaic", "sectional"],
        "conflict_tags": [],
    },
    "rhythm_roofline_banding": {
        "id": "rhythm_roofline_banding",
        "text": "stacked parapet and roofline banding with restrained vertical breaks",
        "category": "rhythm",
        "tags": ["roofline", "parapet", "banding"],
        "conflict_tags": [],
    },
}


EMOTION_DETAIL_PROFILES: dict[str, dict[str, list[str]]] = {
    "calm": {
        "must": ["mat_limewash_band", "rhythm_soft_horizontal"],
        "optional": [
            "mat_patch_plaster",
            "open_recessed_entry",
            "age_soft_patina",
            "ctx_ordered_wires",
            "ctx_ground_plinth",
            "acc_tile_patch",
            "rhythm_roofline_banding",
        ],
    },
    "joy": {
        "must": ["acc_tile_patch"],
        "optional": [
            "mat_patch_plaster",
            "open_recessed_entry",
            "age_soft_patina",
            "ctx_ordered_wires",
            "ctx_rooftop_fence_vents",
            "acc_gate_grille",
            "rhythm_panel_mosaic",
        ],
    },
    "melancholy": {
        "must": ["age_hairline_cracks", "age_soft_patina"],
        "optional": [
            "mat_limewash_band",
            "mat_rubble_stone",
            "open_narrow_gate",
            "open_single_punched_window",
            "ctx_ground_plinth",
            "acc_poster_remnants",
            "rhythm_soft_horizontal",
        ],
    },
    "tension": {
        "must": ["rhythm_compressed_vertical", "ctx_dense_wires"],
        "optional": [
            "mat_corrugated_metal",
            "mat_stone_storefront",
            "open_narrow_gate",
            "open_double_metal_door",
            "age_repair_scars",
            "age_rust_streaks",
            "acc_checker_mural",
        ],
    },
    "awe": {
        "must": ["open_recessed_entry", "rhythm_roofline_banding"],
        "optional": [
            "mat_whitewashed_industrial",
            "mat_limewash_band",
            "ctx_rooftop_water_tank",
            "ctx_rooftop_fence_vents",
            "age_soft_patina",
            "acc_tile_patch",
            "rhythm_panel_mosaic",
        ],
    },
    "nostalgia": {
        "must": ["age_soft_patina", "acc_gate_grille"],
        "optional": [
            "mat_patch_plaster",
            "mat_rubble_stone",
            "open_carved_timber_entry",
            "age_moss_capstone",
            "ctx_service_boxes",
            "acc_hand_marked_gate",
            "rhythm_stone_coursing",
        ],
    },
    "ancestral": {
        "must": ["open_carved_timber_entry", "rhythm_stone_coursing"],
        "optional": [
            "mat_rubble_stone",
            "mat_stone_storefront",
            "age_moss_capstone",
            "age_soft_patina",
            "ctx_ground_plinth",
            "acc_gate_grille",
            "acc_poster_remnants",
        ],
    },
    "worldbuild": {
        "must": ["rhythm_panel_mosaic", "open_central_service_gate"],
        "optional": [
            "mat_whitewashed_industrial",
            "mat_corrugated_metal",
            "mat_galvanized_patch",
            "open_black_grid_gate",
            "ctx_rooftop_fence_vents",
            "ctx_service_boxes",
            "acc_checker_mural",
            "acc_horse_stencil",
        ],
    },
    "burden": {
        "must": ["ctx_dense_wires", "open_double_metal_door"],
        "optional": [
            "mat_stone_storefront",
            "mat_galvanized_patch",
            "age_repair_scars",
            "age_rust_streaks",
            "ctx_service_boxes",
            "acc_hand_marked_gate",
            "rhythm_compressed_vertical",
        ],
    },
    "transition": {
        "must": ["rhythm_asym_window_grid", "open_recessed_entry"],
        "optional": [
            "mat_patch_plaster",
            "mat_whitewashed_industrial",
            "open_narrow_gate",
            "age_peeling_paint",
            "ctx_rooftop_water_tank",
            "acc_tile_patch",
            "acc_poster_remnants",
        ],
    },
}


DETAIL_CATEGORY_QUOTAS: dict[str, int] = {
    "material": 1,
    "opening": 1,
    "aging": 1,
    "context": 1,
    "accent": 1,
}


EMOTION_TO_DETAIL_CLASS: dict[str, str] = {
    "joy": "joy",
    "calm": "calm",
    "sadness": "melancholy",
    "melancholy": "melancholy",
    "detachment": "melancholy",
    "anger": "tension",
    "fear": "tension",
    "instability": "tension",
    "awe": "awe",
    "growth": "awe",
    "confidence": "awe",
    "belonging": "nostalgia",
    "shame": "nostalgia",
    "presence": "calm",
    "uncertainty": "tension",
}


COMPLEX_EMOTION_TO_DETAIL_CLASS: dict[str, str] = {
    "self-deception": "tension",
    "meaninglessness": "melancholy",
    "wisdom-burden": "burden",
    "mortality-devotion": "awe",
    "sacrificial-heroism": "awe",
    "sacred-faith": "awe",
    "purpose-struggle": "transition",
    "directionless-change": "transition",
    "social-invisibility": "melancholy",
    "private-grief": "melancholy",
    "youth-mourning": "nostalgia",
    "paradox-tension": "tension",
    "existential-disbelief": "tension",
    "cosmic-smallness": "awe",
    "bereavement-spiral": "melancholy",
    "mortality-fixation": "tension",
    "surrender-paradox": "transition",
    "ancestral-mysticism": "ancestral",
    "ancestral-continuity": "ancestral",
    "worldbuilding-mandate": "worldbuild",
    "sacred-humanism": "worldbuild",
    "custodial-burden": "burden",
    "time-anxiety": "tension",
    "waiting": "melancholy",
    "anticipation": "transition",
    "aftermath": "melancholy",
    "inner-conflict": "tension",
    "avoidance": "tension",
    "justification": "burden",
    "consequence": "burden",
}


def _normalize_concept_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return normalized


def _sanitize_prompt_language(prompt_text: str) -> str:
    # Keep analysis vocabulary intact, but sanitize prompt wording that tends to push dreamy softness.
    sanitized = prompt_text
    replacements = [
        (r"\bsacred humanism\b", "human condition"),
        (r"\bancestral mysticism\b", "ancestral history"),
        (r"\bsurrender paradox\b", "release conflict"),
        (r"\bmysticism\b", "mystic"),
    ]
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def _map_emotion_to_detail_class(emotion: str) -> str:
    token_normalized = _normalize_token(str(emotion or ""))
    concept_normalized = _normalize_concept_key(str(emotion or ""))
    if concept_normalized in COMPLEX_EMOTION_TO_DETAIL_CLASS:
        return COMPLEX_EMOTION_TO_DETAIL_CLASS[concept_normalized]
    return EMOTION_TO_DETAIL_CLASS.get(token_normalized, "calm")


def _weighted_choice_without_replacement(
    rng: random.Random,
    items: list[str],
    weight_fn: Any,
) -> Optional[str]:
    weighted = [(item, float(max(0.0, weight_fn(item)))) for item in items]
    total = sum(weight for _, weight in weighted)
    if total <= 0:
        return None

    threshold = rng.uniform(0.0, total)
    running = 0.0
    for item, weight in weighted:
        running += weight
        if running >= threshold:
            return item
    return weighted[-1][0]


def _detail_conflicts_with_selected(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
    candidate_tags = set(candidate.get("tags", []))
    candidate_conflicts = set(candidate.get("conflict_tags", []))
    for existing in selected:
        existing_tags = set(existing.get("tags", []))
        existing_conflicts = set(existing.get("conflict_tags", []))
        if candidate_conflicts & existing_tags:
            return True
        if existing_conflicts & candidate_tags:
            return True
    return False


def _select_emotion_prompt_details(
    emotions: list[str],
    themes: list[str],
    seed_basis: str,
) -> dict[str, Any]:
    mapped_classes = [_map_emotion_to_detail_class(emotion) for emotion in emotions if emotion]
    classes: list[str] = []
    for class_name in mapped_classes:
        if class_name and class_name not in classes:
            classes.append(class_name)

    primary_class = classes[0] if classes else "calm"
    secondary_classes = classes[1:3]
    profile = EMOTION_DETAIL_PROFILES.get(primary_class, EMOTION_DETAIL_PROFILES["calm"])
    secondary_profiles = [
        EMOTION_DETAIL_PROFILES[class_name]
        for class_name in secondary_classes
        if class_name in EMOTION_DETAIL_PROFILES
    ]
    rng = random.Random(seed_basis)
    theme_tokens = set(_normalize_token(token) for token in _tokenize(" ".join(themes)) if token)

    selected_ids: list[str] = []
    selected_details: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()

    must_pool = [detail_id for detail_id in profile.get("must", []) if detail_id in EMOTION_DETAIL_LIBRARY]
    if must_pool:
        must_pick = rng.choice(must_pool)
        detail = EMOTION_DETAIL_LIBRARY[must_pick]
        selected_ids.append(must_pick)
        selected_details.append(detail)
        category_counts[detail.get("category", "")] += 1

    candidate_pool = [
        detail_id
        for detail_id in (
            profile.get("optional", [])
            + profile.get("must", [])
            + [
                detail_id
                for secondary_profile in secondary_profiles
                for detail_id in (secondary_profile.get("optional", []) + secondary_profile.get("must", []))
            ]
        )
        if detail_id in EMOTION_DETAIL_LIBRARY and detail_id not in selected_ids
    ]
    # Keep insertion order while removing duplicates.
    candidate_pool = list(dict.fromkeys(candidate_pool))

    primary_pool = set(profile.get("optional", []) + profile.get("must", []))
    secondary_pool = {
        detail_id
        for secondary_profile in secondary_profiles
        for detail_id in (secondary_profile.get("optional", []) + secondary_profile.get("must", []))
    }

    def _weight(detail_id: str) -> float:
        detail = EMOTION_DETAIL_LIBRARY[detail_id]
        tags = set(detail.get("tags", []))
        category = str(detail.get("category", ""))
        category_target = DETAIL_CATEGORY_QUOTAS.get(category, 1)
        under_quota_bonus = 1.6 if category_counts[category] < category_target else 0.55
        theme_bonus = 1.25 if tags & theme_tokens else 1.0
        class_bonus = 1.0
        if detail_id in primary_pool:
            class_bonus = 1.35
        elif detail_id in secondary_pool:
            class_bonus = 1.12
        return under_quota_bonus * theme_bonus * class_bonus

    for category, quota in DETAIL_CATEGORY_QUOTAS.items():
        while category_counts[category] < quota:
            filtered_candidates = []
            for detail_id in candidate_pool:
                detail = EMOTION_DETAIL_LIBRARY[detail_id]
                if detail.get("category") != category:
                    continue
                if _detail_conflicts_with_selected(detail, selected_details):
                    continue
                filtered_candidates.append(detail_id)

            if not filtered_candidates:
                break

            picked = _weighted_choice_without_replacement(rng, filtered_candidates, _weight)
            if not picked:
                break

            detail = EMOTION_DETAIL_LIBRARY[picked]
            selected_ids.append(picked)
            selected_details.append(detail)
            category_counts[category] += 1
            candidate_pool = [detail_id for detail_id in candidate_pool if detail_id != picked]

    # Fill one extra rhythm detail when available to increase variation texture.
    rhythm_candidates = [
        detail_id
        for detail_id in candidate_pool
        if EMOTION_DETAIL_LIBRARY[detail_id].get("category") == "rhythm"
        and not _detail_conflicts_with_selected(EMOTION_DETAIL_LIBRARY[detail_id], selected_details)
    ]
    if rhythm_candidates:
        picked_rhythm = _weighted_choice_without_replacement(rng, rhythm_candidates, _weight)
        if picked_rhythm:
            selected_ids.append(picked_rhythm)
            selected_details.append(EMOTION_DETAIL_LIBRARY[picked_rhythm])

    slot_order = ["material", "opening", "aging", "context", "accent", "rhythm"]
    slot_map: dict[str, str] = {}
    for slot in slot_order:
        for detail in selected_details:
            if detail.get("category") == slot:
                slot_map[slot] = str(detail.get("text", ""))
                break

    detail_phrases = [str(detail.get("text", "")) for detail in selected_details if detail.get("text")]

    return {
        "emotion_class": primary_class,
        "selected_detail_ids": selected_ids,
        "selected_details": selected_details,
        "slot_map": slot_map,
        "detail_phrase": "; ".join(detail_phrases),
    }


def _compose_case_specific_prompt_completion(
    emotions: list[str],
    themes: list[str],
    colors: list[str],
    emotion_detail_plan: Optional[dict[str, Any]],
    structure_hint: Optional[str],
    text_color_hint: Optional[str],
    variation_hint: Optional[str],
    reference_name: Optional[str],
) -> str:
    parts: list[str] = []

    if emotions:
        parts.append(f"Emotion set: {', '.join(emotions)}.")
    if themes:
        parts.append(f"Theme set: {', '.join(themes)}.")
    if colors:
        parts.append(f"Color set: {', '.join(colors[:6])}.")

    if emotion_detail_plan:
        detail_phrase = str(emotion_detail_plan.get("detail_phrase", "")).strip()
        if detail_phrase:
            parts.append(f"Emotion detail translation: {detail_phrase}.")
        slot_map = emotion_detail_plan.get("slot_map", {}) if isinstance(emotion_detail_plan, dict) else {}
        if isinstance(slot_map, dict) and slot_map:
            slot_bits: list[str] = []
            for slot in ["material", "opening", "aging", "context", "accent", "rhythm"]:
                value = str(slot_map.get(slot, "")).strip()
                if value:
                    slot_bits.append(f"{slot}: {value}")
            if slot_bits:
                parts.append("Detail slots: " + " | ".join(slot_bits) + ".")

    # Intentionally do not append structure blueprint prose to the final prompt.
    if text_color_hint:
        parts.append(text_color_hint.strip())
    if variation_hint:
        parts.append(variation_hint.strip())
    if reference_name:
        parts.append(f"Reference anchor: {reference_name}.")

    if not parts:
        fallback_emotion = emotions[0] if emotions else (themes[0] if themes else "internal signal")
        fallback_color = colors[0] if colors else "muted neutral"
        return (
            f"Emotion set: {fallback_emotion}. "
            f"Color set: {fallback_color}. "
            "Emotion detail translation: a single symbolic object with a named material and explicit placement is required."
        )
    return " ".join(parts)


def _pastelize_hex(color: str) -> str:
    return _rgb_to_hex(np.array(_pastelize_rgb(_hex_to_rgb(color)), dtype=np.uint8))


def _extract_text_color_cues(text_analysis: dict[str, Any], limit: int = 8) -> list[str]:
    tokens: list[str] = []
    for keyword in text_analysis.get("keywords", []):
        token = _normalize_token(str(keyword))
        if token:
            tokens.append(token)

    for theme in text_analysis.get("themes", []):
        raw_theme = str(theme.get("theme", ""))
        for piece in re.split(r"[^a-zA-Z]+", raw_theme):
            token = _normalize_token(piece)
            if token:
                tokens.append(token)

    out: list[str] = []
    for token in tokens:
        direct = TEXT_COLOR_HEX_MAP.get(token)
        if direct:
            pastel = _pastelize_hex(direct)
            if pastel not in out:
                out.append(pastel)

        for hinted in TEXT_OBJECT_COLOR_HINTS.get(token, []):
            pastel = _pastelize_hex(hinted)
            if pastel not in out:
                out.append(pastel)

        if len(out) >= limit:
            break

    return out[:limit]


def _outfit_world_building_guidance(
    clothing_style_profile: Optional[dict[str, Any]],
    palette: Optional[list[str]],
    emotions: Optional[list[str]] = None,
    themes: Optional[list[str]] = None,
) -> str:
    label = str((clothing_style_profile or {}).get("label", "casual_everyday")).strip() or "casual_everyday"

    style_world = {
        "uniform_professional": {
            "places": "airfield service lane, terminal operations edge, transport-grade civic frontage",
            "walls": "modular aviation panels, painted metal sections, clean glass-and-composite wall strips",
            "roofing": "hangar roof trusses, corrugated canopy bands, skylight frames",
        },
        "formal_business": {
            "places": "commercial district frontage, corporate plaza edge, structured urban corridor",
            "walls": "stone cladding panels, polished concrete sections, muted glazed blocks",
            "roofing": "flat parapet lines, modern canopy edges, steel soffit bands",
        },
        "rustic_workwear": {
            "places": "workshop yard frontage, industrial-rural service edge, maker district corridor",
            "walls": "timber sections, lime-wash plaster patches, reclaimed brick inserts",
            "roofing": "corrugated roof panels, painted rafters, exposed beam transitions",
        },
        "expressive_street": {
            "places": "creative street frontage, mural district side lane, gallery-adjacent block",
            "walls": "paint-layered wall fragments, poster-remnant sections, mixed masonry patches",
            "roofing": "retro awning strips, asymmetric roof edges, patched canopy layers",
        },
        "evening_editorial": {
            "places": "theater district edge, curated cultural block, minimalist gallery exterior",
            "walls": "dark mineral plaster sections, refined stone slabs, matte concrete panels",
            "roofing": "clean roofline silhouettes, recessed ledge bands, subtle overhang details",
        },
        "casual_everyday": {
            "places": "mixed-use neighborhood frontage, everyday pedestrian street edge, lived-in city block",
            "walls": "painted plaster sections, soft brick segments, patched stucco and tile pieces",
            "roofing": "simple awnings, balcony slab edges, practical roof trim bands",
        },
    }

    world = style_world.get(label, style_world["casual_everyday"])
    palette_values = _normalized_palette(palette, limit=8)
    palette_hint = ", ".join(palette_values) if palette_values else "muted neutrals with selective accent tones"
    collage_color_phrase = _palette_collage_guidance(palette_values)
    collage_emotion_phrase = _emotion_collage_guidance(emotions or [], themes or [])

    return (
        f"World-build the background from outfit archetype '{label}' using places like {world['places']}. "
        f"Vary the building type, facade details, and material stack using wall language from {world['walls']} and roofing cues from {world['roofing']}. "
        "Render the single building facade with mixed but coherent real-material sections that read clearly and crisply: "
        "segmented facade zones, varied roofing pieces, and layered material seams. "
        "Create 6-10 distinct facade zones with variation in symbols, pattern families, texture grain, geometric shapes, "
        "line direction, shade contrast, micro-details, and finish states (matte, satin, painted, glazed, brushed). "
        "Use mixed material identities across sections such as plaster, brick, concrete, tile, painted metal, glass, and masonry. "
        "Keep the building clean and intentional, never distressed or ruin-like. "
        "Keep the environment curated and non-destructive: no ruins, no rubble, no conflict-zone damage, no bombed surfaces. "
        f"{collage_color_phrase} "
        f"{collage_emotion_phrase} "
        f"Keep color continuity with outfit palette anchors: {palette_hint}."
    )


def _merge_prompt_palette(
    subject_palette: Optional[list[str]],
    reference_palette: Optional[list[str]],
    limit: int = 10,
) -> list[str]:
    # Reference colors lead to match the provided visual references; subject colors keep identity continuity.
    ref = _normalized_palette(reference_palette, limit=limit)
    subj = _normalized_palette(subject_palette, limit=limit)

    merged: list[str] = []
    for color in ref + subj:
        if color in merged:
            continue
        merged.append(color)
        if len(merged) >= limit:
            break
    return merged


def _build_foreground_protection_prompt(clothing_style_profile: Optional[dict[str, Any]]) -> str:
    label = str((clothing_style_profile or {}).get("label", "")).strip()
    if label:
        return (
            "single-person portrait subject, preserve original face, body, and outfit exactly, "
            f"outfit archetype remains {label}"
        )
    return "single-person portrait subject, preserve original face, body, and outfit exactly"


def _building_size_style_phrase(
    clothing_style_profile: Optional[dict[str, Any]],
    emotion_detail_plan: Optional[dict[str, Any]],
) -> str:
    style_label = str((clothing_style_profile or {}).get("label", "casual_everyday")).strip() or "casual_everyday"
    detail_class = str((emotion_detail_plan or {}).get("emotion_class", "calm")).strip() or "calm"

    style_map = {
        "uniform_professional": "mid-rise transport-influenced",
        "formal_business": "mid-rise commercial",
        "rustic_workwear": "low-rise industrial-vernacular",
        "expressive_street": "mid-rise mixed-use creative",
        "evening_editorial": "mid-rise contemporary",
        "casual_everyday": "low-to-mid rise neighborhood",
    }
    class_tone_map = {
        "tension": "with compressed facade rhythm",
        "melancholy": "with weathered facade character",
        "awe": "with strong vertical presence",
        "nostalgia": "with inherited masonry character",
        "ancestral": "with heritage wall language",
        "worldbuild": "with layered collage articulation",
        "burden": "with dense service-layer expression",
        "transition": "with shifting panel geometry",
        "joy": "with open patterned articulation",
        "calm": "with restrained linear articulation",
    }

    base = style_map.get(style_label, "low-to-mid rise neighborhood")
    tone = class_tone_map.get(detail_class, "with restrained linear articulation")
    return f"{base} {tone}"


def _detail_color_placement_phrase(
    slot_map: dict[str, Any],
    portrait_named_values: list[str],
    palette_named_values: list[str],
    detail_phrase: str,
) -> str:
    colors = portrait_named_values[:5] + [value for value in palette_named_values if value not in portrait_named_values]
    if not colors:
        return "apply muted pastel color accents to facade inserts and keep primary materials in natural texture tones"

    details: list[str] = []
    ordered_slots = ["opening", "accent", "material", "aging", "context", "rhythm"]
    if isinstance(slot_map, dict) and slot_map:
        for slot in ordered_slots:
            detail_text = str(slot_map.get(slot, "")).strip()
            if detail_text and detail_text not in details:
                details.append(detail_text)

    # Ensure every selected detail phrase is assigned a color, even if it is not in slot_map.
    for chunk in str(detail_phrase or "").split(";"):
        normalized = chunk.strip()
        if normalized and normalized not in details:
            details.append(normalized)

    if not details:
        return "apply portrait-derived named pastel colors to selected details and keep structural materials naturally toned"

    placements: list[str] = []
    color_idx = 0
    for detail_text in details:
        color_name = colors[color_idx % len(colors)]
        placements.append(f"{detail_text} in {color_name}")
        color_idx += 1

    return "; ".join(placements)


def build_emotion_image_prompt(
    text_analysis: dict[str, Any],
    palette: Optional[list[str]] = None,
    portrait_top_colors: Optional[list[str]] = None,
    clothing_style_profile: Optional[dict[str, Any]] = None,
    style_notes: Optional[str] = None,
    variation_hint: Optional[str] = None,
    structure_hint: Optional[str] = None,
    text_color_hint: Optional[str] = None,
    emotion_detail_plan: Optional[dict[str, Any]] = None,
) -> str:
    emotions = _top_dominant_feelings(text_analysis, limit=3)
    themes = _top_terms(text_analysis.get("themes", []), "theme", limit=3)
    # Enhance color accuracy: use top 5 colors
    palette_values = _normalized_palette(palette, limit=5)

    theme_phrase = ", ".join(themes) if themes else "the central ideas in the text"
    palette_phrase = ", ".join(palette_values) if palette_values else "dusty blush, warm beige, chalk cream, soft sage, powder blue"
    palette_named_values = _palette_named_colors(palette_values, limit=5)
    palette_named_phrase = ", ".join(palette_named_values) if palette_named_values else palette_phrase
    portrait_top_values = _normalized_palette(portrait_top_colors, limit=5) or _normalized_palette(
        detected_clothing_palette,
        limit=5,
    )
    portrait_top_phrase = ", ".join(portrait_top_values) if portrait_top_values else palette_phrase
    portrait_named_values = _palette_named_colors(portrait_top_values, limit=5)
    portrait_named_phrase = ", ".join(portrait_named_values) if portrait_named_values else portrait_top_phrase
    theme_anchor_phrase = ", ".join(themes) if themes else "none"
    structural_signal_phrase = _emotion_collage_guidance(emotions, themes)
    theme_support_phrase = f"Support with text themes: {theme_phrase}."
    scene_guidance = str(clothing_style_profile.get("scene_guidance", "")).strip() if clothing_style_profile else ""
    clothing_label = str(clothing_style_profile.get("label", "")).strip() if clothing_style_profile else ""
    world_building_phrase = _outfit_world_building_guidance(
        clothing_style_profile,
        palette_values,
        emotions,
        themes,
    )
    style_phrase = (
        style_notes.strip()
        if style_notes and style_notes.strip()
        else "surreal editorial photomontage grounded in documentary realism"
    )
    variation_phrase = variation_hint.strip() if variation_hint else ""
    structure_phrase = structure_hint.strip() if structure_hint else ""
    text_color_phrase = text_color_hint.strip() if text_color_hint else ""
    detail_phrase = ""
    detail_slot_phrase = ""
    slot_map: dict[str, Any] = {}
    if emotion_detail_plan:
        detail_phrase = str(emotion_detail_plan.get("detail_phrase", "")).strip()
        slot_map = emotion_detail_plan.get("slot_map", {}) if isinstance(emotion_detail_plan, dict) else {}
        if isinstance(slot_map, dict) and slot_map:
            ordered_slots = []
            for slot in ["material", "opening", "aging", "context", "accent", "rhythm"]:
                value = str(slot_map.get(slot, "")).strip()
                if value:
                    ordered_slots.append(f"{slot}: {value}")
            if ordered_slots:
                detail_slot_phrase = "Analysis slot map: " + " | ".join(ordered_slots) + ". "

    clothing_scene_phrase = (
        f"Adapt the location to clothing style '{clothing_label}' with scene cues: {scene_guidance}. "
        if scene_guidance
        else ""
    )

    building_size_style = _building_size_style_phrase(clothing_style_profile, emotion_detail_plan)
    building_material_details = detail_phrase or (
        "patched matte plaster planes, recessed entry, soft patina seams, light utility conduits, and small tile accents"
    )
    color_placement_phrase = _detail_color_placement_phrase(
        slot_map,
        portrait_named_values,
        palette_named_values,
        building_material_details,
    )
    # Use color words in the prompt on a case-by-case basis
    pastel_named_phrase = ", ".join(portrait_named_values[:5]) if portrait_named_values else palette_named_phrase
    # Add a phrase to explicitly instruct the model to use these color words
    color_word_instruction = (
        f"Explicitly use these color words in the background: {', '.join([name.split(' ')[0] for name in portrait_named_values[:5]])}. "
        if portrait_named_values else ""
    )

    prompt = (
        "Everything is in focus, background sharp, clear, and detailed. "
        "The background is hyper vintage photorealistic. "
        "make it look like a photo taken on a phone. "
        "Replace only the portrait background and preserve the exact foreground subject identity, "
        "face, pose, outfit, and camera framing. "
        "Completely replace the original background; do not preserve, trace, reskin, or mimic any original background shape, line, perspective, or layout. "
        "Keep the background clean and visually consistent with the subject. "
        "Camera intent: straight-on front with sharp detail. The entire background must stay fully in focus from the nearest foreground edge to the farthest architectural detail. "
        "Photo collage style. Pastel colors. "
        f"In the background is a {building_size_style} building facade made up of {building_material_details} "
        f"with {pastel_named_phrase} on the following detail placements: {color_placement_phrase}. "
        f"{color_word_instruction}"
        f"{theme_support_phrase} "
        f"{structural_signal_phrase} "
        f"{detail_slot_phrase}"
        f"{clothing_scene_phrase}"
        f"{world_building_phrase} "
        f"{text_color_phrase} "
        f"{variation_phrase} "
        f"{structure_phrase} "
        f"Style direction: {style_phrase}. "
        f"Named palette anchors: {palette_named_phrase}. "
        f"Computed theme anchors: [{theme_anchor_phrase}]. "
        "The entire background must stay fully in focus everywhere, with crisp architectural edges and clear material texture detail. "
        "No blur anywhere in the background. "
        "realistic shadows, clean edges."
    )
    return _sanitize_prompt_language(prompt)


def _start_replace_background_job(
    subject_image_bytes: bytes,
    background_prompt: str,
    negative_prompt: str,
    background_reference_bytes: Optional[bytes] = None,
    foreground_prompt: Optional[str] = None,
    preserve_original_subject: float = 1.0,
    original_background_depth: float = 0.0,
    output_format: str = "png",
) -> str:
    if not STABILITY_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="STABILITY_API_KEY is not configured on the backend.",
        )

    endpoint = "https://api.stability.ai/v2beta/stable-image/edit/replace-background-and-relight"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "application/json",
    }
    files = {
        "subject_image": ("subject.png", subject_image_bytes, "image/png"),
    }
    if background_reference_bytes:
        files["background_reference"] = (
            "background_reference.png",
            background_reference_bytes,
            "image/png",
        )
    data = {
        "background_prompt": background_prompt,
        "negative_prompt": negative_prompt,
        "preserve_original_subject": str(max(0.0, min(1.0, preserve_original_subject))),
        "original_background_depth": str(max(0.0, min(1.0, original_background_depth))),
        "output_format": output_format,
    }
    if foreground_prompt:
        data["foreground_prompt"] = foreground_prompt

    try:
        response = requests.post(endpoint, headers=headers, files=files, data=data, timeout=90)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Stability AI replace-background request failed: {exc}",
        ) from exc

    if response.status_code != 200:
        message = response.text.strip() if response.text else "Unknown Stability AI error."
        if response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Stability AI authentication failed. "
                    "Set a valid STABILITY_API_KEY in your deployment environment and redeploy."
                ),
            )
        raise HTTPException(
            status_code=502,
            detail=(
                f"Stability AI replace-background returned {response.status_code}: "
                f"{message[:400]}"
            ),
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Stability AI response was not valid JSON for async generation start.",
        ) from exc

    generation_id = str(payload.get("id", "")).strip()
    if not generation_id:
        raise HTTPException(
            status_code=502,
            detail="Stability AI did not return a generation id.",
        )
    return generation_id


def _start_sd3_image_to_image_generation(
    subject_image_bytes: bytes,
    prompt: str,
    negative_prompt: str,
    output_format: str = "png",
    model: str = "sd3.5-large",
    strength: float = 0.42,
    cfg_scale: float = 7.0,
) -> tuple[bytes, str]:
    if not STABILITY_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="STABILITY_API_KEY is not configured on the backend.",
        )

    endpoint = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*",
    }
    files = {
        "image": ("subject.png", subject_image_bytes, "image/png"),
    }
    data = {
        "mode": "image-to-image",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model": model,
        "strength": str(max(0.0, min(1.0, strength))),
        "cfg_scale": str(max(1.0, min(10.0, cfg_scale))),
        "output_format": output_format,
    }

    try:
        response = requests.post(endpoint, headers=headers, files=files, data=data, timeout=120)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Stability AI SD3 image-to-image request failed: {exc}",
        ) from exc

    if response.status_code != 200:
        message = response.text.strip() if response.text else "Unknown Stability AI error."
        if response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Stability AI authentication failed. "
                    "Set a valid STABILITY_API_KEY in your deployment environment and redeploy."
                ),
            )
        raise HTTPException(
            status_code=502,
            detail=f"Stability AI SD3 returned {response.status_code}: {message[:400]}",
        )

    mime_type = response.headers.get("content-type", "image/png")
    return response.content, mime_type


def _prepare_subject_image_for_stability(subject_image_bytes: bytes) -> bytes:
    max_pixels = 9_437_184
    min_side = 64

    try:
        image = Image.open(io.BytesIO(subject_image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded photo is not a valid image. Please choose JPG, PNG, or WEBP.",
        ) from exc

    width, height = image.size
    if width < min_side or height < min_side:
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded photo is too small for generation. "
                "Width and height must both be at least 64 pixels."
            ),
        )

    pixel_count = width * height
    if pixel_count > max_pixels:
        scale = (max_pixels / float(pixel_count)) ** 0.5
        new_width = max(min_side, int(width * scale))
        new_height = max(min_side, int(height * scale))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _prepare_reference_image_for_stability(reference_image_bytes: bytes) -> bytes:
    max_pixels = 9_437_184
    min_side = 64

    try:
        image = Image.open(io.BytesIO(reference_image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded background reference is not a valid image. Please choose JPG, PNG, or WEBP.",
        ) from exc

    width, height = image.size
    if width < min_side or height < min_side:
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded background reference is too small for generation. "
                "Width and height must both be at least 64 pixels."
            ),
        )

    pixel_count = width * height
    if pixel_count > max_pixels:
        scale = (max_pixels / float(pixel_count)) ** 0.5
        new_width = max(min_side, int(width * scale))
        new_height = max(min_side, int(height * scale))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _pick_local_background_reference(
    clothing_style_profile: Optional[dict[str, Any]],
    text_analysis: Optional[dict[str, Any]] = None,
    subject_image_bytes: Optional[bytes] = None,
    variation_nonce: Optional[str] = None,
) -> tuple[Optional[bytes], Optional[str]]:
    if not STYLE_REFERENCES_DIR.exists() or not STYLE_REFERENCES_DIR.is_dir():
        raise HTTPException(
            status_code=500,
            detail="style-references directory does not exist or is not a directory. Please ensure the reference images are available."
        )

    # Always use Composition2.jpg as the reference image
    candidate = STYLE_REFERENCES_DIR / "Composition2.jpg"
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(
            status_code=500,
            detail="Composition2.jpg reference image is missing from style-references. Please add it."
        )
    try:
        return _prepare_reference_image_for_stability(candidate.read_bytes()), "Composition2.jpg"
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Composition2.jpg reference image is invalid or corrupted: {exc}"
        )


def _reference_palette_from_name(reference_name: Optional[str], n_colors: int = 8) -> list[str]:
    if not reference_name:
        return []
    path = STYLE_REFERENCES_DIR / reference_name
    if not path.exists() or not path.is_file():
        return []
    try:
        return extract_color_palette(path.read_bytes(), n_colors=n_colors)
    except Exception:
        return []


def _poll_stability_result(generation_id: str, timeout_seconds: int = 140) -> tuple[bytes, str]:
    endpoint = f"https://api.stability.ai/v2beta/results/{generation_id}"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "*/*",
    }

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(endpoint, headers=headers, timeout=60)
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Stability AI results polling failed: {exc}",
            ) from exc

        if response.status_code == 202:
            time.sleep(3)
            continue

        if response.status_code == 200:
            mime_type = response.headers.get("content-type", "image/png")
            return response.content, mime_type

        message = response.text.strip() if response.text else "Unknown Stability AI result error."
        if response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Stability AI authentication failed while polling results. "
                    "Set a valid STABILITY_API_KEY in your deployment environment and redeploy."
                ),
            )
        raise HTTPException(
            status_code=502,
            detail=f"Stability AI results returned {response.status_code}: {message[:400]}",
        )

    raise HTTPException(
        status_code=504,
        detail="Timed out waiting for Stability AI background replacement result.",
    )


def _parse_json_form_dict(raw: Optional[str], field_name: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in {field_name}.") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON object.")
    return parsed


def _parse_json_form_list(raw: Optional[str], field_name: str) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in {field_name}.") from exc
    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON array.")
    return [str(item) for item in parsed]


def _deterministic_prompt_nonce(
    raw_subject_image_bytes: bytes,
    parsed_text_analysis: dict[str, Any],
    palette_mode: Literal["clothing", "full"],
    style_notes: Optional[str],
) -> str:
    analysis_blob = json.dumps(parsed_text_analysis, sort_keys=True, separators=(",", ":"))
    seed = hashlib.sha256(
        raw_subject_image_bytes
        + b"|"
        + analysis_blob.encode("utf-8")
        + b"|"
        + palette_mode.encode("utf-8")
        + b"|"
        + (style_notes or "").strip().encode("utf-8")
    ).hexdigest()
    # Add runtime salt so repeated requests with similar input still produce varied prompts.
    runtime_salt = f"{time.time_ns()}|{random.getrandbits(64)}"
    salted = hashlib.sha256((seed + "|" + runtime_salt).encode("utf-8")).hexdigest()
    return salted[:16]


def _build_generation_prompt_bundle(
    raw_subject_image_bytes: bytes,
    parsed_text_analysis: dict[str, Any],
    parsed_palette: list[str],
    parsed_clothing_palette: list[str],
    parsed_full_palette: list[str],
    parsed_style_profile: dict[str, Any],
    style_notes: Optional[str],
    palette_mode: Literal["clothing", "full"],
) -> dict[str, Any]:
    detected_full_palette = extract_color_palette(raw_subject_image_bytes, n_colors=7)
    detected_clothing_palette = extract_clothing_palette(raw_subject_image_bytes, n_colors=8)

    foreground_prompt = _build_foreground_protection_prompt(parsed_style_profile)
    run_nonce = _deterministic_prompt_nonce(
        raw_subject_image_bytes,
        parsed_text_analysis,
        palette_mode,
        style_notes,
    )
    background_reference_bytes: Optional[bytes] = None
    selected_reference_name: Optional[str] = None
    reference_palette: list[str] = []
    if USE_BACKGROUND_REFERENCE_IMAGE:
        background_reference_bytes, selected_reference_name = _pick_local_background_reference(
            parsed_style_profile,
            parsed_text_analysis,
            raw_subject_image_bytes,
            variation_nonce=run_nonce,
        )
        reference_palette = _reference_palette_from_name(selected_reference_name, n_colors=8)
    text_color_palette = _extract_text_color_cues(parsed_text_analysis, limit=8)
    clothing_primary_palette = detected_clothing_palette or parsed_clothing_palette or parsed_palette
    full_primary_palette = detected_full_palette or parsed_full_palette or parsed_palette

    portrait_top_colors = _normalized_palette(parsed_clothing_palette, limit=5) or _normalized_palette(
        detected_clothing_palette,
        limit=5,
    )
    if len(portrait_top_colors) < 5:
        for color in _normalized_palette(detected_clothing_palette, limit=8):
            if color in portrait_top_colors:
                continue
            portrait_top_colors.append(color)
            if len(portrait_top_colors) >= 5:
                break
    if len(portrait_top_colors) < 5:
        for color in _normalized_palette(parsed_palette, limit=8) + _normalized_palette(parsed_full_palette, limit=8):
            if color in portrait_top_colors:
                continue
            portrait_top_colors.append(color)
            if len(portrait_top_colors) >= 5:
                break

    if palette_mode == "full":
        palette_for_prompt = _build_clothing_harmonized_pastel_palette(
            clothing_primary_palette,
            full_primary_palette,
            reference_palette,
            limit=10,
        )
    else:
        palette_for_prompt = _build_clothing_harmonized_pastel_palette(
            clothing_primary_palette,
            None,
            reference_palette,
            limit=10,
        )

    if not palette_for_prompt:
        palette_for_prompt = _merge_prompt_palette(clothing_primary_palette, reference_palette, limit=10)

    palette_for_prompt = _merge_prompt_palette(palette_for_prompt, text_color_palette, limit=10)

    variation_bank = [
        "Variant target: prioritize painted plaster blocks with tiled inserts, and keep brick as sparse accents.",
        "Variant target: prioritize stucco, concrete, and glazed tile panel rhythm, with minimal brick visibility.",
        "Variant target: prioritize mural-like painted surfaces and mixed masonry patches, with non-red brick tones only.",
        "Variant target: prioritize limewash and patterned ceramic sections, with restrained brick fragments recolored to palette.",
    ]
    variation_seed = hashlib.sha256(
        raw_subject_image_bytes
        + "|".join(palette_for_prompt).encode("utf-8")
        + run_nonce.encode("utf-8")
    ).hexdigest()
    variation_hint = variation_bank[int(variation_seed[:2], 16) % len(variation_bank)]
    emotions = _top_dominant_feelings(parsed_text_analysis, limit=5)
    themes = _top_terms(parsed_text_analysis.get("themes", []), "theme", limit=3)

    # Keep full concept strings for complex-emotion lookup (avoid splitting into isolated tokens).
    concept_candidates: list[str] = []
    for item in parsed_text_analysis.get("subject_matter_feelings", [])[:8]:
        label = str(item.get("feeling", "")).strip().lower()
        if label:
            concept_candidates.append(label)
    for item in parsed_text_analysis.get("connector_context_feelings", [])[:6]:
        label = str(item.get("feeling", "")).strip().lower()
        if label:
            concept_candidates.append(label)
    for theme_value in themes:
        label = str(theme_value).strip().lower()
        if label:
            concept_candidates.append(label)

    translation_terms: list[str] = []
    # Prioritize complex concept strings so custom mappings appear before generic base emotions.
    for candidate in concept_candidates + emotions:
        normalized = str(candidate).strip().lower()
        if not normalized or normalized in translation_terms:
            continue
        translation_terms.append(normalized)
        if len(translation_terms) >= 8:
            break
    detail_seed_basis = (
        f"{variation_seed}|"
        f"{'|'.join(emotions)}|"
        f"{'|'.join(themes)}|"
        f"{'|'.join(_normalized_palette(palette_for_prompt, limit=6))}|"
        f"{run_nonce}"
    )
    emotion_detail_plan = _select_emotion_prompt_details(
        emotions,
        themes,
        detail_seed_basis,
    )
    structure_hint = ""
    text_color_named = _palette_named_colors(text_color_palette, limit=6)
    text_color_hint = (
        f"Color extension from text cues and referenced objects (soft undersaturated pastel tones): {', '.join(text_color_named[:6])}. "
        "Use these as supporting building-surface accents while keeping clothing-harmonized colors dominant."
        if text_color_named
        else ""
    )



    # 1. Create prompt outline (fully written, human-readable, and color-name driven)
    prompt_outline = {
        "lighting": "Soft natural daylight, gentle pastel colors, subtle vintage textures, calm and slightly nostalgic mood. Layered composition with overlapping forms and a sense of depth.",
        "subject": parsed_style_profile.get("label", "the subject") if parsed_style_profile else "the subject",
        "colors": [],
        "emotion_translation": [],
        "scene_items": [],
        "motifs": [],
    }

    color_names = [entry.split(" (", 1)[0] for entry in _palette_named_colors(palette_for_prompt, limit=10)]

    # 2. Fill with analysis and color details
    fallback_objects = [
        "mask", "mirror", "hourglass", "veil", "laurel wreath", "musical instrument", "book", "candle", "key", "locket", "scroll", "quill", "lantern", "ribbon", "coin", "feather", "rose", "handkerchief", "gemstone", "chess piece", "letter", "ring", "shell", "star", "apple", "cup", "bottle"
    ]
    used_symbols = set()
    fallback_idx = 0
    material_options = ["polished ceramic", "brushed metal", "smooth glass", "natural wood", "soft fabric", "sculpted stone", "painted clay", "etched crystal"]
    placement_options = [
        "on a small pedestal at the center foreground",
        "on a low table to the left",
        "on a shelf to the right",
        "hanging from a thin wire in the background",
        "resting on a folded cloth near the subject",
        "leaning against a textured wall",
        "partially hidden behind a translucent screen",
        "arranged in a cluster with other objects"
    ]
    rng_seed = hashlib.sha256(f"{run_nonce}|{variation_seed}".encode("utf-8")).hexdigest()
    rng = random.Random(int(rng_seed[:16], 16))

    manifestation_items: list[str] = []
    for idx, word in enumerate(translation_terms):
        details = get_details_for_complex_word(word)
        color = color_names[idx % len(color_names)] if color_names else "soft neutral"
        prompt_outline["colors"].append(f"Emotion '{word}' is represented with {color} accents.")
        used_this_emotion = False
        if details:
            # Use the first available symbol, motif, and art for maximum clarity
            symbol_candidates = [s for s in details.get("symbols", []) if s not in used_symbols]
            symbol = rng.choice(symbol_candidates) if symbol_candidates else None
            if symbol:
                material = material_options[idx % len(material_options)]
                placement = placement_options[idx % len(placement_options)]
                prompt_outline["emotion_translation"].append(
                    f"Distinct emotional translation: {word} becomes a tangible {symbol} in {color} tones."
                )
                prompt_outline["scene_items"].append(
                    f"A {symbol} made of {material}, colored {color}, is {placement}, representing {word}. The object is depicted with realistic texture and lighting.")
                used_symbols.add(symbol)
                manifestation_items.append(symbol)
                used_this_emotion = True
            trope_candidates = details.get("tropes", [])
            trope = rng.choice(trope_candidates) if trope_candidates else None
            if trope:
                placement = placement_options[(idx+2) % len(placement_options)]
                prompt_outline["motifs"].append(
                    f"A prop illustrating the motif '{trope}' in {color}, {placement}, with visible material and clear placement.")
                used_this_emotion = True
            art_candidates = details.get("art", [])
            art = rng.choice(art_candidates) if art_candidates else None
            if art:
                prompt_outline["motifs"].append(
                    f"Background treatment: {art}, rendered with matching palette and lighting.")
                used_this_emotion = True
        if not used_this_emotion:
            fallback_obj = fallback_objects[fallback_idx % len(fallback_objects)]
            material = material_options[(idx+3) % len(material_options)]
            placement = placement_options[(idx+4) % len(placement_options)]
            prompt_outline["emotion_translation"].append(
                f"Distinct emotional translation: {word} becomes a symbolic {fallback_obj} in {color} tones."
            )
            prompt_outline["scene_items"].append(
                f"A {fallback_obj} made of {material}, colored {color}, is {placement}, representing {word}. The object is depicted with realistic texture and lighting.")
            manifestation_items.append(fallback_obj)
            fallback_idx += 1

    # 3. Verify all required fields are present (at least one scene item per emotion)
    if not prompt_outline["scene_items"]:
        fallback_term = translation_terms[0] if translation_terms else (emotions[0] if emotions else "internal signal")
        fallback_color = color_names[0] if color_names else "muted neutral"
        prompt_outline["emotion_translation"].append(
            f"Distinct emotional translation: {fallback_term} becomes a symbolic lantern in {fallback_color} tones."
        )
        prompt_outline["scene_items"].append(
            f"A lantern made of brushed metal, colored {fallback_color}, is on a low table to the left, representing {fallback_term}. The object is depicted with realistic texture and lighting."
        )
        prompt_outline["motifs"].append(
            f"A prop illustrating the motif 'emergence from uncertainty' in {fallback_color}, leaning against a textured wall, with visible material and clear placement."
        )
        manifestation_items.append("lantern")

    # Use clothing-first anchors so prompt colors reflect the uploaded portrait more directly.
    anchor_palette_source = (
        _normalized_palette(portrait_top_colors, limit=8)
        or _normalized_palette(detected_clothing_palette, limit=8)
        or _normalized_palette(parsed_clothing_palette, limit=8)
        or _normalized_palette(palette_for_prompt, limit=8)
    )
    anchor_named_colors = [entry.split(" (", 1)[0] for entry in _palette_named_colors(anchor_palette_source, limit=8)]
    palette_anchor_names = anchor_named_colors[:8] if anchor_named_colors else ["soft neutral"]
    palette_phrase = ", ".join(palette_anchor_names[:4])
    detail_phrase = str((emotion_detail_plan or {}).get("detail_phrase", "")).strip() if isinstance(emotion_detail_plan, dict) else ""
    slot_map = emotion_detail_plan.get("slot_map", {}) if isinstance(emotion_detail_plan, dict) else {}
    slot_bits: list[str] = []
    for slot in ["material", "opening", "aging", "context", "accent", "rhythm"]:
        value = str(slot_map.get(slot, "")).strip() if isinstance(slot_map, dict) else ""
        if value:
            slot_bits.append(value)
    materials_phrase = "; ".join(slot_bits) if slot_bits else (detail_phrase or "weathered plaster, ceramic tile trims, painted metal seams")

    def _pick_manifest(index: int, fallback: str) -> str:
        if manifestation_items and len(manifestation_items) > index:
            return manifestation_items[index]
        if manifestation_items:
            return manifestation_items[index % len(manifestation_items)]
        return fallback

    def _pick_color(index: int) -> str:
        if palette_anchor_names:
            return palette_anchor_names[index % len(palette_anchor_names)]
        return "soft neutral"

    dominant = " ".join(emotions).lower()
    if any(token in dominant for token in ["fear", "anger", "panic", "grief", "tension"]):
        scene_attribute = "tense and uneasy"
        sky_phrase = "not visible"
        lighting_level = "grey"
    elif any(token in dominant for token in ["joy", "awe", "hope", "confidence", "wonder"]):
        scene_attribute = "uplifting and alive"
        sky_phrase = "visible"
        lighting_level = "light"
    else:
        scene_attribute = "quiet and reflective"
        sky_phrase = "partially visible"
        lighting_level = "light"

    primary_manifest = _pick_manifest(0, "lantern")
    right_manifest = _pick_manifest(1, "mask")
    right_base = _pick_manifest(2, "pedestal")
    dangling_manifest = _pick_manifest(3, "ribbon")
    growing_manifest = _pick_manifest(4, "vine")
    circling_manifest = _pick_manifest(5, "swallow")
    fallen_manifest = _pick_manifest(6, "column")
    contrast_left = _pick_manifest(7, "statue")
    contrast_right = _pick_manifest(8, "mirror")
    foreground_manifest = _pick_manifest(9, "lantern")
    emotional_theme = ", ".join(themes[:2]) if themes else (", ".join(emotions[:2]) if emotions else "emotional duality")

    # 4. Compose the final prompt string using the user-provided outline.
    contrast_left_position = "at the midground entry on the left"
    sky_sentence = f" the sky is {sky_phrase}." if sky_phrase in {"visible", "not visible"} else ""
    style_guide_suffix = (
        "Photo-collage poetic urban surrealism. Ground the scene in a real neighborhood facade with layered material zones: "
        "faded plaster, patched brick, painted concrete, tile bands, corrugated metal, mixed windows/doors, laundry lines, "
        "exposed cables, and light scaffolding. Keep everything crisp and readable, with intentional collage seams and believable "
        "texture detail. Embed theatrical symbolic manifestations into the architecture with dreamlike scale shifts, balancing "
        "tenderness, unease, and mythic memory. Palette is soft, dusty, understated pastels derived from the portrait clothing "
        "colors, with only a few restrained accent pops for emotional emphasis. Keep the composition coherent and editorial, never "
        "chaotic or generic."
    )

    prompt = (
        "Replace only the portrait background and preserve the exact foreground subject identity, face, pose, outfit, and camera framing. "
        "Show exactly one singular building in the background as a medium-wide collaged-style scene. "
        "Photo collage style. Pastel colors. "
        f"In the background is a large {primary_manifest} that is {palette_phrase} on the building facade made up of {materials_phrase} with {_pick_color(0)} on the {slot_bits[0] if slot_bits else 'primary facade seams'}. "
        f"On the right of the screen is {right_manifest} placed on a {right_base} with {dangling_manifest} dangling into the frame. "
        f"{growing_manifest} {_pick_color(1)} grows from beneath as {circling_manifest} is circling the shape of a fallen {fallen_manifest}. "
        f"The scene is {scene_attribute}.{sky_sentence} "
        f"The lighting of the scene is {lighting_level}. "
        "Relight the scene so the subject matches the background. "
        f"There is a {contrast_left} {contrast_left_position} and a {contrast_right} beside it, contrasting {emotional_theme}. "
        f"In the foreground is a {foreground_manifest}, it is {_pick_color(2)}. "
        "Camera intent: straight-on front (roughly 20mm full-frame). "
        f"{style_guide_suffix}"
    )

    negative_prompt = (
        "do not alter subject identity, do not change face, do not change body, do not change outfit, "
        "no duplicate subject, no extra people, no subject warping, no deformed anatomy, "
        "no subject relighting, no background lighting mismatch, no inconsistent light direction, no inconsistent shadow direction, "
        "no mismatched color temperature, no mismatched exposure between subject and background, "
        "do not preserve original background structure, do not copy original wall lines, no original background geometry reuse, "
        "no multiple buildings, no building clusters, no city skyline rows, "
        "no corner-building angle, no oblique facade, no diagonal camera slant, no vanishing-point street canyon, "
        "no text, no logo, no watermark, no blurry background, no blur, no depth blur, no lens blur, no gaussian blur, no motion blur, no bokeh, "
        "no close wall directly behind subject, no generic flat single-wall backdrop"
    )

    return {
        "detected_full_palette": detected_full_palette,
        "detected_clothing_palette": detected_clothing_palette,
        "foreground_prompt": foreground_prompt,
        "run_nonce": run_nonce,
        "background_reference_bytes": background_reference_bytes,
        "selected_reference_name": selected_reference_name,
        "text_color_palette": text_color_palette,
        "portrait_top_colors": portrait_top_colors,
        "palette_for_prompt": palette_for_prompt,
        "variation_hint": variation_hint,
        "emotions": emotions,
        "themes": themes,
        "emotion_detail_plan": emotion_detail_plan,
        "structure_hint": structure_hint,
        "text_color_hint": text_color_hint,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "colors": prompt_outline["colors"],
        "emotion_translation": prompt_outline["emotion_translation"],
        "scene_items": prompt_outline["scene_items"],
        "motifs": prompt_outline["motifs"],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze-text")
def analyze_text(payload: TextRequest) -> dict[str, Any]:
    return {"text_analysis": analyze_text_content(payload.text)}


@app.post("/analyze-photo")
async def analyze_photo(photo: UploadFile = File(...)) -> dict[str, Any]:
    contents = await photo.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    full_palette = extract_color_palette(contents, n_colors=7)
    clothing_palette = extract_clothing_palette(contents, n_colors=8)
    clothing_style_profile = analyze_clothing_style(contents)
    return {
        "filename": photo.filename,
        "palette": clothing_palette,
        "clothing_palette": clothing_palette,
        "full_image_palette": full_palette,
        "clothing_style_profile": clothing_style_profile,
        "notes": "Palette now prioritizes clothing colors via torso + skin-filter heuristic.",
    }


@app.post("/analyze")
async def analyze(text: str = Form(...), photo: UploadFile = File(...)) -> dict[str, Any]:
    contents = await photo.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    text_analysis = analyze_text_content(text)
    full_palette = extract_color_palette(contents, n_colors=7)
    clothing_palette = extract_clothing_palette(contents, n_colors=8)
    clothing_style_profile = analyze_clothing_style(contents)
    prompt_bundle = _build_generation_prompt_bundle(
        raw_subject_image_bytes=contents,
        parsed_text_analysis=text_analysis,
        parsed_palette=clothing_palette,
        parsed_clothing_palette=clothing_palette,
        parsed_full_palette=full_palette,
        parsed_style_profile=clothing_style_profile,
        style_notes=None,
        palette_mode="clothing",
    )
    prompt = prompt_bundle["prompt"]

    return {
        "text_analysis": text_analysis,
        "image_analysis": {
            "filename": photo.filename,
            "palette": clothing_palette,
            "clothing_palette": clothing_palette,
            "full_image_palette": full_palette,
            "clothing_style_profile": clothing_style_profile,
            "background_prompt": prompt,
            "prompt_used_preview": prompt,
            "status": "ready_for_generation",
        },
    }


@app.post("/generate-emotion-image")
async def generate_emotion_image(
    photo: UploadFile = File(...),
    text_analysis: str = Form(...),
    palette: Optional[str] = Form(None),
    clothing_palette: Optional[str] = Form(None),
    full_image_palette: Optional[str] = Form(None),
    palette_mode: Literal["clothing", "full"] = Form("clothing"),
    generation_backend: Literal["replace_background_and_relight", "sd3_image_to_image"] = Form("sd3_image_to_image"),
    clothing_style_profile: Optional[str] = Form(None),
    style_notes: Optional[str] = Form(None),
) -> dict[str, Any]:
    raw_subject_image_bytes = await photo.read()
    if not raw_subject_image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded photo is empty.")
    subject_image_bytes = _prepare_subject_image_for_stability(raw_subject_image_bytes)

    parsed_text_analysis = _parse_json_form_dict(text_analysis, "text_analysis")
    parsed_palette = _parse_json_form_list(palette, "palette")
    parsed_clothing_palette = _parse_json_form_list(clothing_palette, "clothing_palette")
    parsed_full_palette = _parse_json_form_list(full_image_palette, "full_image_palette")
    parsed_style_profile = _parse_json_form_dict(clothing_style_profile, "clothing_style_profile")
    prompt_bundle = _build_generation_prompt_bundle(
        raw_subject_image_bytes=raw_subject_image_bytes,
        parsed_text_analysis=parsed_text_analysis,
        parsed_palette=parsed_palette,
        parsed_clothing_palette=parsed_clothing_palette,
        parsed_full_palette=parsed_full_palette,
        parsed_style_profile=parsed_style_profile,
        style_notes=style_notes,
        palette_mode=palette_mode,
    )

    detected_full_palette = prompt_bundle["detected_full_palette"]
    detected_clothing_palette = prompt_bundle["detected_clothing_palette"]
    foreground_prompt = prompt_bundle["foreground_prompt"]
    run_nonce = prompt_bundle["run_nonce"]
    background_reference_bytes = prompt_bundle["background_reference_bytes"]
    selected_reference_name = prompt_bundle["selected_reference_name"]
    text_color_palette = prompt_bundle["text_color_palette"]
    portrait_top_colors = prompt_bundle["portrait_top_colors"]
    palette_for_prompt = prompt_bundle["palette_for_prompt"]
    variation_hint = prompt_bundle["variation_hint"]
    emotions = prompt_bundle["emotions"]
    themes = prompt_bundle["themes"]
    emotion_detail_plan = prompt_bundle["emotion_detail_plan"]
    structure_hint = prompt_bundle["structure_hint"]
    text_color_hint = prompt_bundle["text_color_hint"]
    prompt = prompt_bundle["prompt"]
    negative_prompt = prompt_bundle["negative_prompt"]
    # Prompt already includes full case-specific template completion.
    generation_id: Optional[str] = None
    if generation_backend == "sd3_image_to_image":
        image_bytes, mime_type = _start_sd3_image_to_image_generation(
            subject_image_bytes,
            prompt,
            negative_prompt,
            output_format="png",
            model="sd3.5-large",
            strength=0.42,
            cfg_scale=7.0,
        )
    else:
        generation_id = _start_replace_background_job(
            subject_image_bytes,
            prompt,
            negative_prompt,
            background_reference_bytes=background_reference_bytes,
            foreground_prompt=foreground_prompt,
            preserve_original_subject=0.9,
            original_background_depth=0.0,
            output_format="png",
        )
        image_bytes, mime_type = _poll_stability_result(generation_id)

    image_data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    palette_used = _normalized_palette(palette_for_prompt)
    portrait_top_color_names = _palette_named_colors(portrait_top_colors, limit=5)

    return {
        "image_data_url": image_data_url,
        "prompt_used": prompt,
        "emotions": emotions,
        "themes": themes,
        "palette_used": palette_used,
        "portrait_top_colors": portrait_top_colors,
        "portrait_top_color_names": portrait_top_color_names,
        "text_color_palette": text_color_palette,
        "detected_clothing_palette": _normalized_palette(detected_clothing_palette, limit=8),
        "detected_full_image_palette": _normalized_palette(detected_full_palette, limit=8),
        "palette_strategy": "clothing-anchored pastel pairing",
        "variation_hint_used": variation_hint,
        "structure_hint_used": structure_hint,
        "palette_mode": palette_mode,
        "clothing_style_profile": parsed_style_profile,
        "reference_image_used": bool(background_reference_bytes),
        "reference_image_name": selected_reference_name,
        "reference_source": (
            f"style-references folder (fixed: {selected_reference_name})"
            if selected_reference_name
            else "disabled"
        ),
        "generation_backend": generation_backend,
        "prompt_source": "generated_bundle",
        "prompt_inputs": {
            "text_emotions": emotions,
            "text_themes": themes,
            "portrait_top_colors": portrait_top_colors,
            "portrait_top_color_names": portrait_top_color_names,
            "palette_for_prompt": palette_used,
            "text_color_palette": text_color_palette,
            "variation_hint": variation_hint,
            "structure_hint": structure_hint,
            "text_color_hint": text_color_hint,
            "emotion_detail_plan": {
                "emotion_class": emotion_detail_plan.get("emotion_class"),
                "selected_detail_ids": emotion_detail_plan.get("selected_detail_ids", []),
                "slot_map": emotion_detail_plan.get("slot_map", {}),
                "detail_phrase": emotion_detail_plan.get("detail_phrase", ""),
            },
            "run_nonce": run_nonce,
        },
        "generation_id": generation_id,
        "status": "generated",
    }


@app.get("/", include_in_schema=False)
def serve_frontend_root():
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "ok", "message": "Frontend build not found. Build frontend/dist to serve UI from this backend."}


@app.get("/{path:path}", include_in_schema=False)
def serve_frontend_path(path: str):
    # Keep API/docs routes handled by their dedicated endpoints.
    protected_prefixes = (
        "health",
        "analyze",
        "analyze-text",
        "analyze-photo",
        "generate-emotion-image",
        "docs",
        "redoc",
        "openapi.json",
    )
    if path.startswith(protected_prefixes):
        raise HTTPException(status_code=404, detail="Not found")

    candidate = FRONTEND_DIST_DIR / path
    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)

    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    raise HTTPException(status_code=404, detail="Not found")


if FRONTEND_DIST_DIR.exists() and (FRONTEND_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets"), name="frontend-assets")
