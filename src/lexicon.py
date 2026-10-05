"""Domain-knowledge lexicon for alcohol references.

The lexicon was written from general domain knowledge (drink categories, grape varieties,
spirit types, cocktails, major brands) *before* looking at the OCR output of the training
set, so it should generalise to unseen beverage terms/brands the training data lacks.
Error analysis on the training OCR text added matching *rules*, not vocabulary: the rn->m
OCR-confusion canonicalisation, the NEAR_MISS_WORDS guard, the exact-alcohol-word guard for
fuzzy decoys ("cocktail" vs "mocktails"), and TEMPLATE_WORDS (used by the text model).

Matching rules (robust to OCR noise without exploding false positives):
* Text is normalised (lowercase, de-leet, letters only) first - see text_normalize.py.
* Hard-negative phrases ("root beer", "ginger beer", "non alcoholic", ...) are matched and
  *removed* before alcohol terms are searched, so "root beer" never fires "beer".
* Single-word terms match tokens with a length-aware Levenshtein budget:
      len <= 5 : exact only        (gin, rum, ale, port, wine, sake, lager ...)
      len 6-8  : <= 1 edit         (bourbon~bourbcn, whisky~whiskv)
      len >= 9 : <= 2 edits        (chardonnay, champagne, distillery)
  Short words must be exact because 1 edit would turn "port"->"post", "wine"->"wind",
  "gin"->"gig", "lager"->"later".
* Long terms (>= 6 chars) are also searched in the space-less string, which recovers words
  OCR split in two ("bour bon", "chardon nay") or glued together ("bourbonevent").
* A hit preceded within 3 tokens by a negator (non, no, zero, free, virgin...) is counted as
  negated instead of positive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein

from .text_normalize import dedupe_tokens, normalize, ocr_canon, spaceless

ALCOHOL_TERMS: dict[str, list[str]] = {
    "beer": [
        "beer", "beers", "lager", "lagers", "ale", "ales", "ipa", "stout", "porter", "pilsner",
        "pilsener", "pils", "bock", "hefeweizen", "weissbier", "witbier", "saison", "lambic",
        "gose", "kolsch", "dunkel", "microbrew", "microbrewery", "brewery", "breweries",
        "brewpub", "craft beer", "pale ale", "on tap", "draught", "draft beer", "keg", "kegs",
        "pint", "pints", "growler", "sixpack", "six pack", "tallboy",
    ],
    "wine": [
        "wine", "wines", "winery", "wineries", "vino", "vineyard", "merlot", "cabernet",
        "sauvignon", "chardonnay", "pinot", "pinot noir", "grigio", "riesling", "malbec", "shiraz",
        "syrah", "zinfandel", "tempranillo", "sangiovese", "chianti", "rioja", "bordeaux",
        "burgundy", "beaujolais", "moscato", "prosecco", "champagne", "cava", "lambrusco",
        "sparkling wine", "port", "sherry", "madeira", "marsala", "vermouth", "sommelier",
        "brut", "bubbly", "vintage wine", "wine tasting",
    ],
    "spirits": [
        "whiskey", "whiskeys", "whisky", "whiskies", "bourbon", "scotch", "rye whiskey",
        "single malt", "vodka", "gin", "rum", "tequila", "mezcal", "brandy", "cognac",
        "armagnac", "absinthe", "liqueur", "liquor", "liquors", "schnapps", "grappa", "ouzo",
        "sake", "soju", "baijiu", "pisco", "cachaca", "amaretto", "limoncello", "triple sec",
        "spirits", "distillery", "moonshine", "hard seltzer", "hard cider", "cider", "mead",
        "jagermeister", "kahlua", "baileys", "aperol", "campari",
    ],
    "cocktail": [
        "cocktail", "cocktails", "margarita", "margaritas", "mojito", "mojitos", "martini",
        "martinis", "daiquiri", "cosmopolitan", "negroni", "manhattan cocktail", "old fashioned",
        "mimosa", "mimosas", "bellini", "sangria", "spritz", "pina colada", "bloody mary",
        "mai tai", "highball", "boilermaker", "punch bowl",
    ],
    "general": [
        "alcohol", "alcoholic", "booze", "boozy", "liquor store", "happy hour", "open bar",
        "pub crawl", "bar crawl", "beer garden", "wine bar", "taproom", "tipsy", "drunk",
        "hangover", "brewski", "bartender", "mixology", "on the rocks", "nightcap",
    ],
    "brand": [
        "heineken", "budweiser", "bud light", "corona", "guinness", "stella artois", "coors",
        "miller lite", "michelob", "carlsberg", "tuborg", "becks", "peroni", "modelo",
        "asahi", "sapporo", "kirin", "tsingtao", "amstel", "hoegaarden", "leffe", "pabst",
        "blue moon", "samuel adams", "sierra nevada", "jack daniels", "jim beam", "jameson",
        "johnnie walker", "chivas", "glenfiddich", "glenlivet", "macallan", "makers mark",
        "wild turkey", "smirnoff", "absolut", "grey goose", "ketel one", "belvedere",
        "bacardi", "captain morgan", "malibu rum", "havana club", "jose cuervo", "patron tequila",
        "don julio", "hennessy", "remy martin", "courvoisier", "moet", "veuve clicquot",
        "dom perignon", "tanqueray", "bombay sapphire", "beefeater", "hendricks",
        "smirnoff ice", "white claw", "strongbow", "angry orchard",
    ],
}

# Beverages / substances that are *not* alcohol - frequent hard negatives in ad text.
NEGATIVE_TERMS = [
    "root beer", "ginger beer", "ginger ale", "non alcoholic", "nonalcoholic", "alcohol free",
    "alcoholfree", "zero alcohol", "zero proof", "mocktail", "mocktails", "virgin",
    "soft drink", "soda", "cola", "lemonade", "juice", "smoothie", "milkshake", "coffee",
    "espresso", "latte", "cappuccino", "tea", "matcha", "kombucha", "energy drink",
    "sports drink", "water", "sparkling water", "seltzer water", "milk", "hot chocolate",
    "cocoa", "cannabis", "weed", "marijuana", "cbd", "thc", "vape", "tobacco", "cigar",
    "cigarette", "protein shake", "iced tea", "boba", "bubble tea", "chai",
]

# Marketing / call-to-action template words. They carry no information about the beverage, yet
# the generator uses some of them at slightly different rates per class (e.g. "dont miss": 11% of
# alcohol vs 4% of non-alcohol texts; "enjoy"/"looking for"/"experience" only in non-alcohol).
# A text model would learn these as spurious cues ("dont miss" alone -> p=0.98), so they are
# removed before TF-IDF vectorisation; the model can then only learn from content words.
TEMPLATE_WORDS = {
    "dont", "don", "do", "miss", "check", "out", "try", "get", "ready", "for", "grab", "now",
    "limited", "edition", "premium", "special", "imported", "import", "flavor", "flavour",
    "taste", "discount", "offer", "menu", "reserve", "event", "events", "explore", "nightlife",
    "night", "celebration", "celebrate", "enjoy", "looking", "experience", "smooth", "new",
    "the", "a", "an", "to", "at", "of", "our", "your", "and", "with", "in", "on", "this", "da",
    "best", "only", "today", "tonight", "deal", "deals", "sale", "free", "fresh", "exclusive",
}

# Real English words within the fuzzy-edit budget of an alcohol term; never fuzzy-match them.
NEAR_MISS_WORDS = {"ported", "sported", "sorted", "porta", "potter", "poster", "pinto", "pilot",
                   "whisker", "whiskers", "brandt", "lagged", "cinder", "rummy",
                   "sherri", "merlin"}

NEGATORS = {"non", "no", "zero", "free", "virgin", "without", "sober"}



def _max_edits(term: str) -> int:
    n = len(term)
    return 0 if n <= 5 else (1 if n <= 8 else 2)


@dataclass
class LexiconMatch:
    """Result of matching one text against the lexicon."""
    alcohol: list[str] = field(default_factory=list)   # matched alcohol terms
    negative: list[str] = field(default_factory=list)  # matched decoy terms
    negated: list[str] = field(default_factory=list)   # alcohol terms preceded by a negator


class Lexicon:
    """Matches OCR text against the alcohol terms, decoys and negators defined above."""
    def __init__(self):
        self.alcohol_terms = [t for terms in ALCOHOL_TERMS.values() for t in terms]
        self.alcohol_words = {w for t in self.alcohol_terms for w in t.split()}

    @staticmethod
    def _find_phrase(tokens: list[str], phrase: str) -> list[tuple[int, int, bool]]:
        """(start, end, exact) spans where `phrase` matches the token list."""
        ptoks = phrase.split()
        out = []
        for i in range(len(tokens) - len(ptoks) + 1):
            exact = True
            for t, p in zip(tokens[i:i + len(ptoks)], ptoks):
                if t == p:
                    continue
                exact = False
                if len(p) >= 5 and ocr_canon(t) == ocr_canon(p):  # rn->m style OCR confusion
                    continue
                budget = _max_edits(p)
                if t in NEAR_MISS_WORDS or budget == 0 or abs(len(t) - len(p)) > budget                         or Levenshtein.distance(t, p, score_cutoff=budget) > budget:
                    break
            else:
                out.append((i, i + len(ptoks), exact))
        return out

    def match(self, text: str) -> LexiconMatch:
        """Find decoys, then non-negated alcohol terms, in normalised OCR text."""
        tokens = dedupe_tokens(normalize(text)).split()
        m = LexiconMatch()
        used = [False] * len(tokens)

        # 1) decoys first; their tokens are consumed so "root beer" never fires "beer"
        for phrase in sorted(NEGATIVE_TERMS, key=len, reverse=True):
            for s, e, exact in self._find_phrase(tokens, phrase):
                # a fuzzy decoy must not swallow an exact alcohol word ("cocktail" ~ "mocktails")
                if not exact and any(t in self.alcohol_words for t in tokens[s:e]):
                    continue
                if not any(used[s:e]):
                    m.negative.append(phrase)
                    used[s:e] = [True] * (e - s)

        free = [t if not u else "#" for t, u in zip(tokens, used)]
        flat = spaceless(" ".join(t for t in free if t != "#"))
        for phrase in NEGATIVE_TERMS:  # glued decoys ("cannabisspecial")
            p = spaceless(phrase)
            if len(p) >= 6 and phrase not in m.negative and p in flat:
                m.negative.append(phrase)

        # 2) alcohol terms on the remaining tokens
        for term in self.alcohol_terms:
            spans = self._find_phrase(free, term)
            found = False
            for s, _, _ in spans:
                if any(t in NEGATORS for t in tokens[max(0, s - 3):s]):  # "non alcoholic beer"
                    m.negated.append(term)
                else:
                    found = True
            if not spans and len(spaceless(term)) >= 6:  # glued or split words ("bourbonevent")
                st = spaceless(term)
                found = st in flat or ocr_canon(st) in ocr_canon(flat)
            if found:
                m.alcohol.append(term)
        return m
