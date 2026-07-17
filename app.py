# ============================================================
# CUSTOMER REVIEW INTELLIGENCE APP - PROFESSIONAL FINAL VERSION
# Consumer + Seller / Marketing Team modes
# ============================================================

import html
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# 0. Visual theme (palette + plotly styling)
# ============================================================

PALETTE = {
    "ink": "#172033",
    "muted": "#667085",
    "consumer": "#5FAFA3",
    "consumer_soft": "#9ED8CF",
    "seller": "#7394D8",
    "seller_soft": "#B9C9EC",
    "positive": "#62B89F",
    "neutral": "#E7B96A",
    "negative": "#D98B8B",
    "grid": "#E7ECF3",
}

# Soft, semantic colors for sentiment charts (covers common label variants).
SENTIMENT_COLORS = {
    "Positive": PALETTE["positive"], "positive": PALETTE["positive"],
    "POSITIVE": PALETTE["positive"], "2": PALETTE["positive"],
    "Neutral": PALETTE["neutral"], "neutral": PALETTE["neutral"],
    "NEUTRAL": PALETTE["neutral"], "1": PALETTE["neutral"],
    "Negative": PALETTE["negative"], "negative": PALETTE["negative"],
    "NEGATIVE": PALETTE["negative"], "0": PALETTE["negative"],
}

# A light, airy plotly template applied to every figure in the app.
pio.templates["review_soft"] = pio.templates["plotly_white"]
_tmpl = pio.templates["review_soft"]
_tmpl.layout.font.family = "Inter, sans-serif"
_tmpl.layout.font.color = PALETTE["ink"]
_tmpl.layout.font.size = 13
_tmpl.layout.paper_bgcolor = "rgba(0,0,0,0)"
_tmpl.layout.plot_bgcolor = "rgba(0,0,0,0)"
_tmpl.layout.colorway = [
    PALETTE["seller"], PALETTE["consumer"], PALETTE["seller_soft"],
    PALETTE["consumer_soft"], PALETTE["neutral"], PALETTE["negative"],
]
_tmpl.layout.title.font.family = "Plus Jakarta Sans, sans-serif"
_tmpl.layout.title.font.size = 17
_tmpl.layout.title.font.color = PALETTE["ink"]
_tmpl.layout.margin = dict(l=60, r=30, t=60, b=50)
_tmpl.layout.xaxis.gridcolor = PALETTE["grid"]
_tmpl.layout.yaxis.gridcolor = PALETTE["grid"]
_tmpl.layout.xaxis.linecolor = PALETTE["grid"]
_tmpl.layout.yaxis.linecolor = PALETTE["grid"]
pio.templates.default = "review_soft"
px.defaults.template = "review_soft"


# ============================================================
# 1. Paths and configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "app"

REVIEWS_PATH = DATA_DIR / "app_reviews_sample.csv"
CLUSTER_PATH = DATA_DIR / "category_cluster_df.csv"
CLUSTER_PROFILE_PATH = DATA_DIR / "category_cluster_profile_df.csv"
GENERATED_ARTICLES_PATH = DATA_DIR / "generated_articles.csv"

# Deployment preference:
# DistilBERT first because it is cheaper/faster for inference.
# RoBERTa remains as fallback if DistilBERT is not available.
MODEL_CANDIDATES = [
    PROJECT_DIR / "outputs" / "distilbert_sentiment_model",
    PROJECT_DIR / "outputs" / "baseline_distilbert",
    PROJECT_DIR / "outputs" / "distilbert_sentiment_model_v2",
    PROJECT_DIR / "outputs" / "roberta_sentiment_model_v2",
    PROJECT_DIR / "outputs" / "roberta_sentiment_model",
]

DEFAULT_SAMPLE_ROWS = 30000
MAX_SUMMARY_REVIEWS = 80
MAX_REVIEW_CHARS_FOR_SUMMARY = 280


# ============================================================
# 2. Utility functions
# ============================================================

def find_existing_model_path() -> Optional[Path]:
    """Return the first complete local Hugging Face classifier directory."""
    weight_files = ("model.safetensors", "pytorch_model.bin")
    for path in MODEL_CANDIDATES:
        if not path.exists() or not (path / "config.json").exists():
            continue
        if any((path / filename).exists() for filename in weight_files):
            return path
    return None


def read_csv_safe(path: Path, required: bool = True) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.DataFrame()


def normalize_colname(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def build_column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return {normalize_colname(c): c for c in df.columns}


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lookup = build_column_lookup(df)
    for candidate in candidates:
        key = normalize_colname(candidate)
        if key in lookup:
            return lookup[key]
    return None


def coerce_percent_series(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    if len(values.dropna()) > 0 and values.dropna().max() <= 1.0:
        values = values * 100
    return values


def short_list_text(value, max_items: int = 12) -> str:
    if pd.isna(value):
        return "Not available"
    text = str(value)
    text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    parts = [p.strip() for p in re.split(r",|;|\|", text) if p.strip()]
    if not parts:
        return text
    return ", ".join(parts[:max_items])


def format_pct(value) -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "N/A"


def format_number(value) -> str:
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return "N/A"


def safe_unique(df: pd.DataFrame, col: Optional[str]) -> List[str]:
    if col is None or col not in df.columns:
        return []
    values = df[col].dropna().astype(str).unique().tolist()
    return sorted(values)


# ============================================================
# 3. Load data
# ============================================================

reviews_df = read_csv_safe(REVIEWS_PATH, required=True)
cluster_df = read_csv_safe(CLUSTER_PATH, required=True)
cluster_profile_df = read_csv_safe(CLUSTER_PROFILE_PATH, required=True)
articles_df = read_csv_safe(GENERATED_ARTICLES_PATH, required=False)

if len(reviews_df) > DEFAULT_SAMPLE_ROWS:
    reviews_df = reviews_df.sample(DEFAULT_SAMPLE_ROWS, random_state=42).reset_index(drop=True)

review_text_col = find_column(reviews_df, ["review_text", "reviews_text", "text", "review", "content", "reviews.text", "review_body", "body"])
sentiment_col = find_column(reviews_df, ["sentiment", "label", "sentiment_label", "target", "class"])
rating_col = find_column(reviews_df, ["rating", "reviews_rating", "reviews.rating", "stars", "score"])
category_col = find_column(reviews_df, ["category", "categories", "primary_category", "raw_category", "product_category", "main_category"])
product_col = find_column(reviews_df, ["product_name", "product.name", "product_title", "product", "name"])

cluster_meta_name_col = find_column(cluster_df, ["meta_category_name", "meta_category", "cluster_name"])
cluster_id_col = find_column(cluster_df, ["meta_cluster", "cluster", "cluster_id"])
cluster_raw_category_col = find_column(cluster_df, ["raw_category", "category", "categories", "product_category"])
cluster_num_reviews_col = find_column(cluster_df, ["num_reviews", "total_reviews", "review_count", "count"])
cluster_pos_col = find_column(cluster_df, ["positive_pct", "positive_percentage", "pos_pct"])
cluster_neu_col = find_column(cluster_df, ["neutral_pct", "neutral_percentage", "neu_pct"])
cluster_neg_col = find_column(cluster_df, ["negative_pct", "negative_percentage", "neg_pct"])

profile_meta_name_col = find_column(cluster_profile_df, ["meta_category_name", "meta_category", "cluster_name"])
profile_cluster_id_col = find_column(cluster_profile_df, ["meta_cluster", "cluster", "cluster_id"])
profile_categories_col = find_column(cluster_profile_df, ["categories", "raw_categories", "category_list"])
profile_keywords_col = find_column(cluster_profile_df, ["keywords", "top_keywords", "terms"])
profile_total_reviews_col = find_column(cluster_profile_df, ["total_reviews", "num_reviews", "review_count", "count"])
profile_num_categories_col = find_column(cluster_profile_df, ["num_categories", "category_count"])
profile_pos_col = find_column(cluster_profile_df, ["avg_positive_pct", "positive_pct", "positive_percentage"])
profile_neu_col = find_column(cluster_profile_df, ["avg_neutral_pct", "neutral_pct", "neutral_percentage"])
profile_neg_col = find_column(cluster_profile_df, ["avg_negative_pct", "negative_pct", "negative_percentage"])
profile_top_products_col = find_column(cluster_profile_df, ["top_products", "products", "product_names"])

article_meta_name_col = find_column(articles_df, ["meta_category_name", "meta_category", "cluster_name"])
article_cluster_id_col = find_column(articles_df, ["meta_cluster", "cluster", "cluster_id"])
article_text_col = find_column(articles_df, ["article", "summary", "generated_article", "summary_text"])
article_top_products_col = find_column(articles_df, ["top_products", "products"])
article_worst_product_col = find_column(articles_df, ["worst_product", "product_to_avoid", "avoid_product"])
article_num_reviews_col = find_column(articles_df, ["num_reviews", "total_reviews", "review_count"])
article_pos_col = find_column(articles_df, ["positive_pct", "positive_percentage"])
article_neu_col = find_column(articles_df, ["neutral_pct", "neutral_percentage"])
article_neg_col = find_column(articles_df, ["negative_pct", "negative_percentage"])
article_positive_summary_col = find_column(articles_df, ["positive_summary", "what_customers_value", "strengths_summary"])
article_neutral_summary_col = find_column(articles_df, ["neutral_summary", "mixed_feedback", "mixed_summary"])
article_negative_summary_col = find_column(articles_df, ["negative_summary", "common_concerns", "concerns_summary"])
article_positive_source_col = find_column(articles_df, ["positive_summary_source", "positive_source"])
article_neutral_source_col = find_column(articles_df, ["neutral_summary_source", "neutral_source"])
article_negative_source_col = find_column(articles_df, ["negative_summary_source", "negative_source"])

if cluster_meta_name_col is None and cluster_id_col is not None:
    cluster_df["meta_category_name"] = cluster_df[cluster_id_col].apply(lambda x: f"Meta-category {x}")
    cluster_meta_name_col = "meta_category_name"

if profile_meta_name_col is None and profile_cluster_id_col is not None:
    cluster_profile_df["meta_category_name"] = cluster_profile_df[profile_cluster_id_col].apply(lambda x: f"Meta-category {x}")
    profile_meta_name_col = "meta_category_name"


# ============================================================
# 4. Load sentiment model
# ============================================================

model_path = find_existing_model_path()
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

if model_path is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model.to(device)
        model.eval()
        MODEL_STATUS = f"Loaded model: {model_path.relative_to(PROJECT_DIR)} on {device.upper()}"
    except Exception as e:
        MODEL_STATUS = f"Model path found but failed to load: {model_path}. Error: {e}"
        tokenizer = None
        model = None
else:
    MODEL_STATUS = "No local fine-tuned model found. Sentiment classifier will use a simple rule-based fallback."

DEFAULT_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


def get_label_map() -> Dict[int, str]:
    if model is None:
        return DEFAULT_LABEL_MAP
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) > 0:
        mapped = {}
        for k, v in id2label.items():
            try:
                idx = int(k)
            except Exception:
                idx = k
            label = str(v).replace("LABEL_0", "Negative").replace("LABEL_1", "Neutral").replace("LABEL_2", "Positive")
            low = label.lower()
            if "neg" in low or label == "0":
                label = "Negative"
            elif "neu" in low or label == "1":
                label = "Neutral"
            elif "pos" in low or label == "2":
                label = "Positive"
            mapped[idx] = label
        return mapped
    return DEFAULT_LABEL_MAP


LABEL_MAP = get_label_map()


# ============================================================
# 5. Analytics helpers
# ============================================================

def get_meta_categories() -> List[str]:
    values = safe_unique(cluster_profile_df, profile_meta_name_col)
    if not values:
        values = safe_unique(cluster_df, cluster_meta_name_col)
    return values


META_CATEGORIES = get_meta_categories()


def raw_category_count() -> int:
    if cluster_raw_category_col:
        return len(safe_unique(cluster_df, cluster_raw_category_col))
    if category_col:
        return len(safe_unique(reviews_df, category_col))
    return 0


def _split_category_labels(value: str) -> List[str]:
    """
    Split noisy marketplace category paths into smaller labels.
    This helps avoid broad labels such as 'Computers & Tablets' pulling
    tablet reviews into a generic computer group.
    """

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    parts = [
        p.strip()
        for p in re.split(r"\||;|,|>|/", text)
        if p and p.strip()
    ]

    return parts if parts else [text]


def _canonical_consumer_group(raw_category: str) -> Optional[str]:
    """
    Convert noisy marketplace taxonomy labels into consumer-facing product groups.

    Rules:
    - Specific product/accessory families win over broad marketplace categories.
    - Broad labels such as 'Electronics' are dropped.
    - Broad computer labels are kept as a clean PC/laptop/office group.
    - If a row contains both a broad computer label and a specific tablet/Kindle
      product name, the specific tablet/Kindle rule wins.
    """

    labels = _split_category_labels(raw_category)
    if not labels:
        return None

    low = " ".join(labels).lower()
    original_low = str(raw_category).strip().lower()

    # Exact broad/noisy marketplace labels.
    # Some are dropped; some are kept as clean consumer-facing groups.
    drop_exact = {
        "aa",
        "frys",
        "mazon.co.uk",
        "amazon.co.uk",
        "featured brands",
        "electronics features",
        "walmart for business",
        "consumer electronics",
        "electronics",
    }

    broad_computer_exact = {
        "computers & tablets",
        "computers & accessories",
        "computers/tablets & networking",
        "computers, office & electronics",
        "office",
    }

    # --------------------------------------------------------
    # Specific accessory groups first.
    # A Kindle charger should be a charger, not an e-reader.
    # A tablet case should be an accessory, not a tablet.
    # --------------------------------------------------------
    if any(k in low for k in ["battery", "batteries"]):
        return "Batteries"

    if any(k in low for k in ["charger", "charging", "power adapter", "adapter", "usb", "cable"]):
        return "Chargers, Cables & Power Adapters"

    # --------------------------------------------------------
    # Specific device families.
    # This catches product names such as Fire HD Tablet / Kindle Paperwhite.
    # Exact broad labels like 'Computers & Tablets' are handled later, so they
    # do not automatically make every computer-category item a tablet.
    # --------------------------------------------------------
    exact_broad_computer = original_low in broad_computer_exact

    tablet_terms = ["kindle", "e-reader", "ereader", "ebook", "e-book", "fire tablet", "tablet"]
    if not exact_broad_computer and any(k in low for k in tablet_terms):
        return "Tablets, Kindle & E-readers"

    if any(k in low for k in ["fire tv", "streaming", "tv", "television", "stereo", "entertainment", "echo", "alexa", "smart"]):
        return "Streaming, Echo & Home Entertainment"

    # --------------------------------------------------------
    # PC / laptop / office group.
    # This is intentionally separate from tablets/e-readers.
    # --------------------------------------------------------
    pc_terms = [
        "computer", "computers", "laptop", "pc", "notebook", "keyboard", "mouse",
        "monitor", "printer", "office product", "office products", "networking"
    ]
    if exact_broad_computer or any(k in low for k in pc_terms):
        return "Computers, Laptops & Office Accessories"

    # Generic accessories after PC rules, so laptop/computer stands stay with PCs.
    if any(k in low for k in ["case", "cover", "stand", "accessor", "peripheral", "bag", "leather"]):
        return "Cases, Covers, Stands & Accessories"

    if any(k in low for k in ["pet", "crate", "dog", "cat", "litter"]):
        return "Pet Supplies"

    # Drop generic marketplace navigation labels after specific checks.
    if original_low in drop_exact:
        return None

    return "Other Amazon Devices & Accessories"

def build_consumer_category_map() -> Dict[str, List[str]]:
    """
    Build a dropdown map: clean consumer label -> original raw categories.
    The original categories are kept for filtering/recommendation logic.
    """

    raw_categories = []
    if cluster_raw_category_col:
        raw_categories = safe_unique(cluster_df, cluster_raw_category_col)
    elif category_col:
        raw_categories = safe_unique(reviews_df, category_col)

    grouped: Dict[str, List[str]] = {}
    for raw in raw_categories:
        group = _canonical_consumer_group(raw)
        if group is None:
            continue
        grouped.setdefault(group, []).append(raw)

    # Sort original categories inside each group for reproducibility.
    return {group: sorted(set(values)) for group, values in sorted(grouped.items())}


RAW_CATEGORY_COUNT = raw_category_count()
CONSUMER_CATEGORY_MAP = build_consumer_category_map()


def consumer_category_options() -> List[str]:
    return list(CONSUMER_CATEGORY_MAP.keys())


CATEGORY_OPTIONS = consumer_category_options()



def get_raw_categories_for_consumer_option(selected_category: str) -> List[str]:
    if not selected_category:
        return []
    return CONSUMER_CATEGORY_MAP.get(str(selected_category), [str(selected_category)])


def _normalise_taxonomy_label(value: str) -> str:
    """Normalise a taxonomy label for exact, reproducible matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


RAW_CATEGORY_GROUP_BY_NORM: Dict[str, str] = {}
for _group_name, _raw_values in CONSUMER_CATEGORY_MAP.items():
    for _raw_value in _raw_values:
        _norm = _normalise_taxonomy_label(_raw_value)
        if _norm:
            RAW_CATEGORY_GROUP_BY_NORM[_norm] = _group_name


def _category_segments(value: str) -> List[str]:
    """Return normalised category-path segments and the complete path."""
    raw = str(value)
    if not raw or raw.lower() == "nan":
        return []

    segments = [
        _normalise_taxonomy_label(part)
        for part in re.split(r"\||;|,|>|/|\n", raw)
    ]
    segments = [segment for segment in segments if segment]

    complete = _normalise_taxonomy_label(raw)
    if complete and complete not in segments:
        segments.append(complete)

    return list(dict.fromkeys(segments))


def _known_groups_for_category_value(value: str) -> List[str]:
    """
    Map a review category to known consumer groups using exported cluster
    categories. Exact path segments are preferred over fuzzy substring matching.
    """
    segments = _category_segments(value)
    if not segments:
        return []

    matches: List[str] = []

    for segment in segments:
        exact_group = RAW_CATEGORY_GROUP_BY_NORM.get(segment)
        if exact_group:
            matches.append(exact_group)

    if not matches:
        complete = _normalise_taxonomy_label(value)
        for raw_norm, group in RAW_CATEGORY_GROUP_BY_NORM.items():
            # Avoid matching very short/generic labels accidentally.
            if len(raw_norm) >= 5 and (
                complete == raw_norm
                or complete.startswith(raw_norm + " ")
                or complete.endswith(" " + raw_norm)
                or f" {raw_norm} " in f" {complete} "
            ):
                matches.append(group)

    return list(dict.fromkeys(matches))


def _consumer_group_for_review_row(row: pd.Series) -> Optional[str]:
    """
    Resolve a review to one consumer-facing group.

    Category taxonomy is the primary source. Product name is used only to
    disambiguate multiple category matches or when the category is generic.
    Review titles and review text are never used as product identifiers.
    """
    category_value = str(row.get(category_col, "")) if category_col else ""
    groups = _known_groups_for_category_value(category_value)

    product_value = ""
    if product_col and product_col in row.index:
        product_value = str(row.get(product_col, ""))

    product_group = _canonical_consumer_group(product_value) if product_value else None

    if len(groups) == 1:
        return groups[0]

    if len(groups) > 1:
        if product_group in groups:
            return product_group
        # Prefer the most specific non-catch-all group.
        specific = [group for group in groups if not group.startswith("Other ")]
        return specific[0] if specific else groups[0]

    # No exported category match: classify the category itself.
    category_group = _canonical_consumer_group(category_value)
    if category_group:
        return category_group

    # Use product name only as a final fallback for generic/missing taxonomies.
    return product_group


def _row_matches_raw_categories(row: pd.Series, raw_categories: List[str]) -> bool:
    if not category_col or category_col not in row.index:
        return False

    raw_value = row.get(category_col, "")
    segments = set(_category_segments(raw_value))
    complete = _normalise_taxonomy_label(raw_value)

    if not segments and not complete:
        return False

    target_norms = {
        _normalise_taxonomy_label(category)
        for category in raw_categories
        if _normalise_taxonomy_label(category)
    }

    if segments.intersection(target_norms):
        return True

    for target in target_norms:
        if len(target) >= 5 and (
            complete == target
            or complete.startswith(target + " ")
            or complete.endswith(" " + target)
            or f" {target} " in f" {complete} "
        ):
            return True

    return False


def get_categories_for_meta(meta_category: str) -> List[str]:
    if not meta_category:
        return []
    if cluster_meta_name_col and cluster_raw_category_col:
        sub = cluster_df[
            cluster_df[cluster_meta_name_col].astype(str) == str(meta_category)
        ]
        return safe_unique(sub, cluster_raw_category_col)
    return []


def get_reviews_for_category_or_meta(
    category: Optional[str] = None,
    meta_category: Optional[str] = None,
    limit: int = 200,
) -> pd.DataFrame:
    """
    Filter reviews using exact exported taxonomy labels.

    This avoids broad regex matching such as "tablet" or "electronics" pulling
    unrelated products into the selected category/meta-category.
    """
    df = reviews_df.copy()

    if category and category_col:
        target = _normalise_taxonomy_label(category)
        mask = df[category_col].apply(
            lambda value: target in set(_category_segments(value))
        )
        df = df[mask]

    if meta_category and category_col:
        raw_categories = get_categories_for_meta(meta_category)
        if raw_categories:
            mask = df.apply(
                lambda row: _row_matches_raw_categories(row, raw_categories),
                axis=1,
            )
            df = df[mask]

    if len(df) > limit:
        df = df.sample(limit, random_state=42)

    return df.reset_index(drop=True)


def get_reviews_for_consumer_option(
    selected_category: Optional[str],
    limit: int = 400,
) -> pd.DataFrame:
    """
    Filter reviews by an exact consumer-facing product group.

    Every review is mapped from its exported category taxonomy. Product name is
    used only as a controlled fallback, preventing Kindle/e-reader reviews from
    appearing under chargers merely because of a noisy title or broad category.
    """
    if (
        not selected_category
        or category_col is None
        or category_col not in reviews_df.columns
    ):
        return pd.DataFrame()

    df = reviews_df.copy()
    group_series = df.apply(_consumer_group_for_review_row, axis=1)
    df = df[group_series.astype(str) == str(selected_category)]

    if len(df) > limit:
        df = df.sample(limit, random_state=42)

    return df.reset_index(drop=True)


def get_cluster_rows_for_consumer_option(
    selected_category: Optional[str],
) -> pd.DataFrame:
    """Return cluster rows mapped to the selected consumer product group."""
    if not selected_category or not cluster_raw_category_col:
        return pd.DataFrame()

    mask = cluster_df[cluster_raw_category_col].apply(
        lambda value: _canonical_consumer_group(value) == selected_category
    )
    return cluster_df[mask].copy()


def weighted_cluster_percent(cluster_rows: pd.DataFrame, percent_col: Optional[str], weight_col: Optional[str]) -> float:
    if cluster_rows.empty or percent_col is None or percent_col not in cluster_rows.columns:
        return np.nan

    values = coerce_percent_series(cluster_rows[percent_col])

    if weight_col and weight_col in cluster_rows.columns:
        weights = pd.to_numeric(cluster_rows[weight_col], errors="coerce").fillna(1)
    else:
        weights = pd.Series(np.ones(len(cluster_rows)), index=cluster_rows.index)

    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan

    return float(np.average(values[valid], weights=weights[valid]))


def sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"sentiment": [], "count": []})
    if sentiment_col and sentiment_col in df.columns:
        temp = df[sentiment_col].astype(str).value_counts().reset_index()
        temp.columns = ["sentiment", "count"]
        return temp
    if rating_col and rating_col in df.columns:
        ratings = pd.to_numeric(df[rating_col], errors="coerce")
        labels = np.where(ratings >= 4, "Positive", np.where(ratings <= 2, "Negative", "Neutral"))
        temp = pd.Series(labels).value_counts().reset_index()
        temp.columns = ["sentiment", "count"]
        return temp
    return pd.DataFrame({"sentiment": [], "count": []})


def generate_buy_recommendation(positive_pct, neutral_pct, negative_pct, avg_rating=None) -> str:
    positive_pct = float(positive_pct) if pd.notna(positive_pct) else 0
    negative_pct = float(negative_pct) if pd.notna(negative_pct) else 0
    rating_text = ""
    if avg_rating is not None and pd.notna(avg_rating):
        rating_text = f" Average rating is around {float(avg_rating):.2f}/5."
    if positive_pct >= 70 and negative_pct <= 15:
        return "Strong buy signal: most customers report positive experiences." + rating_text
    if positive_pct >= 55 and negative_pct <= 25:
        return "Generally safe choice: feedback is mostly positive, but check the recurring complaints before buying." + rating_text
    if negative_pct >= 35:
        return "High risk: this category/product has a relevant share of negative feedback. Compare alternatives carefully." + rating_text
    return "Mixed feedback: customer opinion is not clearly positive or negative. Compare alternatives before buying." + rating_text



def extract_common_terms(texts: List[str], top_n: int = 12) -> List[str]:
    stopwords = {
        "the", "and", "for", "this", "that", "with", "was", "were", "are", "you", "have", "has", "had", "but", "not", "very", "from",
        "they", "them", "its", "it", "just", "product", "item", "one", "use", "used", "get", "got", "can", "will", "would", "also",
        "really", "much", "more", "than", "been", "good", "great", "like", "well", "when", "what", "your", "about", "there", "their"
    }
    counts = {}
    for text in texts:
        words = re.findall(r"[a-zA-Z]{3,}", str(text).lower())
        for w in words:
            if w not in stopwords:
                counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def simple_feedback_summary(df: pd.DataFrame, context: str = "selected data") -> str:
    if df.empty:
        return "No reviews available for this selection."
    text_values = []
    if review_text_col and review_text_col in df.columns:
        text_values = df[review_text_col].dropna().astype(str).head(MAX_SUMMARY_REVIEWS).str.slice(0, MAX_REVIEW_CHARS_FOR_SUMMARY).tolist()
    dist = sentiment_distribution(df)
    total = int(dist["count"].sum()) if not dist.empty else len(df)
    sentiment_text = "Sentiment information is not available."
    if not dist.empty and total > 0:
        parts = []
        for _, row in dist.iterrows():
            pct = 100 * row["count"] / total
            parts.append(f"{row['sentiment']}: {pct:.1f}%")
        sentiment_text = "; ".join(parts)
    avg_rating = None
    if rating_col and rating_col in df.columns:
        avg_rating = pd.to_numeric(df[rating_col], errors="coerce").mean()
    terms = extract_common_terms(text_values, top_n=10) if text_values else []
    rating_line = f"\n- Average rating: {avg_rating:.2f}/5" if avg_rating is not None and pd.notna(avg_rating) else ""
    examples = ""
    if text_values:
        examples = "\n\nRepresentative review snippets:\n"
        for t in text_values[:3]:
            examples += f"- {t[:220]}...\n"
    return f"""### Feedback summary for {context}

- Reviews analyzed: {len(df):,}
- Sentiment mix: {sentiment_text}{rating_line}
- Frequent terms: {", ".join(terms) if terms else "Not available"}

Interpretation:
Customers in this selection show the sentiment pattern above. Frequent terms should be used as signals for the main topics customers mention, but they should be interpreted together with the original review examples.

{examples}
"""


# ============================================================
# 6. Sentiment prediction
# ============================================================

def rule_based_sentiment(text: str) -> Dict[str, float]:
    text_l = text.lower()
    positive_words = ["good", "great", "excellent", "amazing", "perfect", "love", "loved", "easy", "useful", "recommend", "best", "works", "quality", "happy"]
    negative_words = ["bad", "terrible", "awful", "broken", "poor", "waste", "hate", "disappointed", "return", "worst", "defective", "problem", "issue", "doesn't", "not work", "stopped"]
    pos = sum(1 for w in positive_words if w in text_l)
    neg = sum(1 for w in negative_words if w in text_l)
    if pos > neg:
        return {"Positive": 0.70, "Neutral": 0.20, "Negative": 0.10}
    if neg > pos:
        return {"Negative": 0.70, "Neutral": 0.20, "Positive": 0.10}
    return {"Neutral": 0.60, "Positive": 0.20, "Negative": 0.20}


def predict_sentiment(review_text: str) -> Tuple[Dict[str, float], str]:
    if not review_text or not review_text.strip():
        return {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0}, "Write or paste a review first."
    if model is None or tokenizer is None:
        probs = rule_based_sentiment(review_text)
        pred = max(probs, key=probs.get)
        return probs, f"Predicted sentiment: {pred}. Note: model fallback is rule-based because no local model was loaded."
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs_tensor = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()
    probs = {}
    for i, p in enumerate(probs_tensor):
        label = LABEL_MAP.get(i, f"Label {i}")
        probs[label] = float(p)
    pred = max(probs, key=probs.get)
    confidence = probs[pred] * 100
    if pred.lower().startswith("negative"):
        action = "Suggested action: inspect the review for complaints and consider customer support follow-up."
    elif pred.lower().startswith("neutral"):
        action = "Suggested action: review this as mixed or low-intensity feedback; look for improvement opportunities."
    else:
        action = "Suggested action: this can support marketing claims if the review is representative and authentic."
    explanation = f"Predicted sentiment: {pred} ({confidence:.1f}% confidence).\n\n{action}"
    return probs, explanation


# ============================================================
# 7. Plot functions
# ============================================================

def plot_overall_sentiment():
    dist = sentiment_distribution(reviews_df)
    if dist.empty:
        return px.bar(title="Sentiment distribution unavailable")
    return px.bar(dist, x="sentiment", y="count", title="Overall Sentiment Distribution", labels={"sentiment": "Sentiment", "count": "Number of reviews"}, color="sentiment", color_discrete_map=SENTIMENT_COLORS).update_layout(showlegend=False)


def plot_reviews_by_meta_category():
    if profile_meta_name_col and profile_total_reviews_col:
        temp = cluster_profile_df[[profile_meta_name_col, profile_total_reviews_col]].copy()
        temp[profile_total_reviews_col] = pd.to_numeric(temp[profile_total_reviews_col], errors="coerce").fillna(0)
        temp = temp.sort_values(profile_total_reviews_col, ascending=True)
        return px.bar(temp, x=profile_total_reviews_col, y=profile_meta_name_col, orientation="h", title="Reviews by Meta-Category", labels={profile_total_reviews_col: "Reviews", profile_meta_name_col: "Meta-category"})
    return px.bar(title="Reviews by meta-category unavailable")


def plot_negative_rate_by_meta_category():
    if profile_meta_name_col and profile_neg_col:
        temp = cluster_profile_df[[profile_meta_name_col, profile_neg_col]].copy()
        temp[profile_neg_col] = coerce_percent_series(temp[profile_neg_col])
        temp = temp.sort_values(profile_neg_col, ascending=True)
        return px.bar(temp, x=profile_neg_col, y=profile_meta_name_col, orientation="h", title="Negative Review Rate by Meta-Category", labels={profile_neg_col: "Negative reviews (%)", profile_meta_name_col: "Meta-category"})
    return px.bar(title="Negative review rate unavailable")



def plot_cluster_3d(meta_category: Optional[str] = None):
    """Show the complete semantic map and visually emphasise the selection."""
    required = {"x", "y", "z"}
    if not required.issubset(cluster_df.columns):
        fig = px.scatter_3d(
            title="3D cluster plot unavailable: x/y/z columns were not exported"
        )
        fig.update_layout(height=560)
        return fig

    temp = cluster_df.copy()
    selected = str(meta_category).strip() if meta_category else None

    color_col = cluster_meta_name_col if cluster_meta_name_col else cluster_id_col
    hover_col = cluster_raw_category_col if cluster_raw_category_col else None
    size_col = cluster_num_reviews_col if cluster_num_reviews_col else None

    if selected and cluster_meta_name_col:
        temp["_selection"] = np.where(
            temp[cluster_meta_name_col].astype(str) == selected,
            selected,
            "Other meta-categories",
        )
        color_col = "_selection"
        color_map = {
            selected: PALETTE["seller"],
            "Other meta-categories": "#CBD5E1",
        }
        title = f"Semantic category map — {selected}"
    else:
        color_map = None
        title = "Semantic category map"

    fig = px.scatter_3d(
        temp,
        x="x",
        y="y",
        z="z",
        color=color_col,
        size=size_col,
        hover_name=hover_col,
        title=title,
        color_discrete_map=color_map,
    )

    fig.update_traces(
        marker=dict(opacity=0.88, line=dict(width=0.4, color="#FFFFFF"))
    )
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=58, b=10),
        scene=dict(
            xaxis_title="Semantic dimension 1",
            yaxis_title="Semantic dimension 2",
            zaxis_title="Semantic dimension 3",
            bgcolor="rgba(0,0,0,0)",
        ),
        legend_title_text="Meta-category",
    )
    return fig


# ============================================================
# 8. Consumer functions
# ============================================================

THEME_STOPWORDS = {
    "amazon", "kindle", "product", "products", "item", "items", "review",
    "reviews", "customer", "customers", "good", "great", "nice", "love",
    "loved", "really", "just", "very", "well", "works", "work", "use",
    "used", "using", "bought", "buy", "purchase", "purchased", "would",
    "could", "also", "one", "much", "get", "got", "thing", "things",
    "excellent", "perfect", "amazing", "happy", "stopped", "little",
    "better", "best", "worst", "awesome", "okay", "ok",
}


def _sentiment_rates_for_df(df: pd.DataFrame) -> Dict[str, float]:
    rates = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    dist = sentiment_distribution(df)
    total = float(dist["count"].sum()) if not dist.empty else 0.0

    for _, row in dist.iterrows():
        label = str(row["sentiment"]).lower()
        pct = 100.0 * float(row["count"]) / total if total else 0.0
        if "pos" in label or label in {"2", "positive"}:
            rates["positive"] = pct
        elif "neu" in label or label in {"1", "neutral"}:
            rates["neutral"] = pct
        elif "neg" in label or label in {"0", "negative"}:
            rates["negative"] = pct

    return rates


def _review_sentiment_labels(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")

    if sentiment_col and sentiment_col in df.columns:
        def canonical(value):
            text = str(value).lower()
            if "pos" in text or text == "2":
                return "Positive"
            if "neu" in text or text == "1":
                return "Neutral"
            if "neg" in text or text == "0":
                return "Negative"
            return None
        return df[sentiment_col].map(canonical)

    if rating_col and rating_col in df.columns:
        ratings = pd.to_numeric(df[rating_col], errors="coerce")
        return pd.Series(
            np.where(
                ratings >= 4,
                "Positive",
                np.where(ratings <= 2, "Negative", "Neutral"),
            ),
            index=df.index,
        )

    return pd.Series([None] * len(df), index=df.index)


CONSUMER_THEME_TAXONOMY = {
    "battery life": {
        "patterns": (r"\bbattery\b", r"\bbatteries\b"),
        "roles": {"value", "concern"},
    },
    "charging speed and reliability": {
        "patterns": (r"\bcharg(?:e|ed|er|ers|ing)\b", r"\bpower adapter\b"),
        "roles": {"value", "concern"},
    },
    "screen quality and readability": {
        "patterns": (r"\bscreen\b", r"\bdisplay\b", r"\breadab(?:le|ility)\b", r"\bglare\b"),
        "roles": {"value", "concern"},
    },
    "ease of use and setup": {
        "patterns": (r"\beasy to use\b", r"\beasy setup\b", r"\bset up\b", r"\bsetup\b", r"\bintuitive\b"),
        "roles": {"value", "concern"},
    },
    "speed and responsiveness": {
        "patterns": (r"\bspeed\b", r"\bslow\b", r"\bfast\b", r"\bresponsiv(?:e|eness)\b", r"\blag(?:gy|ging)?\b"),
        "roles": {"value", "concern"},
    },
    "sound quality and volume": {
        "patterns": (r"\bsound\b", r"\baudio\b", r"\bspeaker(?:s)?\b", r"\bvolume\b"),
        "roles": {"value", "concern"},
    },
    "connectivity and voice response": {
        "patterns": (r"\bwi-?fi\b", r"\bbluetooth\b", r"\bconnect(?:ion|ivity|ed|ing)?\b", r"\bvoice\b", r"\balexa\b"),
        "roles": {"value", "concern"},
    },
    "build quality and durability": {
        "patterns": (r"\bbuild quality\b", r"\bdurab(?:le|ility)\b", r"\bsturdy\b", r"\bflimsy\b", r"\bbrok(?:e|en)\b"),
        "roles": {"value", "concern"},
    },
    "case protection and fit": {
        "patterns": (r"\bprotective case\b", r"\bcase\b", r"\bcover\b", r"\bstand\b", r"\bprotection\b"),
        "roles": {"value", "concern"},
    },
    "compatibility and fit": {
        "patterns": (r"\bcompatib(?:le|ility)\b", r"\bworks with\b", r"\bdoes not fit\b", r"\bdoesn't fit\b"),
        "roles": {"value", "concern"},
    },
    "value for money": {
        "patterns": (r"\bvalue for money\b", r"\bgood value\b", r"\bworth (?:it|the money)\b", r"\baffordable\b", r"\boverpriced\b"),
        "roles": {"value", "concern"},
    },
    "software and app usability": {
        "patterns": (r"\bsoftware\b", r"\bapp(?:s)?\b", r"\binterface\b", r"\bnavigation\b"),
        "roles": {"value", "concern"},
    },
    "advertising and bloatware": {
        "patterns": (r"\bad(?:s|vertising)?\b", r"\bbloatware\b", r"\bspecial offers\b"),
        "roles": {"concern"},
    },
    "storage capacity": {
        "patterns": (r"\bstorage\b", r"\bmemory\b", r"\bfree space\b", r"\bcapacity\b"),
        "roles": {"value", "concern"},
    },
    "streaming quality and content access": {
        "patterns": (r"\bstream(?:ing|s)?\b", r"\bmovie(?:s)?\b", r"\blive tv\b", r"\bchannel(?:s)?\b"),
        "roles": {"value", "concern"},
    },
    "device reliability and stability": {
        "patterns": (r"\breliable\b", r"\bunreliable\b", r"\bfreeze(?:s|ing)?\b", r"\bcrash(?:es|ed|ing)?\b", r"\bstopped working\b", r"\bdefective\b"),
        "roles": {"value", "concern"},
    },
    "portability and size": {
        "patterns": (r"\bportable\b", r"\blightweight\b", r"\bcompact\b", r"\bsize\b", r"\bweight\b"),
        "roles": {"value", "concern"},
    },
}


def _extract_actionable_themes(
    texts: List[str],
    top_n: int = 4,
    role: str = "value",
) -> List[str]:
    """Return recurring, business-readable themes from a controlled taxonomy."""
    clean_texts = [
        re.sub(r"\s+", " ", str(text).lower()).strip()
        for text in texts
        if str(text).strip()
    ]
    if not clean_texts:
        return []

    minimum_mentions = 2 if len(clean_texts) >= 8 else 1
    scored = []

    for label, config in CONSUMER_THEME_TAXONOMY.items():
        if role not in config["roles"]:
            continue

        mentions = sum(
            any(re.search(pattern, text) for pattern in config["patterns"])
            for text in clean_texts
        )
        if mentions < minimum_mentions:
            continue

        coverage = mentions / len(clean_texts)
        scored.append((mentions + coverage, mentions, label))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [label for _, _, label in scored[:top_n]]

def _theme_lists_for_reviews(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    if df.empty or not review_text_col or review_text_col not in df.columns:
        return [], []

    labels = _review_sentiment_labels(df)
    text_series = df[review_text_col].fillna("").astype(str).str.strip()

    positive_texts = text_series[labels == "Positive"].tolist()
    concern_texts = text_series[labels.isin(["Negative", "Neutral"])].tolist()

    values = _extract_actionable_themes(
        positive_texts,
        top_n=4,
        role="value",
    )
    concerns = _extract_actionable_themes(
        concern_texts,
        top_n=4,
        role="concern",
    )

    return values, concerns

def _representative_review_rows(df: pd.DataFrame, limit: int = 2) -> List[str]:
    """Choose concise, non-duplicate evidence across sentiment classes."""
    if df.empty or not review_text_col or review_text_col not in df.columns:
        return []

    temp = df.copy()
    temp["_sentiment_label"] = _review_sentiment_labels(temp)
    temp["_review_text"] = temp[review_text_col].fillna("").astype(str).str.strip()
    temp = temp[temp["_review_text"].str.len().between(35, 700)]
    temp = temp.drop_duplicates("_review_text")

    selected: List[str] = []
    for label in ["Positive", "Negative", "Neutral"]:
        subset = temp[temp["_sentiment_label"] == label]
        if not subset.empty:
            selected.append(subset.iloc[0]["_review_text"])
        if len(selected) >= limit:
            return selected

    for text in temp["_review_text"].tolist():
        if text not in selected:
            selected.append(text)
        if len(selected) >= limit:
            break

    return selected


def _sentiment_bar_html(
    positive: float,
    neutral: float,
    negative: float,
    compact: bool = False,
) -> str:
    positive = max(0.0, float(positive or 0.0))
    neutral = max(0.0, float(neutral or 0.0))
    negative = max(0.0, float(negative or 0.0))
    total = positive + neutral + negative

    if total <= 0:
        positive = neutral = negative = 0.0
    elif abs(total - 100.0) > 0.15:
        positive = 100.0 * positive / total
        neutral = 100.0 * neutral / total
        negative = 100.0 * negative / total

    compact_class = " sentiment-compact" if compact else ""
    return f"""
<div class="sentiment-distribution{compact_class}">
  <div class="sentiment-track" aria-label="Sentiment distribution">
    <span class="sentiment-positive" style="width:{positive:.2f}%"></span>
    <span class="sentiment-neutral" style="width:{neutral:.2f}%"></span>
    <span class="sentiment-negative" style="width:{negative:.2f}%"></span>
  </div>
  <div class="sentiment-labels">
    <span><i class="dot-positive"></i><strong>{positive:.1f}%</strong> Positive</span>
    <span><i class="dot-neutral"></i><strong>{neutral:.1f}%</strong> Neutral</span>
    <span><i class="dot-negative"></i><strong>{negative:.1f}%</strong> Negative</span>
  </div>
</div>
"""


def _theme_list_html(items: List[str], empty_text: str) -> str:
    if not items:
        return f'<div class="empty-inline">{html.escape(empty_text)}</div>'
    return "<ul class='theme-list'>" + "".join(
        f"<li>{html.escape(str(item).replace('_', ' ').title())}</li>"
        for item in items
    ) + "</ul>"


def _plain_list_html(items: List[str], empty_text: str) -> str:
    """Render product names without changing brand capitalization."""
    if not items:
        return f'<div class="empty-inline">{html.escape(empty_text)}</div>'
    return "<ul class='theme-list'>" + "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in items
    ) + "</ul>"


def consumer_product_check(selected_category: str) -> str:
    """Return a concise product-group performance view for shoppers."""
    if not selected_category:
        return '<div class="empty-state">Select a product group first.</div>'

    selected_reviews = get_reviews_for_consumer_option(
        selected_category,
        limit=900,
    )
    cluster_rows = get_cluster_rows_for_consumer_option(selected_category)

    if selected_reviews.empty and cluster_rows.empty:
        return (
            '<div class="empty-state">'
            'No review data was found for this product group.'
            '</div>'
        )

    # Review-level evidence is preferred because it keeps ratings, examples,
    # sentiment and topics aligned to the same filtered records.
    rates = _sentiment_rates_for_df(selected_reviews)

    if selected_reviews.empty and not cluster_rows.empty:
        rates = {
            "positive": weighted_cluster_percent(
                cluster_rows, cluster_pos_col, cluster_num_reviews_col
            ),
            "neutral": weighted_cluster_percent(
                cluster_rows, cluster_neu_col, cluster_num_reviews_col
            ),
            "negative": weighted_cluster_percent(
                cluster_rows, cluster_neg_col, cluster_num_reviews_col
            ),
        }

    rates = {
        key: float(value) if pd.notna(value) else 0.0
        for key, value in rates.items()
    }

    avg_rating = np.nan
    if rating_col and not selected_reviews.empty:
        avg_rating = pd.to_numeric(
            selected_reviews[rating_col],
            errors="coerce",
        ).mean()

    n_reviews = len(selected_reviews)
    if n_reviews == 0 and not cluster_rows.empty:
        if cluster_num_reviews_col:
            n_reviews = pd.to_numeric(
                cluster_rows[cluster_num_reviews_col],
                errors="coerce",
            ).fillna(0).sum()
        else:
            n_reviews = len(cluster_rows)

    pos = rates["positive"]
    neu = rates["neutral"]
    neg = rates["negative"]

    if pos >= 70 and neg <= 15:
        verdict, badge_class = "Generally positive", "decision-good"
        decision = (
            "Most customers report a positive experience and recurring "
            "complaints remain limited."
        )
    elif pos >= 55 and neg <= 25:
        verdict, badge_class = "Mostly positive", "decision-watch"
        decision = (
            "The group performs well overall, although repeated complaints "
            "should be checked before choosing a specific product."
        )
    elif neg >= 35:
        verdict, badge_class = "High complaint rate", "decision-risk"
        decision = (
            "Negative feedback is substantial. Compare individual products "
            "and recurring issues carefully."
        )
    else:
        verdict, badge_class = "Mixed feedback", "decision-watch"
        decision = (
            "Customer opinion is divided. Product-level differences are likely "
            "to matter more than the group average."
        )

    values, concerns = _theme_lists_for_reviews(selected_reviews)
    representative_reviews = _representative_review_rows(
        selected_reviews,
        limit=2,
    )

    rating_text = f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "N/A"
    reviews_html = "".join(
        f'<div class="review-quote">“{html.escape(review[:300])}'
        f'{"…" if len(review) > 300 else ""}”</div>'
        for review in representative_reviews
    ) or '<div class="empty-inline">No representative reviews available.</div>'

    return f"""
<div class="decision-card">
  <div class="decision-header">
    <div>
      <div class="eyebrow">Product-group performance</div>
      <div class="decision-title">{html.escape(selected_category)}</div>
    </div>
    <span class="decision-badge {badge_class}">{verdict}</span>
  </div>

  <p class="decision-copy">{decision}</p>

  {_sentiment_bar_html(pos, neu, neg)}

  <div class="supporting-metrics">
    <div><strong>{rating_text}</strong><span>Average rating</span></div>
    <div><strong>{format_number(n_reviews)}</strong><span>Reviews analysed</span></div>
  </div>

  <div class="two-column-insights">
    <section>
      <span class="insight-label">What customers value</span>
      {_theme_list_html(values, "No consistent positive theme detected.")}
    </section>
    <section>
      <span class="insight-label">Common concerns</span>
      {_theme_list_html(concerns, "No recurring concern detected.")}
    </section>
  </div>

  <div class="insight-block evidence-block">
    <span class="insight-label">Representative feedback</span>
    {reviews_html}
  </div>
</div>
"""

# ============================================================
# 9. Seller functions
# ============================================================

def _article_row_for_meta(meta_category: str) -> pd.DataFrame:
    if articles_df.empty or not meta_category:
        return pd.DataFrame()

    if article_meta_name_col and article_meta_name_col in articles_df.columns:
        match = articles_df[
            articles_df[article_meta_name_col].astype(str) == str(meta_category)
        ]
        if not match.empty:
            return match

    if (
        article_cluster_id_col
        and profile_meta_name_col
        and profile_cluster_id_col
        and profile_cluster_id_col in cluster_profile_df.columns
    ):
        profile_match = cluster_profile_df[
            cluster_profile_df[profile_meta_name_col].astype(str)
            == str(meta_category)
        ]
        if not profile_match.empty:
            cluster_id = profile_match.iloc[0][profile_cluster_id_col]
            return articles_df[
                articles_df[article_cluster_id_col].astype(str)
                == str(cluster_id)
            ]

    return pd.DataFrame()


def _article_value(row: pd.Series, column: Optional[str], default=None):
    if column is None or column not in row.index:
        return default
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return value


def _article_percent(row: pd.Series, column: Optional[str]) -> float:
    value = pd.to_numeric(
        pd.Series([_article_value(row, column, np.nan)]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(value):
        return np.nan
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def _split_exported_list(value, limit: int = 5) -> List[str]:
    if value is None or pd.isna(value):
        return []
    parts = [
        part.strip()
        for part in re.split(r"\s*\|\s*|\s*;\s*", str(value))
        if part.strip()
    ]
    return list(dict.fromkeys(parts))[:limit]


def _parse_structured_article(article: str) -> Dict[str, str]:
    """Fallback parser for older exports that only contain the article column."""
    text = str(article or "").strip()
    if not text:
        return {}

    headings = [
        "Overview",
        "What customers value",
        "Mixed feedback",
        "Common concerns",
        "Top products",
        "Product requiring attention",
    ]
    sections: Dict[str, str] = {}

    for index, heading in enumerate(headings):
        start = text.find(heading)
        if start < 0:
            continue
        content_start = start + len(heading)
        later_positions = [
            text.find(next_heading, content_start)
            for next_heading in headings[index + 1:]
        ]
        later_positions = [position for position in later_positions if position >= 0]
        content_end = min(later_positions) if later_positions else len(text)
        sections[heading] = text[content_start:content_end].strip()

    return sections


def _portfolio_weighted_metrics() -> Dict[str, float]:
    """Return full-portfolio sentiment from the exported category metrics."""
    result = {
        "positive": np.nan,
        "neutral": np.nan,
        "negative": np.nan,
        "total_reviews": 0.0,
    }

    if cluster_df.empty:
        rates = _sentiment_rates_for_df(reviews_df)
        result.update(rates)
        result["total_reviews"] = float(len(reviews_df))
        return result

    if cluster_num_reviews_col and cluster_num_reviews_col in cluster_df.columns:
        weights = pd.to_numeric(
            cluster_df[cluster_num_reviews_col],
            errors="coerce",
        ).fillna(0.0)
    else:
        weights = pd.Series(np.ones(len(cluster_df)), index=cluster_df.index)

    valid_weights = weights > 0
    result["total_reviews"] = float(weights[valid_weights].sum())

    for key, column in {
        "positive": cluster_pos_col,
        "neutral": cluster_neu_col,
        "negative": cluster_neg_col,
    }.items():
        if column is None or column not in cluster_df.columns:
            continue
        values = coerce_percent_series(cluster_df[column])
        valid = values.notna() & valid_weights
        if valid.any():
            result[key] = float(np.average(values[valid], weights=weights[valid]))

    if any(pd.isna(result[key]) for key in ["positive", "neutral", "negative"]):
        sample_rates = _sentiment_rates_for_df(reviews_df)
        for key in ["positive", "neutral", "negative"]:
            if pd.isna(result[key]):
                result[key] = sample_rates[key]

    total_pct = sum(float(result[key]) for key in ["positive", "neutral", "negative"])
    if total_pct > 0 and abs(total_pct - 100.0) <= 1.0:
        for key in ["positive", "neutral", "negative"]:
            result[key] = 100.0 * float(result[key]) / total_pct

    return result


def _overall_sentiment_rates() -> Dict[str, float]:
    metrics = _portfolio_weighted_metrics()
    return {
        "positive": float(metrics["positive"]),
        "neutral": float(metrics["neutral"]),
        "negative": float(metrics["negative"]),
    }

def _meta_category_cluster_rows(meta_category: str) -> pd.DataFrame:
    """Return the category-level rows that belong to one meta-category."""
    if (
        not meta_category
        or cluster_meta_name_col is None
        or cluster_meta_name_col not in cluster_df.columns
    ):
        return pd.DataFrame()

    return cluster_df[
        cluster_df[cluster_meta_name_col].astype(str) == str(meta_category)
    ].copy()


def _meta_category_weighted_metrics(meta_category: str) -> Dict[str, float]:
    """
    Calculate the real sentiment distribution for a meta-category.

    Each category percentage is weighted by its number of reviews:
        sum(category_rate * category_review_count) / sum(review_count)

    This is equivalent to counting all positive/neutral/negative reviews across
    the meta-category and avoids the misleading unweighted mean stored in the
    legacy avg_* profile columns.
    """
    rows = _meta_category_cluster_rows(meta_category)

    result = {
        "positive": np.nan,
        "neutral": np.nan,
        "negative": np.nan,
        "total_reviews": 0.0,
        "num_categories": 0.0,
    }

    if rows.empty:
        return result

    result["num_categories"] = float(len(rows))

    if (
        cluster_num_reviews_col
        and cluster_num_reviews_col in rows.columns
    ):
        weights = pd.to_numeric(
            rows[cluster_num_reviews_col],
            errors="coerce",
        ).fillna(0.0)
    else:
        weights = pd.Series(
            np.ones(len(rows), dtype=float),
            index=rows.index,
        )

    valid_weights = weights.notna() & (weights > 0)
    result["total_reviews"] = float(weights[valid_weights].sum())

    percent_map = {
        "positive": cluster_pos_col,
        "neutral": cluster_neu_col,
        "negative": cluster_neg_col,
    }

    for key, column in percent_map.items():
        if column is None or column not in rows.columns:
            continue

        values = coerce_percent_series(rows[column])
        valid = values.notna() & valid_weights

        if valid.any():
            result[key] = float(
                np.average(
                    values[valid],
                    weights=weights[valid],
                )
            )

    # Validation: all three classes should describe the same population.
    sentiment_values = [
        result["positive"],
        result["neutral"],
        result["negative"],
    ]
    if all(pd.notna(value) for value in sentiment_values):
        total_pct = float(sum(sentiment_values))

        # Small deviations are only rounding noise. Normalize them to 100%.
        if total_pct > 0 and abs(total_pct - 100.0) <= 1.0:
            scale = 100.0 / total_pct
            for key in ["positive", "neutral", "negative"]:
                result[key] *= scale

    return result


def _meta_category_metrics_table() -> pd.DataFrame:
    """Build one verified, weighted row per meta-category."""
    rows = []

    for meta_category in META_CATEGORIES:
        metrics = _meta_category_weighted_metrics(meta_category)

        if pd.isna(metrics["negative"]):
            # Last-resort fallback when category-level exported metrics are
            # unavailable. This uses the app review rows, not avg_* columns.
            reviews = get_reviews_for_category_or_meta(
                meta_category=meta_category,
                limit=10**9,
            )
            rates = _sentiment_rates_for_df(reviews)
            metrics.update(rates)
            metrics["total_reviews"] = float(len(reviews))

        rows.append({
            "meta_category": str(meta_category),
            "positive_pct": metrics["positive"],
            "neutral_pct": metrics["neutral"],
            "negative_pct": metrics["negative"],
            "total_reviews": metrics["total_reviews"],
            "num_categories": metrics["num_categories"],
        })

    return pd.DataFrame(rows)


def _portfolio_extremes() -> Tuple[str, str]:
    strongest = "Not available"
    priority = "Not available"

    metrics_df = _meta_category_metrics_table()
    if metrics_df.empty:
        return strongest, priority

    metrics_df = metrics_df.dropna(subset=["negative_pct"])
    if metrics_df.empty:
        return strongest, priority

    strongest_row = metrics_df.sort_values("negative_pct").iloc[0]
    priority_row = metrics_df.sort_values(
        "negative_pct",
        ascending=False,
    ).iloc[0]

    strongest = (
        f"{strongest_row['meta_category']} · "
        f"{strongest_row['negative_pct']:.1f}% negative"
    )
    priority = (
        f"{priority_row['meta_category']} · "
        f"{priority_row['negative_pct']:.1f}% negative"
    )
    return strongest, priority

def seller_overview_html() -> str:
    """Return a compact overview based on the complete exported portfolio."""
    portfolio = _portfolio_weighted_metrics()
    strongest, priority = _portfolio_extremes()

    return f"""
<div class="overview-headline-grid">
  <div class="headline-card">
    <strong>{format_number(portfolio["total_reviews"])}</strong>
    <span>Portfolio reviews</span>
  </div>
  <div class="headline-card">
    <strong>{len(META_CATEGORIES):,}</strong>
    <span>Meta-categories</span>
  </div>
</div>

<div class="portfolio-sentiment">
  <div class="eyebrow">Portfolio sentiment</div>
  {_sentiment_bar_html(
      portfolio["positive"],
      portfolio["neutral"],
      portfolio["negative"],
      compact=True,
  )}
</div>

<div class="priority-grid">
  <div class="priority-card priority-risk">
    <span class="eyebrow">Needs attention</span>
    <strong>{html.escape(priority)}</strong>
    <p>Highest verified negative-feedback rate in the portfolio.</p>
  </div>
  <div class="priority-card priority-good">
    <span class="eyebrow">Strongest performer</span>
    <strong>{html.escape(strongest)}</strong>
    <p>Lowest verified negative-feedback rate in the portfolio.</p>
  </div>
</div>
"""

def plot_priority_meta_categories(limit: int = 8):
    """Show meta-categories ranked by verified weighted negative-review rate."""
    temp = _meta_category_metrics_table()

    if temp.empty:
        fig = px.bar(title="Category priority chart unavailable")
        fig.update_layout(height=390)
        return fig

    temp = temp.dropna(subset=["negative_pct"])
    temp = temp.sort_values(
        "negative_pct",
        ascending=False,
    ).head(limit)
    temp = temp.sort_values("negative_pct", ascending=True)

    fig = px.bar(
        temp,
        x="negative_pct",
        y="meta_category",
        orientation="h",
        title="Meta-categories with the highest negative feedback",
        labels={
            "negative_pct": "Negative reviews (%)",
            "meta_category": "",
        },
        hover_data={
            "positive_pct": ":.1f",
            "neutral_pct": ":.1f",
            "total_reviews": ":,.0f",
            "meta_category": False,
        },
    )
    fig.update_traces(marker_color=PALETTE["negative"])
    fig.update_layout(
        height=390,
        showlegend=False,
        margin=dict(l=175, r=25, t=55, b=45),
        xaxis=dict(range=[0, max(5.0, float(temp["negative_pct"].max()) * 1.15)])
        if not temp.empty else None,
    )
    return fig

def _profile_values(meta_category: str) -> Dict[str, object]:
    profile = pd.DataFrame()
    if profile_meta_name_col:
        profile = cluster_profile_df[
            cluster_profile_df[profile_meta_name_col].astype(str)
            == str(meta_category)
        ]

    reviews = get_reviews_for_category_or_meta(
        meta_category=meta_category,
        limit=1200,
    )

    verified = _meta_category_weighted_metrics(meta_category)
    article_match = _article_row_for_meta(meta_category)
    article_row = article_match.iloc[0] if not article_match.empty else None

    values: Dict[str, object] = {
        "profile": profile,
        "reviews": reviews,
        "total_reviews": verified["total_reviews"] or len(reviews),
        "num_categories": (
            int(verified["num_categories"])
            if verified["num_categories"]
            else "N/A"
        ),
        "keywords": [],
        "top_products": [],
        "positive": verified["positive"],
        "neutral": verified["neutral"],
        "negative": verified["negative"],
    }

    if not profile.empty:
        row = profile.iloc[0]
        if values["num_categories"] == "N/A" and profile_num_categories_col:
            values["num_categories"] = row.get(profile_num_categories_col, "N/A")
        if profile_keywords_col:
            raw_keywords = short_list_text(row.get(profile_keywords_col, ""), max_items=6)
            values["keywords"] = [
                item.strip()
                for item in raw_keywords.split(",")
                if item.strip() and item.strip() != "Not available"
            ]
        if profile_top_products_col:
            raw_products = short_list_text(row.get(profile_top_products_col, ""), max_items=5)
            values["top_products"] = [
                item.strip()
                for item in raw_products.split(",")
                if item.strip() and item.strip() != "Not available"
            ]

    # The final generated_articles export is the source of truth for the
    # validated meta-category summaries and their review-level metrics.
    if article_row is not None:
        article_total = pd.to_numeric(
            pd.Series([_article_value(article_row, article_num_reviews_col, np.nan)]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(article_total):
            values["total_reviews"] = float(article_total)

        for key, column in {
            "positive": article_pos_col,
            "neutral": article_neu_col,
            "negative": article_neg_col,
        }.items():
            article_rate = _article_percent(article_row, column)
            if pd.notna(article_rate):
                values[key] = article_rate

        exported_products = _split_exported_list(
            _article_value(article_row, article_top_products_col, None),
            limit=5,
        )
        if exported_products:
            values["top_products"] = exported_products

    if any(pd.isna(values[key]) for key in ["positive", "neutral", "negative"]):
        rates = _sentiment_rates_for_df(reviews)
        for key in ["positive", "neutral", "negative"]:
            if pd.isna(values[key]):
                values[key] = rates[key]

    for key in ["positive", "neutral", "negative"]:
        values[key] = float(values[key]) if pd.notna(values[key]) else 0.0

    return values

def category_profile_html(meta_category: str) -> str:
    """Return the selected category, review volume and sentiment distribution."""
    if not meta_category:
        return '<div class="empty-state">Select a meta-category.</div>'

    values = _profile_values(meta_category)
    reviews = values["reviews"]
    profile = values["profile"]

    if profile.empty and reviews.empty:
        return '<div class="empty-state">No data found for this meta-category.</div>'

    return f"""
<div class="category-profile">
  <div class="category-profile-heading">
    <div>
      <div class="eyebrow">Selected meta-category</div>
      <h3>{html.escape(str(meta_category))}</h3>
    </div>
    <div class="profile-counts">
      <div><strong>{format_number(values["total_reviews"])}</strong><span>Reviews</span></div>
      <div><strong>{html.escape(str(values["num_categories"]))}</strong><span>Categories</span></div>
    </div>
  </div>

  {_sentiment_bar_html(
      values["positive"],
      values["neutral"],
      values["negative"],
  )}
</div>
"""

def _top_products_from_reviews(
    reviews: pd.DataFrame,
    limit: int = 5,
) -> List[str]:
    if (
        reviews.empty
        or not product_col
        or product_col not in reviews.columns
    ):
        return []

    values = (
        reviews[product_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    values = values[
        values.ne("")
        & values.str.lower().ne("nan")
    ]
    return values.value_counts().head(limit).index.tolist()



# ------------------------------------------------------------
# Category-safe product and summary safeguards
# ------------------------------------------------------------

def _clean_product_display_name(value) -> str:
    """Normalise duplicated/empty product names before displaying them."""
    text = re.sub(r"\s+", " ", str(value or "")).strip(" ,;|-")
    if not text or text.lower() in {"nan", "none", "unknown", "unknown product"}:
        return ""

    # Common export issue: "Echo (White), Echo (White)".
    duplicate_match = re.fullmatch(r"(.+?)\s*,\s*\1", text, flags=re.IGNORECASE)
    if duplicate_match:
        text = duplicate_match.group(1).strip()

    return text


def _product_rules_for_meta(meta_category: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Return conservative include/exclude patterns for the five named clusters.

    These rules do not redefine the clustering. They only prevent obviously
    cross-family product names from being shown as category leaders.
    """
    meta = str(meta_category).lower()

    if "fire tv" in meta:
        return (
            (r"\bfire tv\b", r"\bstreaming\b", r"\bvoice remote\b"),
            (r"\becho\b", r"\bkindle\b", r"\btablet\b", r"\bcharger\b", r"\bcable\b", r"\badapter\b"),
        )

    if "echo" in meta or "alexa" in meta:
        return (
            (r"\becho\b", r"\balexa\b", r"\bspeaker\b", r"\bsmart home\b", r"\bhome entertainment\b"),
            (r"\bfire tv\b", r"\bkindle\b", r"\btablet\b", r"\bcharger\b", r"\bcable\b", r"\badapter\b"),
        )

    if "charger" in meta or "cable" in meta:
        return (
            (r"\bcharger\b", r"\bcharging\b", r"\bcable\b", r"\badapter\b", r"\busb\b", r"\bpowerfast\b", r"\bpower adapter\b", r"\bconnector\b"),
            (),
        )

    if "small accessories" in meta or "cases" in meta or "pet supplies" in meta:
        return (
            (r"\bcase\b", r"\bcover\b", r"\bstand\b", r"\bbag\b", r"\bsleeve\b", r"\baccessor", r"\bpet\b", r"\bdog\b", r"\bcat\b", r"\bcrate\b", r"\blitter\b"),
            (r"\bfire tv\b", r"\becho\b", r"\balexa\b"),
        )

    if "kindle" in meta or "tablets" in meta:
        return (
            (r"\bkindle\b", r"\btablet\b", r"\bfire hd\b", r"\be-reader\b", r"\bereader\b", r"\bbattery\b", r"\boffice\b"),
            (r"\bfire tv\b", r"\becho\b", r"\balexa\b"),
        )

    return (), ()


def _product_matches_meta_category(product_name: str, meta_category: str) -> bool:
    """Reject product names that clearly belong to another named family."""
    product = _clean_product_display_name(product_name)
    if not product:
        return False

    include_patterns, exclude_patterns = _product_rules_for_meta(meta_category)
    low = product.lower()

    if any(re.search(pattern, low) for pattern in exclude_patterns):
        return False

    if include_patterns:
        return any(re.search(pattern, low) for pattern in include_patterns)

    return True


def _filter_products_for_meta(
    products: List[str],
    meta_category: str,
    limit: int = 5,
) -> List[str]:
    """Clean, deduplicate and keep only products compatible with the category."""
    filtered: List[str] = []
    seen = set()

    for value in products:
        product = _clean_product_display_name(value)
        key = re.sub(r"[^a-z0-9]+", " ", product.lower()).strip()

        if not product or not key or key in seen:
            continue
        if not _product_matches_meta_category(product, meta_category):
            continue

        seen.add(key)
        filtered.append(product)

        if len(filtered) >= limit:
            break

    return filtered


def _rank_category_specific_products(
    reviews: pd.DataFrame,
    meta_category: str,
    limit: int = 5,
) -> Tuple[List[str], str]:
    """
    Build conservative product rankings from the selected category reviews.

    Review volume is used as a Bayesian prior so that products with only a few
    perfect or poor reviews do not dominate the ranking.
    """
    if (
        reviews.empty
        or not product_col
        or product_col not in reviews.columns
    ):
        return [], ""

    temp = reviews.copy()
    temp["_product"] = temp[product_col].map(_clean_product_display_name)
    temp = temp[
        temp["_product"].apply(
            lambda value: _product_matches_meta_category(value, meta_category)
        )
    ].copy()

    if temp.empty:
        return [], ""

    temp["_sentiment"] = _review_sentiment_labels(temp)
    if rating_col and rating_col in temp.columns:
        temp["_rating"] = pd.to_numeric(temp[rating_col], errors="coerce")
    else:
        temp["_rating"] = np.nan

    rows = []
    for product, group in temp.groupby("_product"):
        count = len(group)
        if count < 3:
            continue

        sentiments = group["_sentiment"].value_counts()
        labelled = int(sentiments.sum())
        positive_pct = 100.0 * sentiments.get("Positive", 0) / labelled if labelled else np.nan
        negative_pct = 100.0 * sentiments.get("Negative", 0) / labelled if labelled else np.nan
        avg_rating = group["_rating"].mean()

        rows.append({
            "product": product,
            "reviews": count,
            "positive_pct": positive_pct,
            "negative_pct": negative_pct,
            "avg_rating": avg_rating,
        })

    stats = pd.DataFrame(rows)
    if stats.empty:
        return [], ""

    portfolio_positive = stats["positive_pct"].dropna().mean()
    portfolio_negative = stats["negative_pct"].dropna().mean()
    portfolio_positive = float(portfolio_positive) if pd.notna(portfolio_positive) else 50.0
    portfolio_negative = float(portfolio_negative) if pd.notna(portfolio_negative) else 20.0
    prior = 10.0

    stats["top_score"] = (
        stats["positive_pct"].fillna(portfolio_positive) * stats["reviews"]
        + portfolio_positive * prior
    ) / (stats["reviews"] + prior)

    stats["risk_score"] = (
        stats["negative_pct"].fillna(portfolio_negative) * stats["reviews"]
        + portfolio_negative * prior
    ) / (stats["reviews"] + prior)

    top_products = (
        stats.sort_values(
            ["top_score", "avg_rating", "reviews"],
            ascending=[False, False, False],
        )
        .head(limit)["product"]
        .tolist()
    )

    worst_product = (
        stats.sort_values(
            ["risk_score", "avg_rating", "reviews"],
            ascending=[False, True, False],
        )
        .iloc[0]["product"]
    )

    return top_products, str(worst_product)


def _is_insufficient_summary(text: str) -> bool:
    low = str(text or "").strip().lower()
    return (
        not low
        or low == "nan"
        or low.startswith("insufficient recurring")
        or low.startswith("no recurring")
        or low.startswith("insufficient evidence")
    )


def _join_themes(items: List[str], limit: int = 3) -> str:
    clean = [str(item).strip() for item in items if str(item).strip()][:limit]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def _display_summaries(
    positive_summary: str,
    neutral_summary: str,
    negative_summary: str,
    values: Dict[str, object],
) -> Tuple[str, str, str, bool]:
    """
    Replace non-informative exported placeholders with concise, grounded text.

    The validated notebook summary remains the primary source. The app fallback
    is used only when that source explicitly reports insufficient recurrence.
    """
    reviews: pd.DataFrame = values.get("reviews", pd.DataFrame())
    value_themes, concern_themes = _theme_lists_for_reviews(reviews)

    used_fallback = False

    if _is_insufficient_summary(positive_summary):
        used_fallback = True
        themes = _join_themes(value_themes)
        if themes:
            positive_summary = f"Customers most often praise {themes}."
        elif float(values.get("positive", 0.0)) >= 70.0:
            positive_summary = (
                "Customer sentiment is strongly positive, although no single "
                "recurring strength clearly dominates the available review evidence."
            )
        else:
            positive_summary = "No clear recurring positive theme stands out."

    if _is_insufficient_summary(neutral_summary):
        used_fallback = True
        neutral_rate = float(values.get("neutral", 0.0))
        if neutral_rate <= 5.0:
            neutral_summary = (
                "Neutral feedback is limited and no recurring mixed theme stands out."
            )
        else:
            themes = _join_themes(concern_themes[:2])
            neutral_summary = (
                f"Mixed feedback mainly relates to {themes}."
                if themes
                else "No single recurring mixed-feedback theme clearly dominates."
            )

    if _is_insufficient_summary(negative_summary):
        used_fallback = True
        themes = _join_themes(concern_themes)
        if themes:
            negative_summary = f"Recurring complaints focus on {themes}."
        elif float(values.get("negative", 0.0)) <= 5.0:
            negative_summary = (
                "Negative feedback is limited and no single recurring concern clearly dominates."
            )
        else:
            negative_summary = "No clear recurring concern was identified."

    return (
        positive_summary,
        neutral_summary,
        negative_summary,
        used_fallback,
    )

def review_summary_html(meta_category: str) -> str:
    """Render validated summaries with category-safe product safeguards."""
    if not meta_category:
        return '<div class="empty-state">Select a meta-category.</div>'

    article_match = _article_row_for_meta(meta_category)
    values = _profile_values(meta_category)

    if article_match.empty:
        return (
            '<div class="empty-state">'
            'No validated summary was exported for this meta-category.'
            '</div>'
        )

    row = article_match.iloc[0]
    parsed = _parse_structured_article(
        _article_value(row, article_text_col, "")
    )

    positive_summary = str(
        _article_value(
            row,
            article_positive_summary_col,
            parsed.get("What customers value", ""),
        )
    ).strip()
    neutral_summary = str(
        _article_value(
            row,
            article_neutral_summary_col,
            parsed.get("Mixed feedback", ""),
        )
    ).strip()
    negative_summary = str(
        _article_value(
            row,
            article_negative_summary_col,
            parsed.get("Common concerns", ""),
        )
    ).strip()

    (
        positive_summary,
        neutral_summary,
        negative_summary,
        used_summary_fallback,
    ) = _display_summaries(
        positive_summary,
        neutral_summary,
        negative_summary,
        values,
    )

    exported_products = _split_exported_list(
        _article_value(row, article_top_products_col, None),
        limit=8,
    )
    profile_products = list(values.get("top_products", []))

    top_products = _filter_products_for_meta(
        exported_products + profile_products,
        meta_category,
        limit=5,
    )

    ranked_products, ranked_worst = _rank_category_specific_products(
        values.get("reviews", pd.DataFrame()),
        meta_category,
        limit=5,
    )

    top_products = _filter_products_for_meta(
        top_products + ranked_products,
        meta_category,
        limit=5,
    )

    exported_worst = _clean_product_display_name(
        _article_value(
            row,
            article_worst_product_col,
            parsed.get("Product requiring attention", ""),
        )
    )

    if _product_matches_meta_category(exported_worst, meta_category):
        worst_product = exported_worst
    elif ranked_worst and _product_matches_meta_category(ranked_worst, meta_category):
        worst_product = ranked_worst
    else:
        worst_product = "Insufficient category-specific evidence."

    def summary_block(text: str) -> str:
        return f'<p class="summary-text">{html.escape(str(text))}</p>'

    products_html = _plain_list_html(
        top_products,
        "Insufficient category-specific product evidence.",
    )

    source_label = (
        "BART-assisted · category safeguards applied"
        if used_summary_fallback
        else "BART-assisted · quality-audited"
    )

    return f"""
<div class="summary-card">
  <div class="summary-heading">
    <div>
      <div class="eyebrow">Validated review summary</div>
      <h3>{html.escape(str(meta_category))}</h3>
    </div>
    <span class="summary-source">{source_label}</span>
  </div>

  <div class="summary-grid">
    <section>
      <span class="insight-label">What customers value</span>
      {summary_block(positive_summary)}
    </section>
    <section>
      <span class="insight-label">Mixed feedback</span>
      {summary_block(neutral_summary)}
    </section>
    <section>
      <span class="insight-label">Common concerns</span>
      {summary_block(negative_summary)}
    </section>
  </div>

  <div class="two-column-insights summary-products">
    <section>
      <span class="insight-label">Top products</span>
      {products_html}
    </section>
    <section>
      <span class="insight-label">Product requiring attention</span>
      <p class="summary-text">{html.escape(worst_product)}</p>
    </section>
  </div>
</div>
"""

def category_intelligence(meta_category: str):
    return (
        plot_cluster_3d(meta_category),
        category_profile_html(meta_category),
        review_summary_html(meta_category),
    )


# ============================================================
# 9b. Product ranking (Top / Worst products)
# ============================================================

def _canonical_sentiment_value(v) -> Optional[str]:
    v = str(v).lower()
    if "pos" in v or v == "2":
        return "Positive"
    if "neg" in v or v == "0":
        return "Negative"
    if "neu" in v or v == "1":
        return "Neutral"
    return None


def _row_sentiment_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if sentiment_col and sentiment_col in df.columns:
        return df[sentiment_col].map(_canonical_sentiment_value)
    if rating_col and rating_col in df.columns:
        r = pd.to_numeric(df[rating_col], errors="coerce")
        return pd.Series(np.where(r >= 4, "Positive", np.where(r <= 2, "Negative", "Neutral")), index=df.index)
    return None


def _clean_product_label(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip(" ,;|-")
    duplicate = re.fullmatch(r"(.{3,}?)\s*,\s*\1", text, flags=re.I)
    return duplicate.group(1).strip() if duplicate else text


def build_product_stats(min_reviews: int = 8) -> pd.DataFrame:
    if not product_col or product_col not in reviews_df.columns:
        return pd.DataFrame()

    df = reviews_df.copy()
    df["_product"] = df[product_col].map(_clean_product_label)
    df = df[
        df["_product"].ne("")
        & df["_product"].str.lower().ne("nan")
        & df["_product"].str.lower().ne("unknown product")
    ]
    if df.empty:
        return pd.DataFrame()

    sent = _row_sentiment_series(df)
    if sent is not None:
        df = df.assign(_sent=sent)

    rows = []
    for name, group in df.groupby("_product"):
        count = len(group)
        if count < min_reviews:
            continue

        positive = neutral = negative = np.nan
        if "_sent" in group.columns:
            counts = group["_sent"].value_counts()
            denominator = int(counts.sum())
            if denominator:
                positive = 100.0 * counts.get("Positive", 0) / denominator
                neutral = 100.0 * counts.get("Neutral", 0) / denominator
                negative = 100.0 * counts.get("Negative", 0) / denominator

        avg_rating = (
            pd.to_numeric(group[rating_col], errors="coerce").mean()
            if rating_col and rating_col in group.columns
            else np.nan
        )

        rows.append({
            "product": name,
            "reviews": count,
            "positive_pct": positive,
            "neutral_pct": neutral,
            "negative_pct": negative,
            "avg_rating": avg_rating,
        })

    stats = pd.DataFrame(rows)
    if stats.empty:
        return stats

    # Bayesian smoothing prevents a product with very few reviews from
    # dominating the ranking solely because it has an extreme percentage.
    prior_strength = 20.0
    overall_positive = float(stats["positive_pct"].mean()) if stats["positive_pct"].notna().any() else 50.0
    overall_negative = float(stats["negative_pct"].mean()) if stats["negative_pct"].notna().any() else 10.0
    volume = stats["reviews"].astype(float)

    stats["adjusted_positive"] = (
        stats["positive_pct"].fillna(overall_positive) * volume
        + overall_positive * prior_strength
    ) / (volume + prior_strength)
    stats["adjusted_negative"] = (
        stats["negative_pct"].fillna(overall_negative) * volume
        + overall_negative * prior_strength
    ) / (volume + prior_strength)

    return stats

def get_top_bottom_products(n: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stats = build_product_stats(min_reviews=8)
    if stats.empty or len(stats) < max(2, n):
        stats = build_product_stats(min_reviews=3)
    if stats.empty:
        return pd.DataFrame(), pd.DataFrame()

    if stats["positive_pct"].notna().any():
        top = stats.sort_values(
            ["adjusted_positive", "reviews"],
            ascending=[False, False],
        ).head(n)
        worst = stats.sort_values(
            ["adjusted_negative", "reviews"],
            ascending=[False, False],
        ).head(n)
    else:
        top = stats.sort_values(["avg_rating", "reviews"], ascending=[False, False]).head(n)
        worst = stats.sort_values(["avg_rating", "reviews"], ascending=[True, False]).head(n)

    return top.reset_index(drop=True), worst.reset_index(drop=True)

def _format_product_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"Info": ["No product-level data available."]})

    out = pd.DataFrame()
    out["Rank"] = range(1, len(df) + 1)
    out["Product"] = df["product"].astype(str).str.slice(0, 80)
    out["Reviews"] = df["reviews"].astype(int)

    if df["positive_pct"].notna().any():
        out["Positive"] = df["positive_pct"].map(lambda v: f"{v:.0f}%" if pd.notna(v) else "—")
        out["Negative"] = df["negative_pct"].map(lambda v: f"{v:.0f}%" if pd.notna(v) else "—")

    if df["avg_rating"].notna().any():
        out["Avg rating"] = df["avg_rating"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")

    return out


def _product_bar(df: pd.DataFrame, value_col: str, title: str, x_title: str, color: str):
    if df.empty or df[value_col].isna().all():
        return px.bar(title=f"{title} (no data)")

    # Keep exactly one visible bar per ranked row.
    # The rank prefix avoids Plotly merging/stacking products that share the same truncated label.
    d = df.copy().reset_index(drop=True)
    d["rank"] = range(1, len(d) + 1)
    d["short_product"] = d["product"].astype(str).str.slice(0, 42)
    d["label"] = d.apply(lambda r: f"#{int(r['rank'])} {r['short_product']}", axis=1)

    # Keep ranking order in the table and chart: #1 at the top.
    category_order = d["label"].tolist()[::-1]

    fig = px.bar(
        d,
        x=value_col,
        y="label",
        orientation="h",
        title=title,
        hover_name="product",
        hover_data={
            value_col: ":.1f",
            "reviews": True,
            "label": False,
            "short_product": False,
            "rank": False,
        },
    )

    fig.update_traces(marker_color=color)
    fig.update_yaxes(categoryorder="array", categoryarray=category_order, title="")

    if value_col.endswith("_pct"):
        x_range = [0, 100]
    elif value_col == "avg_rating":
        x_range = [0, 5]
    else:
        x_range = None

    fig.update_layout(
        height=330,
        yaxis_title="",
        xaxis_title=x_title,
        xaxis=dict(range=x_range) if x_range else None,
        margin=dict(l=190, r=25, t=55, b=45),
        showlegend=False,
    )

    return fig


def build_top_bottom_views(n: int = 3):
    top, worst = get_top_bottom_products(n)

    if not top.empty and top["positive_pct"].notna().any():
        top_fig = _product_bar(
            top,
            "positive_pct",
            f"Top {len(top)} most loved products",
            "Positive reviews (%)",
            PALETTE["consumer"],
        )
        worst_fig = _product_bar(
            worst,
            "negative_pct",
            f"Worst {len(worst)} most criticised products",
            "Negative reviews (%)",
            PALETTE["negative"],
        )
    else:
        top_fig = _product_bar(
            top,
            "avg_rating",
            f"Top {len(top)} highest rated products",
            "Average rating",
            PALETTE["consumer"],
        )
        worst_fig = _product_bar(
            worst,
            "avg_rating",
            f"Worst {len(worst)} lowest rated products",
            "Average rating",
            PALETTE["negative"],
        )

    return top_fig, _format_product_table(top), worst_fig, _format_product_table(worst)


# Console audit: exact weighted sentiment rates used by the seller dashboard.
# This does not add content to the interface.
try:
    print("\nVerified weighted meta-category sentiment rates:")
    print(
        _meta_category_metrics_table()[
            [
                "meta_category",
                "positive_pct",
                "neutral_pct",
                "negative_pct",
                "total_reviews",
            ]
        ].to_string(index=False)
    )
except Exception as metrics_audit_error:
    print(f"Could not print meta-category metric audit: {metrics_audit_error}")


# ============================================================
# 10. Gradio UI — simplified, decision-first design
# ============================================================

N_REVIEWS = len(reviews_df)
N_CATEGORIES = RAW_CATEGORY_COUNT
N_META = len(META_CATEGORIES)

if model is not None and model_path is not None:
    model_path_text = str(model_path).lower()
    if "distilbert" in model_path_text:
        ENGINE_LABEL = "DistilBERT"
    elif "roberta" in model_path_text:
        ENGINE_LABEL = "RoBERTa"
    else:
        ENGINE_LABEL = "Local model"
else:
    ENGINE_LABEL = "Rule-based fallback"

TOP_FIG, TOP_TABLE, WORST_FIG, WORST_TABLE = build_top_bottom_views(3)

def _product_table_html(df: pd.DataFrame, status: str) -> str:
    """Render a compact, readable HTML table instead of Gradio Dataframe."""
    if df is None or df.empty:
        return '<div class="empty-state">No product data available.</div>'

    rows = []
    for _, row in df.iterrows():
        rank = row.get("Rank", row.get("rank", ""))
        product = row.get("Product", row.get("product", "Unknown product"))

        metric_parts = []
        for col in df.columns:
            if col in {"Rank", "rank", "Product", "product"}:
                continue
            value = row.get(col)
            if pd.notna(value):
                metric_parts.append(f"<span><strong>{html.escape(str(col))}:</strong> {html.escape(str(value))}</span>")

        rows.append(
            "<tr>"
            f"<td class='rank-cell'>{html.escape(str(rank))}</td>"
            f"<td><div class='product-name'>{html.escape(str(product))}</div>"
            f"<div class='product-metrics'>{' · '.join(metric_parts)}</div></td>"
            "</tr>"
        )

    badge = "Top performers" if status == "top" else "Needs attention"
    badge_class = "status-good" if status == "top" else "status-risk"

    return (
        f"<div class='product-list-card'>"
        f"<div class='product-list-header'><span class='{badge_class}'>{badge}</span></div>"
        "<table class='product-list'><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def classifier_result(review_text: str) -> str:
    """Return the prediction as one clean HTML result card."""
    probs, explanation = predict_sentiment(review_text)

    if not review_text or not review_text.strip():
        return '<div class="empty-state">Write or paste a review first.</div>'

    prediction = max(probs, key=probs.get)
    confidence = probs[prediction] * 100

    ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    bars = "".join(
        f"<div class='prob-row'>"
        f"<div class='prob-label'><span>{html.escape(str(label))}</span><strong>{value * 100:.1f}%</strong></div>"
        f"<div class='prob-track'><div class='prob-fill' style='width:{max(0, min(100, value * 100)):.1f}%'></div></div>"
        f"</div>"
        for label, value in ordered
    )

    explanation_html = html.escape(explanation).replace("\n\n", "<br><br>").replace("\n", "<br>")

    return (
        "<div class='classifier-result'>"
        "<div class='prediction-summary'>"
        "<span class='prediction-kicker'>Prediction</span>"
        f"<div class='prediction-value'>{html.escape(str(prediction))}</div>"
        f"<div class='prediction-confidence'>{confidence:.1f}% confidence</div>"
        "</div>"
        f"<div class='probabilities'>{bars}</div>"
        f"<div class='prediction-explanation'>{explanation_html}</div>"
        "</div>"
    )


TOP_TABLE_HTML = _product_table_html(TOP_TABLE, "top")
WORST_TABLE_HTML = _product_table_html(WORST_TABLE, "worst")

FORCE_LIGHT_JS = """
() => {
  try {
    document.documentElement.style.colorScheme = "light";
    document.documentElement.classList.remove("dark");
    document.body.classList.remove("dark");
    localStorage.setItem("theme", "light");

    const url = new URL(window.location.href);
    if (url.searchParams.get("__theme") !== "light") {
      url.searchParams.set("__theme", "light");
      window.history.replaceState({}, "", url.toString());
    }
  } catch (e) {
    console.warn("Could not force light theme:", e);
  }
}
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root,
html,
body,
.gradio-container {
  color-scheme: light !important;
  --body-background-fill: #F7F9FC !important;
  --background-fill-primary: #FFFFFF !important;
  --background-fill-secondary: #F3F6FA !important;
  --body-text-color: #172033 !important;
  --body-text-color-subdued: #667085 !important;
  --block-background-fill: #FFFFFF !important;
  --block-border-color: #DCE4EE !important;
  --block-label-background-fill: #FFFFFF !important;
  --block-label-text-color: #526176 !important;
  --input-background-fill: #FFFFFF !important;
  --input-border-color: #CFD8E5 !important;
  --button-primary-background-fill: #5E82C4 !important;
  --button-primary-background-fill-hover: #4D70B0 !important;
  --button-primary-text-color: #FFFFFF !important;
}

html,
body,
.gradio-container {
  margin: 0 !important;
  min-height: 100% !important;
  background: #F7F9FC !important;
  color: #172033 !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.gradio-container { max-width: none !important; padding: 0 !important; }
footer { display: none !important; }

#landing-view,
#consumer-view,
#seller-view,
.gradio-container .view {
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  gap: 0 !important;
}

.gradio-container .hide,
.gradio-container [hidden] { display: none !important; }

.app-screen {
  width: min(1180px, calc(100% - 48px)) !important;
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 34px 0 52px !important;
}

h1, h2, h3, h4 { color: #172033 !important; letter-spacing: -.025em; }
.brand { color: #172033; font-size: .9rem; font-weight: 700; }
.brand span { color: #5E82C4; }

.hero { max-width: 820px; margin: 0 auto; padding: 42px 0 24px; text-align: center; }
.hero h1 { margin: 10px 0 16px; font-size: clamp(2.25rem, 5vw, 3.75rem); line-height: 1.04; }
.hero p, .mode-copy, .entry-copy, .section-copy { color: #667085; line-height: 1.6; }

/* Force every Gradio wrapper used as a card to remain light. */
.gradio-container .entry-card,
.gradio-container .content-card,
.gradio-container .entry-card > div,
.gradio-container .content-card > div,
.gradio-container .entry-card .form,
.gradio-container .content-card .form {
  background: #FFFFFF !important;
  color: #172033 !important;
}

.entry-card,
.content-card {
  border: 1px solid #DCE4EE !important;
  box-shadow: 0 10px 28px rgba(55, 74, 102, .06) !important;
}
.entry-card { height: 100%; padding: 26px !important; border-radius: 18px !important; }
.content-card { width: 100% !important; margin-top: 20px !important; padding: 24px !important; border-radius: 18px !important; }

.entry-kicker, .eyebrow {
  color: #5E82C4;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.entry-title { margin: 8px 0; color: #172033; font-size: 1.35rem; font-weight: 700; }
.entry-copy { min-height: 52px; margin-bottom: 18px; }

.mode-header {
  width: 100%; margin-bottom: 18px !important; padding-bottom: 20px !important;
  align-items: center; border-bottom: 1px solid #DCE4EE;
}
.mode-heading { margin: 5px 0; font-size: 1.9rem; font-weight: 700; }
.section-title { margin-bottom: 5px; color: #172033; font-size: 1.08rem; font-weight: 700; }
.section-copy { margin-bottom: 18px; font-size: .91rem; }

.gradio-container button.primary,
.gradio-container .consumer-action button {
  min-height: 43px !important;
  color: #FFFFFF !important;
  background: #5E82C4 !important;
  border: 1px solid #5E82C4 !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 12px rgba(94,130,196,.16) !important;
}
.gradio-container button.primary:hover,
.gradio-container .consumer-action button:hover {
  background: #4D70B0 !important; border-color: #4D70B0 !important;
}
.gradio-container .back-btn button {
  min-height: 40px !important; color: #43526A !important; background: #EEF3F8 !important;
  border: 1px solid #D3DDE9 !important; border-radius: 10px !important; box-shadow: none !important;
}
.gradio-container .back-btn button:hover { background: #E4EBF3 !important; }

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container [role="listbox"] {
  color: #172033 !important; background: #FFFFFF !important;
  border-color: #CFD8E5 !important; border-radius: 10px !important;
}
.gradio-container label, .gradio-container .label-wrap { color: #526176 !important; }

.gradio-container .tabs { width: 100% !important; background: transparent !important; }
.gradio-container .tab-nav,
.gradio-container [role="tablist"] {
  gap: 26px !important; padding: 0 !important; background: transparent !important;
  border-bottom: 1px solid #DCE4EE !important; box-shadow: none !important;
}
.gradio-container .tab-nav button {
  min-height: 46px !important; padding: 12px 2px !important;
  color: #667085 !important; background: transparent !important; border: 0 !important;
  border-radius: 0 !important; font-size: .9rem !important; font-weight: 600 !important; box-shadow: none !important;
}
.gradio-container .tab-nav button.selected { color: #32435E !important; border-bottom: 2px solid #7F9FD8 !important; }

.gradio-container .plot-container,
.gradio-container .js-plotly-plot,
.gradio-container .plot-container > div { background: #FFFFFF !important; border-radius: 12px !important; }

.decision-card, .category-profile, .summary-card, .classifier-result {
  margin-top: 18px; padding: 22px; background: #FFFFFF;
  border: 1px solid #DCE4EE; border-radius: 14px;
}
.decision-header, .category-profile-heading, .summary-heading {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 20px;
}
.decision-title, .category-profile h3, .summary-card h3 {
  margin: 5px 0 0; color: #172033; font-size: 1.25rem; font-weight: 700;
}
.decision-copy { margin: 10px 0 18px; color: #667085; line-height: 1.55; }
.decision-badge, .summary-source {
  display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px;
  font-size: .74rem; font-weight: 700; white-space: nowrap;
}
.decision-good { color: #286451; background: #E7F5EF; }
.decision-watch { color: #805D1C; background: #FFF5D9; }
.decision-risk { color: #8A4040; background: #FCEAEA; }
.summary-source { color: #526176; background: #EEF3F8; font-weight: 600; white-space: normal; text-align: right; }

.sentiment-distribution { margin: 18px 0 20px; }
.sentiment-track { display:flex; width:100%; height:13px; overflow:hidden; background:#E5EBF2; border-radius:999px; }
.sentiment-track span { display:block; height:100%; }
.sentiment-positive { background:#62B89F; } .sentiment-neutral { background:#E7B96A; } .sentiment-negative { background:#D98B8B; }
.sentiment-labels { display:flex; flex-wrap:wrap; gap:12px 24px; margin-top:10px; color:#667085; font-size:.8rem; }
.sentiment-labels span { display:inline-flex; align-items:center; gap:6px; }
.sentiment-labels strong { color:#172033; }
.sentiment-labels i { width:8px; height:8px; border-radius:50%; }
.dot-positive { background:#62B89F; } .dot-neutral { background:#E7B96A; } .dot-negative { background:#D98B8B; }
.sentiment-compact { margin-bottom:2px; }

.supporting-metrics, .overview-headline-grid, .profile-counts { display:grid; gap:12px; }
.supporting-metrics { grid-template-columns:repeat(2,minmax(0,1fr)); margin:18px 0; }
.supporting-metrics > div, .headline-card, .profile-counts > div {
  padding:16px; background:#F4F7FB; border:1px solid #DCE4EE; border-radius:11px;
}
.supporting-metrics strong, .headline-card strong, .profile-counts strong { display:block; color:#172033; font-size:1.16rem; }
.supporting-metrics span, .headline-card span, .profile-counts span { display:block; margin-top:4px; color:#667085; font-size:.76rem; }

.two-column-insights, .summary-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin-top:18px; }
.two-column-insights section, .summary-grid section, .summary-takeaway, .insight-block {
  padding:16px; background:#F6F8FB; border:1px solid #DCE4EE; border-radius:11px;
}
.insight-label { display:block; margin-bottom:9px; color:#526176; font-size:.72rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }
.theme-list { margin:0; padding-left:18px; color:#344054; line-height:1.65; font-size:.86rem; }
.evidence-block { margin-top:14px; }
.review-quote { margin-top:8px; padding:12px 14px; color:#526176; background:#FFFFFF; border-left:3px solid #B8C8DD; border-radius:0 8px 8px 0; font-size:.83rem; line-height:1.55; }

.overview-headline-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
.headline-card strong { font-size:1.45rem; }
.portfolio-sentiment { margin-top:16px; padding:18px; background:#F4F7FB; border:1px solid #DCE4EE; border-radius:12px; }
.priority-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin-top:14px; }
.priority-card { padding:18px; border:1px solid #DCE4EE; border-radius:12px; }
.priority-card strong { display:block; margin-top:7px; color:#172033; line-height:1.45; }
.priority-card p { margin:8px 0 0; color:#667085; font-size:.82rem; line-height:1.5; }
.priority-risk { background:#FFF8F7; } .priority-good { background:#F2FAF7; }
.profile-counts { grid-template-columns:repeat(2,minmax(110px,1fr)); min-width:270px; }
.category-evidence { margin-top:12px; }
.summary-takeaway { margin-top:18px; }
.summary-takeaway p, .summary-text { margin:0; color:#344054; font-size:.88rem; line-height:1.65; }
.summary-grid { grid-template-columns:repeat(3,minmax(0,1fr)); }
.summary-products { grid-template-columns:minmax(0,2fr) minmax(0,1fr); margin-top:14px; }

.product-list-card { overflow:hidden; background:#FFFFFF; border:1px solid #DCE4EE; border-radius:12px; }
.product-list-header { padding:15px 16px; background:#F4F7FB; border-bottom:1px solid #DCE4EE; }
.status-good, .status-risk { display:inline-flex; padding:5px 9px; border-radius:999px; font-size:.75rem; font-weight:700; }
.status-good { color:#286451; background:#E7F5EF; } .status-risk { color:#8A4040; background:#FCEAEA; }
.product-list { width:100%; border-collapse:collapse; }
.product-list tr + tr { border-top:1px solid #E4EAF1; }
.product-list td { padding:15px 16px; vertical-align:top; }
.rank-cell { width:36px; color:#667085; font-weight:700; }
.product-name { color:#172033; font-size:.9rem; font-weight:600; }
.product-metrics { margin-top:5px; color:#667085; font-size:.78rem; }

.prediction-summary { padding-bottom:18px; border-bottom:1px solid #DCE4EE; }
.prediction-kicker { color:#667085; font-size:.75rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.prediction-value { margin-top:6px; font-size:1.75rem; font-weight:700; }
.prediction-confidence { margin-top:3px; color:#667085; font-size:.86rem; }
.probabilities { display:grid; gap:13px; padding:20px 0; }
.prob-label { display:flex; justify-content:space-between; margin-bottom:6px; color:#344054; font-size:.84rem; }
.prob-track { height:8px; overflow:hidden; background:#E5EBF2; border-radius:999px; }
.prob-fill { height:100%; background:#7F9FD8; border-radius:999px; }
.prediction-explanation { padding-top:17px; color:#526176; font-size:.88rem; line-height:1.55; border-top:1px solid #DCE4EE; }
.empty-state, .empty-inline { color:#667085; font-size:.84rem; }
.empty-state { padding:18px; margin-top:16px; background:#F4F7FB; border:1px dashed #BFCBDD; border-radius:10px; text-align:center; }

@media (max-width:850px) {
  .app-screen { width:min(100% - 28px,1180px) !important; }
  .mode-header { align-items:flex-start; }
  .summary-grid, .summary-products { grid-template-columns:1fr; }
}
@media (max-width:620px) {
  .app-screen { width:min(100% - 20px,1180px) !important; padding-top:22px !important; }
  .two-column-insights, .supporting-metrics, .overview-headline-grid, .priority-grid, .summary-grid, .summary-products { grid-template-columns:1fr; }
  .decision-header, .category-profile-heading, .summary-heading { flex-direction:column; }
  .profile-counts { width:100%; min-width:0; }
  .gradio-container .tab-nav { gap:14px !important; overflow-x:auto !important; }
}
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

LANDING_HTML = """
<div class="hero">
  <div class="brand">Customer Review <span>Intelligence</span></div>
  <h1>Turn customer reviews into decisions.</h1>
  <p>Use a shopper view for clear product-group evidence or a business view for portfolio, category, product and review intelligence.</p>
</div>
"""


def show_consumer():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def show_seller():
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def show_home():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


with gr.Blocks(
    title="Customer Review Intelligence",
) as app:

    # ---------------- Landing ----------------
    with gr.Column(
        visible=True,
        elem_id="landing-view",
        elem_classes=["view", "app-screen"],
    ) as landing_view:
        gr.HTML(LANDING_HTML, container=False)

        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Group(elem_classes="entry-card"):
                    gr.HTML(
                        '<div class="entry-kicker">For shoppers</div>'
                        '<div class="entry-title">Consumer view</div>'
                        '<div class="entry-copy">'
                        'Understand how a product group performs, what customers '
                        'value and which concerns appear repeatedly.'
                        '</div>',
                        container=False,
                    )
                    enter_consumer_btn = gr.Button(
                        "Open consumer view",
                        variant="primary",
                        elem_classes="consumer-action",
                    )

            with gr.Column():
                with gr.Group(elem_classes="entry-card"):
                    gr.HTML(
                        '<div class="entry-kicker">For business teams</div>'
                        '<div class="entry-title">Seller view</div>'
                        '<div class="entry-copy">'
                        'Find portfolio priorities, explore semantic clusters, '
                        'compare products and classify new reviews.'
                        '</div>',
                        container=False,
                    )
                    enter_seller_btn = gr.Button(
                        "Open seller view",
                        variant="primary",
                    )

    # ---------------- Consumer ----------------
    with gr.Column(
        visible=False,
        elem_id="consumer-view",
        elem_classes=["view", "app-screen"],
    ) as consumer_view:
        with gr.Row(elem_classes="mode-header"):
            with gr.Column(scale=5):
                gr.HTML(
                    '<div class="brand">Customer Review <span>Intelligence</span></div>'
                    '<div class="mode-heading">Consumer view</div>'
                    '<div class="mode-copy">'
                    'Consumer-friendly product groups replace technical cluster '
                    'labels and keep the decision easy to understand.'
                    '</div>',
                    container=False,
                )
            with gr.Column(scale=1, min_width=150):
                back_consumer_btn = gr.Button(
                    "Back to menu",
                    elem_classes="back-btn",
                )

        with gr.Group(elem_classes="content-card"):
            gr.HTML(
                '<div class="section-title">Explore customer feedback</div>'
                '<div class="section-copy">'
                'Select a product group to review its sentiment, rating, '
                'recurring strengths and common concerns.'
                '</div>',
                container=False,
            )

            with gr.Row():
                consumer_product_dd = gr.Dropdown(
                    choices=CATEGORY_OPTIONS,
                    label="Product group",
                    interactive=True,
                    scale=4,
                )
                consumer_product_btn = gr.Button(
                    "Analyse",
                    variant="primary",
                    elem_classes="consumer-action",
                    scale=1,
                    min_width=120,
                )

            consumer_product_out = gr.HTML(
                '<div class="empty-state">'
                'Select a product group to begin.'
                '</div>',
                container=False,
            )

            consumer_product_btn.click(
                fn=consumer_product_check,
                inputs=consumer_product_dd,
                outputs=consumer_product_out,
            )

    # ---------------- Seller ----------------
    default_meta = META_CATEGORIES[0] if META_CATEGORIES else None

    with gr.Column(
        visible=False,
        elem_id="seller-view",
        elem_classes=["view", "app-screen"],
    ) as seller_view:
        with gr.Row(elem_classes="mode-header"):
            with gr.Column(scale=5):
                gr.HTML(
                    '<div class="brand">Customer Review <span>Intelligence</span></div>'
                    '<div class="mode-heading">Seller view</div>'
                    '<div class="mode-copy">'
                    'Four focused workflows with no duplicated dashboards.'
                    '</div>',
                    container=False,
                )
            with gr.Column(scale=1, min_width=150):
                back_seller_btn = gr.Button(
                    "Back to menu",
                    elem_classes="back-btn",
                )

        with gr.Tabs():
            with gr.Tab("Overview"):
                with gr.Group(elem_classes="content-card"):
                    gr.HTML(
                        '<div class="section-title">Portfolio overview</div>'
                        '<div class="section-copy">'
                        'Overall sentiment, the area requiring attention and '
                        'the strongest-performing meta-category.'
                        '</div>',
                        container=False,
                    )
                    gr.HTML(
                        value=seller_overview_html(),
                        container=False,
                    )
                    gr.Plot(
                        value=plot_priority_meta_categories(),
                        show_label=False,
                    )

            with gr.Tab("Category intelligence"):
                with gr.Group(elem_classes="content-card"):
                    gr.HTML(
                        '<div class="section-title">Category intelligence</div>'
                        '<div class="section-copy">'
                        'Explore the semantic clusters, verified sentiment and the final '
                        'quality-audited review summary.'
                        '</div>',
                        container=False,
                    )

                    category_meta_dd = gr.Dropdown(
                        choices=META_CATEGORIES,
                        value=default_meta,
                        label="Meta-category",
                        interactive=True,
                    )

                    category_plot_out = gr.Plot(
                        value=(
                            plot_cluster_3d(default_meta)
                            if default_meta
                            else px.scatter_3d()
                        ),
                        show_label=False,
                    )

                    category_profile_out = gr.HTML(
                        value=(
                            category_profile_html(default_meta)
                            if default_meta
                            else '<div class="empty-state">'
                                 'No meta-categories available.'
                                 '</div>'
                        ),
                        container=False,
                    )

                    category_summary_out = gr.HTML(
                        value=(
                            review_summary_html(default_meta)
                            if default_meta
                            else '<div class="empty-state">'
                                 'No summary available.'
                                 '</div>'
                        ),
                        container=False,
                    )

                    category_meta_dd.change(
                        fn=category_intelligence,
                        inputs=category_meta_dd,
                        outputs=[
                            category_plot_out,
                            category_profile_out,
                            category_summary_out,
                        ],
                    )

            with gr.Tab("Product performance"):
                with gr.Group(elem_classes="content-card"):
                    gr.HTML(
                        '<div class="section-title">Product performance</div>'
                        '<div class="section-copy">'
                        'Products ranked using sentiment and review volume from the '
                        'review sample loaded in the app.'
                        '</div>',
                        container=False,
                    )
                    with gr.Row(equal_height=True):
                        gr.Plot(value=TOP_FIG, show_label=False)
                        gr.Plot(value=WORST_FIG, show_label=False)

            with gr.Tab("Review analyzer"):
                with gr.Group(elem_classes="content-card"):
                    gr.HTML(
                        '<div class="section-title">Review analyzer</div>'
                        '<div class="section-copy">'
                        'Classify a new customer review with the deployed '
                        'Transformer sentiment model.'
                        '</div>',
                        container=False,
                    )
                    seller_review_in = gr.Textbox(
                        label="Customer review",
                        lines=5,
                        placeholder="Paste a customer review here...",
                    )
                    seller_predict_btn = gr.Button(
                        "Analyse review",
                        variant="primary",
                    )
                    seller_classifier_out = gr.HTML(
                        '<div class="empty-state">'
                        'Enter a review to see the prediction.'
                        '</div>',
                        container=False,
                    )
                    seller_predict_btn.click(
                        fn=classifier_result,
                        inputs=seller_review_in,
                        outputs=seller_classifier_out,
                    )

    nav_outputs = [landing_view, consumer_view, seller_view]
    enter_consumer_btn.click(
        fn=show_consumer,
        outputs=nav_outputs,
    )
    enter_seller_btn.click(
        fn=show_seller,
        outputs=nav_outputs,
    )
    back_consumer_btn.click(
        fn=show_home,
        outputs=nav_outputs,
    )
    back_seller_btn.click(
        fn=show_home,
        outputs=nav_outputs,
    )


if __name__ == "__main__":
    share_enabled = os.getenv("GRADIO_SHARE", "true").strip().lower() in {
        "1", "true", "yes", "y"
    }
    app.queue(default_concurrency_limit=2).launch(
        share=share_enabled,
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        theme=THEME,
        css=CUSTOM_CSS,
        js=FORCE_LIGHT_JS,
    )
