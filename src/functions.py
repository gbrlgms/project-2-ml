"""To use this module from a notebook in `notebooks/`:

    import sys
    sys.path.append("..")
    from src.functions import *

"""

from pathlib import Path
import math
from numbers import Real
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLIT = ROOT / "data" / "split"
MODELS = ROOT / "models"

RANDOM_STATE = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def build_dataframe(root_dir):
    """Index image files stored under artist-named folders.

    Recursively scans ``root_dir`` for supported image extensions. Each image
    becomes one row with its path, parent-folder name as ``artist``, and file
    stem as ``title``. The paths are left as ``Path`` objects so callers can
    pass them directly to image utilities.

    Args:
        root_dir: Dataset directory containing ``<artist>/<image>`` folders.

    Returns:
        A DataFrame with ``path``, ``artist``, and ``title`` columns. If no
        supported images are found, the returned DataFrame has no rows.
    """
    root = Path(root_dir)
    rows = [
        {"path": p, "artist": p.parent.name, "title": p.stem}
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return pd.DataFrame(rows)

def plot_artist_grids(df, artists=None, n_rows=3, n_cols=3, seed=RANDOM_STATE,
                      figsize=(10, 10), show_titles=True, max_image_size=512):
    """Display a sampled grid of paintings for each requested artist.

    One figure is created per artist. The sample is capped at the grid
    capacity and uses ``seed`` for repeatable selection. Images are converted
    to RGB and resized in memory to fit within ``max_image_size``; this does
    not modify the source files. Unreadable images are represented by a
    placeholder, and unused grid cells are hidden.

    Args:
        df: DataFrame with ``artist`` and ``path`` columns; ``title`` is
            optional and shown when present and ``show_titles`` is true.
        artists: Artists to plot. Defaults to every artist in first-seen
            order in ``df``.
        n_rows: Number of grid rows per artist.
        n_cols: Number of grid columns per artist.
        seed: Random seed used when sampling images.
        figsize: Matplotlib figure size for each artist.
        show_titles: Whether to label each image with its title.
        max_image_size: Maximum width or height, in pixels, for displayed
            thumbnails.

    Returns:
        None. Figures are displayed and closed as a side effect.
    """
    n = n_rows * n_cols
    if artists is None:
        artists = df["artist"].unique()

    for artist in artists:
        sub = df[df["artist"] == artist]
        if sub.empty:
            print(f"No images found for '{artist}'")
            continue
        sample = sub.sample(min(n, len(sub)), random_state=seed)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        for ax in axes.ravel():
            ax.axis("off")                       # blank cells if fewer than n images
        for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
            try:
                with Image.open(row["path"]) as source:
                    image = source.convert("RGB")
                    image.thumbnail((max_image_size, max_image_size))
                ax.imshow(image)
                image.close()
            except (OSError, ValueError):
                ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center")
            if show_titles and "title" in df.columns:
                ax.set_title(str(row["title"])[:30], fontsize=7)

        header = f"{artist}  |  {len(sub)} images"
        fig.suptitle(header, fontsize=12)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


def plot_phash_duplicates_by_artist(
    duplicates,
    df,
    root_dir,
    top_n_artists=8,
    max_pairs_per_artist=4,
    max_image_size=512,
):
    """Count and display perceptual-hash duplicate pairs by artist.

    Cross-artist pairs count toward both artists, making possible attribution
    mismatches visible in each artist's gallery. Reversed duplicate entries
    from the PHash result are collapsed to one pair. A bar chart shows the
    artists with the most pairs, followed by a small gallery of matching
    images for each shown artist. PHash names are interpreted relative to
    ``root_dir``; when a name is absent from ``df``, its parent folder is used
    as the artist label.

    Args:
        duplicates: Mapping of image names to PHash matches. A match may be a
            filename or a tuple/list whose first item is the filename.
        df: DataFrame containing image ``path`` and ``artist`` columns.
        root_dir: Root folder from which PHash paths are relative.
        top_n_artists: Maximum artists to include in the chart and galleries.
        max_pairs_per_artist: Maximum pairs displayed for each artist.
        max_image_size: Maximum width or height, in pixels, for thumbnails.

    Returns:
        A DataFrame with one row per artist and a ``duplicate_pairs`` count.
        If there are no matches, an empty count DataFrame is returned.
    """
    root = Path(root_dir).resolve()
    artist_by_name = {
        Path(path).resolve().relative_to(root).as_posix(): artist
        for path, artist in zip(df["path"], df["artist"])
    }

    def artist_for_name(name):
        """Resolve a PHash relative filename to its artist label."""
        normalized_name = Path(name).as_posix()
        if normalized_name in artist_by_name:
            return artist_by_name[normalized_name]
        # PHash paths retain the source folder structure (<artist>/<image>),
        # so this also works after duplicate rows have been removed from df.
        folder_artist = Path(normalized_name).parent.name
        return folder_artist if folder_artist not in {"", "."} else "Unknown artist"

    unique_pairs = set()
    for name, matches in duplicates.items():
        for match in matches:
            match_name = match[0] if isinstance(match, (tuple, list)) else match
            pair = tuple(sorted((str(name), str(match_name))))
            if pair[0] != pair[1]:
                unique_pairs.add(pair)

    pairs_by_artist = {}
    for left, right in unique_pairs:
        left_artist = artist_for_name(left)
        right_artist = artist_for_name(right)
        pairs_by_artist.setdefault(left_artist, []).append((left, right))
        if right_artist != left_artist:
            pairs_by_artist.setdefault(right_artist, []).append((left, right))

    counts = (
        pd.Series({artist: len(pairs) for artist, pairs in pairs_by_artist.items()},
                  name="duplicate_pairs")
        .sort_values(ascending=False)
        .rename_axis("artist")
        .reset_index()
    )
    if counts.empty:
        print("No PHash matches to plot.")
        return counts

    shown = counts.head(top_n_artists)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(shown))))
    sns.barplot(data=shown, y="artist", x="duplicate_pairs", color="steelblue", ax=ax)
    ax.set(title="Unique PHash duplicate pairs involving each artist",
           xlabel="Number of pairs", ylabel="Artist")
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    for artist in shown["artist"]:
        pairs = pairs_by_artist[artist][:max_pairs_per_artist]
        fig, axes = plt.subplots(len(pairs), 2,
                                 figsize=(9, 3.4 * len(pairs)), squeeze=False)
        for row_index, pair in enumerate(pairs):
            for col_index, name in enumerate(pair):
                image_path = root / name
                ax = axes[row_index, col_index]
                ax.axis("off")
                try:
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                        image.thumbnail((max_image_size, max_image_size))
                    ax.imshow(image)
                    image.close()
                    pair_artist = artist_for_name(name)
                    ax.set_title(f"{pair_artist}\n{Path(name).name[:36]}", fontsize=8)
                except (OSError, ValueError):
                    ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center")
        fig.suptitle(f"{artist}: up to {len(pairs)} matching pairs", fontsize=12)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    return counts


def remove_phash_duplicates(df, duplicates, root_dir):
    """Keep one representative row per connected PHash duplicate group.

    Duplicate links are treated as an undirected graph: connected components
    form duplicate groups, including transitive matches. For groups with
    conflicting artist labels, retain a row from the most common artist. Ties
    are broken by image resolution (when available), then by relative path
    for deterministic results. If resolution columns are absent, path order
    breaks the remaining ties. Rows not mentioned in a match remain untouched.

    Args:
        df: DataFrame containing image ``path`` and ``artist`` columns. If
            ``megapixels`` or both ``width`` and ``height`` are present, they
            are used to prefer higher-resolution representatives.
        duplicates: Mapping of relative image names to PHash matches.
        root_dir: Root directory used to convert stored paths into relative
            names matching the PHash output.

    Returns:
        ``(retained, removed)`` DataFrames, each with a fresh integer index.
    """
    root = Path(root_dir).resolve()
    path_to_name = {
        Path(path).resolve().relative_to(root).as_posix(): index
        for index, path in df["path"].items()
    }
    name_to_index = {name: index for name, index in path_to_name.items()}
    parent = {name: name for name in name_to_index}

    def find(name):
        """Find a component representative and compress the traversed path."""
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left, right):
        """Join the connected components containing two image names."""
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for name, matches in duplicates.items():
        if name not in parent:
            continue
        for match in matches:
            match_name = match[0] if isinstance(match, (tuple, list)) else match
            match_name = str(match_name)
            if match_name in parent:
                union(name, match_name)

    groups = {}
    for name in parent:
        groups.setdefault(find(name), []).append(name)

    resolution = {}
    if "megapixels" in df.columns:
        resolution = df["megapixels"].to_dict()
    elif {"width", "height"}.issubset(df.columns):
        resolution = (df["width"] * df["height"]).to_dict()

    def get_resolution(name):
        """Return a numeric resolution value, treating missing values as zero."""
        value = resolution.get(name_to_index[name], 0)
        return float(value) if pd.notna(value) else 0.0

    drop_indices = set()
    for names in groups.values():
        if len(names) < 2:
            continue
        artist_counts = {}
        for name in names:
            artist = df.at[name_to_index[name], "artist"]
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
        names.sort(
            key=lambda name: (
                -artist_counts[df.at[name_to_index[name], "artist"]],
                -get_resolution(name),
                name,
            )
        )
        drop_indices.update(name_to_index[name] for name in names[1:])

    removed = df.loc[sorted(drop_indices)].copy()
    retained = df.drop(index=drop_indices).copy()
    return retained.reset_index(drop=True), removed.reset_index(drop=True)

def add_image_sizes(df):
    """Add image dimensions and derived size metrics to a copy of ``df``.

    Reads each image's metadata through Pillow without decoding pixels. For
    unreadable files, width and height are set to ``None``; derived columns
    then remain missing as well.

    Args:
        df: DataFrame with a ``path`` column.

    Returns:
        A copy of the input with ``width``, ``height``, ``aspect`` (width /
        height), and ``megapixels`` (width * height / 1,000,000) columns.
    """
    sizes = []
    for p in df["path"]:
        try:
            with Image.open(p) as img:
                sizes.append(img.size)
        except Exception:                     # malformed or inaccessible image
            sizes.append((None, None))          # unreadable file
    out = df.copy()
    out[["width", "height"]] = pd.DataFrame(sizes, index=out.index)
    out["aspect"] = out["width"] / out["height"]
    out["megapixels"] = out["width"] * out["height"] / 1e6
    return out

def plot_size_ranges(df, top_n=15):
    """Plot width, height, and aspect-ratio distributions for common artists.

    Artists are selected by row count, then ordered by median megapixels so
    the three boxplots use the same readable artist order. This function
    expects ``width``, ``height``, ``aspect``, and ``megapixels`` columns,
    typically added by :func:`add_image_sizes`.

    Args:
        df: Image metadata DataFrame with ``artist`` and size columns.
        top_n: Number of most frequent artists to include.

    Returns:
        None. Displays a Matplotlib figure.
    """
    top = df["artist"].value_counts().head(top_n).index
    d = df[df["artist"].isin(top)]
    order = d.groupby("artist")["megapixels"].median().sort_values().index

    fig, axes = plt.subplots(1, 3, figsize=(18, 0.4 * top_n + 2), sharey=True)
    for ax, col, title in zip(
        axes,
        ["width", "height", "aspect"],
        ["Width (px)", "Height (px)", "Aspect ratio (w/h)"],
    ):
        sns.boxplot(data=d, y="artist", x=col, order=order, ax=ax, fliersize=2)
        ax.set_title(title)
        ax.set_ylabel("")
    axes[2].axvline(1, color="grey", linestyle="--", linewidth=1)   # square images
    plt.tight_layout()
    plt.show()


def get_class_mapping(df):
    """Create a stable integer encoding for artist labels.

    Labels are sorted before assigning indices, so the same set of artist
    names always receives the same mapping regardless of row order.

    Args:
        df: DataFrame with an ``artist`` column.

    Returns:
        A pair ``(classes, class_to_idx)`` where ``classes`` is the sorted
        list of artist names and ``class_to_idx`` maps each name to its index.
    """
    classes = sorted(df["artist"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx


def data_split(
    df,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    random_state=RANDOM_STATE,
    output_dir=None,
):
    """Create stratified train, validation, and test splits without title leakage.

    Sizes are real-number proportions (for example, 0.6) that must be positive
    and sum to 1.0. StratifiedGroupKFold uses equal-sized folds, so actual
    proportions are the closest achievable while keeping every title group
    together. The function first selects one outer fold for the test set, then
    selects an inner fold from the remaining data for validation. Candidate
    fold counts are capped by the number of distinct title groups in the
    least represented artist. Assertions verify that no title appears in
    more than one split. Finally, the split frames are reset, summarized, and
    written as CSV files to ``output_dir`` or the project notebooks directory
    by default.

    Args:
        df: DataFrame containing at least ``artist`` and ``title`` columns.
        train_size: Requested fraction for training data.
        val_size: Requested fraction for validation data.
        test_size: Requested fraction for test data.
        random_state: Seed controlling the shuffled group folds.
        output_dir: Destination for ``split_train.csv``, ``split_val.csv``,
            and ``split_test.csv``. Defaults to the project's notebooks
            directory.

    Returns:
        ``(train, val, test)`` DataFrames with fresh integer indexes. Because
        folds are discrete, their actual sizes can differ slightly from the
        requested proportions.

    Raises:
        ValueError: If proportions are invalid, do not sum to one, or there
            are too few title groups per artist to create the requested split.
    """
    requested = (train_size, val_size, test_size)
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
            for value in requested
        )
        or not math.isclose(sum(requested), 1.0, rel_tol=0, abs_tol=1e-9)
    ):
        raise ValueError(
            "train_size, val_size, and test_size must be positive real-number "
            "proportions that sum to 1.0."
        )

    target = requested
    class_group_counts = df.groupby("artist")["title"].nunique()
    max_outer_splits = min(50, int(class_group_counts.min()))
    if max_outer_splits < 2:
        raise ValueError("Each artist needs at least two distinct title groups to split.")

    # One outer fold forms the test set; choose the fold count closest to its target.
    outer_splits = min(
        range(2, max_outer_splits + 1),
        key=lambda folds: (abs(1 / folds - target[2]), folds),
    )
    outer = StratifiedGroupKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    trainval_idx, test_idx = next(outer.split(df, df["artist"], df["title"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    trainval_fraction = len(trainval) / len(df)
    actual_test_fraction = len(test) / len(df)
    trainval_group_counts = trainval.groupby("artist")["title"].nunique()
    max_inner_splits = min(50, int(trainval_group_counts.min()))
    if max_inner_splits < 2:
        raise ValueError("The train/validation portion has too few title groups to split.")

    # Choose the inner fold count that best matches all three requested shares.
    def inner_split_error(folds):
        """Measure squared deviation from all requested split proportions."""
        val_fraction = trainval_fraction / folds
        train_fraction = trainval_fraction - val_fraction
        actual = (train_fraction, val_fraction, actual_test_fraction)
        return sum((got - want) ** 2 for got, want in zip(actual, target))

    inner_splits = min(
        range(2, max_inner_splits + 1),
        key=lambda folds: (inner_split_error(folds), folds),
    )
    inner = StratifiedGroupKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state
    )
    train_idx, val_idx = next(
        inner.split(trainval, trainval["artist"], trainval["title"])
    )
    train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]

    assert not set(train.title) & set(val.title)
    assert not set(train.title) & set(test.title)
    assert not set(val.title) & set(test.title)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    total_rows = len(train) + len(val) + len(test)
    for name, split_df, target_size in [
        ("train", train, train_size),
        ("val", val, val_size),
        ("test", test, test_size),
    ]:
        actual_pct = 100 * len(split_df) / total_rows
        print(
            f"{name}: {len(split_df)} rows ({actual_pct:.1f}%; "
            f"requested {target_size:.1%})"
        )
        print(split_df["artist"].value_counts(), "\n")

    split_dir = Path(output_dir) if output_dir is not None else ROOT / "notebooks"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        csv_path = split_dir / f"split_{name}.csv"
        split_df.to_csv(csv_path, index=False)
        print(f"Saved {name} split to {csv_path}")

    return train, val, test
