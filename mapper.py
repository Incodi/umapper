import os
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap
import lzma
from lexicalrichness import LexicalRichness # For MTLD calculation
import argparse


def hms_to_seconds(hms):

    """Convert time string like '1:23:45' to seconds"""

    if not hms:

        return 0

    

    parts = hms.split(':')

    parts = [float(p) for p in parts]

    

    # Handle different formats: SS, MM:SS, HH:MM:SS

    if len(parts) == 1:

        return parts[0]  # seconds only

    elif len(parts) == 2:

        return parts[0] * 60 + parts[1]  # minutes:seconds

    elif len(parts) == 3:

        return parts[0] * 3600 + parts[1] * 60 + parts[2]  # hours:minutes:seconds

    return 0


def extract_video_id(filename):

    """Get video ID from filename"""

    # Look for pattern like [VIDEO_ID] on the filenames

    match = re.search(r'\[([A-Za-z0-9_-]{11})\]', filename)

    if match:

        return match.group(1)

    return filename.replace('.txt', '').split('.')[-1]


def tokenize_text(content):

    """Split text into words"""

    # Removes special characters but keep words with apostrophes

    text = re.sub(r"([^\w\s']|'(?!\w)|\s'+|'+\s)", ' ', content)

    return text.lower().split()


def load_metadata(channel_name):

    """Load metadata.json file for a channel"""

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    meta_file = os.path.join(base_dir, "data", "input", channel_name, "metadata.json")

    

    if not os.path.exists(meta_file):

        return {}

    

    with open(meta_file, 'r', encoding='utf-8') as f:

        data = json.load(f)

        # Convert to dictionary with video ID as key

        return {v['id']: v for v in data if 'id' in v}

    

    return {}


def compute_lzma_ratio(chunks, max_samples=1500):

    """

    Calculate LZMA compression ratio for text chunks

    using lzma library. Is entropy approximation.

    """

    if not chunks:

        return 0.0

    

    # Sample chunks if there are too many

    if len(chunks) > max_samples:

        step = max(1, len(chunks) // max_samples)

        chunks = [chunks[i] for i in range(0, len(chunks), step)][:max_samples]

    

    # Join all chunks into one big text

    text = ' '.join(chunks)

    text_bytes = text.encode('utf-8')

    

    if not text_bytes:

        return 0.0

    

    try:

        compressed = lzma.compress(text_bytes)

        return len(compressed) / len(text_bytes)

    except:

        return 0.0


def compute_mtld_score(chunks, max_samples=1500):

    """

    Calculate MTLD score for text chunks

    using the lexicalrichness library.

    """

    if not chunks:

        return 0.0

    

    # Sample chunks if there are too many

    if len(chunks) > max_samples:

        step = max(1, len(chunks) // max_samples)

        chunks = [chunks[i] for i in range(0, len(chunks), step)][:max_samples]

    

    text = ' '.join(chunks)

    

    try:

        lex = LexicalRichness(text)

        return lex.mtld(threshold=0.72)

    except:

        return 0.0


def process_channel(channel_name, filters, model):

    """Process a single channel and return its features"""

    print(f"Processing: {channel_name}")

    

    # Get paths with os.path

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    channel_path = os.path.join(base_dir, "data", "input", channel_name)

    txt_dir = os.path.join(channel_path, "txt_files")

    

    # Check if channel exists

    if not os.path.exists(channel_path):

        print(f"  Skipped: Channel directory doesn't exist: {channel_path}")

        return None

    

    if not os.path.exists(txt_dir):

        print(f"  Skipped: No txt_files directory: {txt_dir}")

        return None

    

    # Get all text files

    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    if not txt_files:

        print(f"  Skipped: No text files found!")

        return None

    

    # Load metadata

    metadata = load_metadata(channel_name)

    

    # Prepare filters

    min_dur = hms_to_seconds(filters.get('duration_from', '')) if filters.get('duration_from') else None

    max_dur = hms_to_seconds(filters.get('duration_to', '')) if filters.get('duration_to') else None

    date_from = filters.get('date_from')

    date_to = filters.get('date_to')

    exclude_live = filters.get('exclude_live', False)

    min_tokens_per_file = filters.get('min_tokens_per_file', 0)

    

    # Collect files that pass filters

    files_to_process = []

    

    for filename in txt_files:

        filepath = os.path.join(txt_dir, filename)

        video_id = extract_video_id(filename)

        meta = metadata.get(video_id, {})

        

        # Apply video-level filters

        upload_date = meta.get('upload_date', '99999999')

        

        if date_from and upload_date < date_from:

            continue

        if date_to and upload_date > date_to:

            continue

        

        duration = meta.get('duration', 0)

        if min_dur and duration < min_dur:

            continue

        if max_dur and duration > max_dur:

            continue

        

        if exclude_live:

            was_live = meta.get('was_live', '') == 'True'

            is_live = meta.get('is_live', '') == 'True'

            if was_live or is_live:

                continue

        

        files_to_process.append((upload_date, filepath, video_id, meta))

    

    # Sort by upload date (oldest first)

    files_to_process.sort()

    

    # Read and tokenize files

    all_tokens = []

    files_processed = 0

    

    for upload_date, filepath, video_id, meta in files_to_process:

        try:

            with open(filepath, 'r', encoding='utf-8') as f:

                content = f.read()

            

            tokens = tokenize_text(content)

            

            # Skip files with too few tokens if the user wants it

            if len(tokens) < min_tokens_per_file:

                continue

            

            all_tokens.extend(tokens)

            files_processed += 1

            

        except Exception as e:

            print(f"    Warning: Could not read {filepath}: {e}")

            continue

    

    if not all_tokens:

        print(f"  Skipped: No tokens extracted from files!")

        return None

    

    # Apply token limit if specified

    token_limit = filters.get('token_limit')

    if token_limit and token_limit > 0 and len(all_tokens) > token_limit:

        all_tokens = all_tokens[-token_limit:]

        print(f"    Applied token limit: kept newest {token_limit:,} tokens")

    

    # Check minimum total tokens

    min_tokens_total = filters.get('min_tokens_total', 0)

    if len(all_tokens) < min_tokens_total:

        print(f"  Skipped: Only {len(all_tokens):,} tokens (minimum: {min_tokens_total:,})")

        return None

    

    # Create chunks

    chunk_size = 256

    min_chunk = 200

    

    chunks = [] # Create text chunks

    for i in range(0, len(all_tokens), chunk_size):

        chunk_tokens = all_tokens[i:i + chunk_size]

        if len(chunk_tokens) >= min_chunk:

            chunks.append(' '.join(chunk_tokens))

    

    if not chunks:

        print(f"  Skipped: No chunks were created!")

        return None

    

    # Compute metrics

    compression_ratio = compute_lzma_ratio(chunks)

    mtld_score = compute_mtld_score(chunks)

    

    # Get embeddings

    chunk_embeddings = model.encode(

        chunks,

        batch_size=256,

        show_progress_bar=False,

        convert_to_numpy=True,

        normalize_embeddings=False

    )

    

    # Average all chunk embeddings to get channel embedding

    channel_embedding = np.mean(chunk_embeddings, axis=0).astype(np.float32)

    

    print(f"  Done: {channel_name} ({len(all_tokens):,} tokens, {len(chunks)} chunks, {files_processed} files)")

    

    return channel_embedding, compression_ratio, mtld_score


def visualize_results(reduced_embeddings, channel_names, compression_ratios=None, mtld_scores=None, 

                     coloring=None):

    """Create a visualization of the channels"""


    plt.figure(figsize=(14, 10))

    

    # Determine what to color the points by

    if coloring == 'compression' and compression_ratios is not None:

        colors = compression_ratios

        color_label = 'LZMA Compression Ratio'

        cmap = plt.cm.RdBu_r

        title = 'YouTube Channels - Colored by LZMA Compression Ratio'

    elif coloring == 'mtld' and mtld_scores is not None:

        colors = mtld_scores

        color_label = 'MTLD Score'

        cmap = plt.cm.plasma

        title = 'YouTube Channels - Colored by MTLD Score'

    else:

        # Default: cluster with KMeans

        n_clusters = min(10, max(2, len(channel_names) // 2))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        labels = kmeans.fit_predict(reduced_embeddings)

        colors = labels

        color_label = 'Cluster'

        cmap = plt.cm.viridis

        title = 'YouTube Channels - Clustered by Semantic Similarity'

    

    # Create scatter plot

    scatter = plt.scatter(

        reduced_embeddings[:, 0],

        reduced_embeddings[:, 1],

        c=colors,

        cmap=cmap,

        s=200,

        alpha=0.7,

        edgecolors='black',

        linewidth=1

    )

    

    # Add colorbar

    cbar = plt.colorbar(scatter)

    cbar.set_label(color_label)

    

    # Add channel names as annotations

    for i, name in enumerate(channel_names):

        plt.annotate(

            name,

            (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),

            fontsize=8,

            alpha=0.8,

            ha='center',

            va='center'

        )

    

    # Customize plot

    plt.title(title, fontsize=16, pad=20)

    plt.xlabel('UMAP Component 1', fontsize=12)

    plt.ylabel('UMAP Component 2', fontsize=12)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    

    # Show plot

    plt.show()


def main():

    # Parse command line arguments

    parser = argparse.ArgumentParser(description='Simple YouTube Channel Classifier')

    parser.add_argument('channels', nargs='+', help='Channel folder names to process')

    

    # Filter add arguments

    parser.add_argument('--date-from',          help='Only include videos from this date (YYYYMMDD)')

    parser.add_argument('--date-to',            help='Only include videos up to this date (YYYYMMDD)')

    parser.add_argument('--duration-from',      help='Minimum video duration (HH:MM:SS)')

    parser.add_argument('--duration-to',        help='Maximum video duration (HH:MM:SS)')

    parser.add_argument('--exclude-live', action='store_true', help='Exclude live stream videos')

    parser.add_argument('--min-words-per-file', type=int, help='Skip files with fewer than this many tokens')

    parser.add_argument('--min-words', type=int, help='Skip channels with fewer than this many total tokens')

    parser.add_argument('--token-limit', type=int, help='Use only the newest N tokens per channel')

    

    # Visualization options

    parser.add_argument('--compression-coloring', action='store_true', help='Color by LZMA compression ratio')

    parser.add_argument('--mtld-coloring', action='store_true', help='Color by MTLD score')

    

    args = parser.parse_args()

    

    # Filters dictionary

    filters = {

        'date_from': args.date_from,

        'date_to': args.date_to,

        'duration_from': args.duration_from,

        'duration_to': args.duration_to,

        'exclude_live': args.exclude_live,

        'min_words_per_file': args.min_words_per_file if args.min_words_per_file else 0,

        'min_words': args.min_words if args.min_words else 0,

        'token_limit': args.token_limit

    }

    

    # Load the embedding model here

    print("Loading sentence transformer model...")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("SBERT Model loaded!\n")

    

    # Process each channel

    successful_channels = []

    embeddings_list = []

    compression_ratios = []

    mtld_scores = []

    

    for channel in args.channels:

        result = process_channel(channel, filters, model)

        

        if result is not None:

            embedding, comp_ratio, mtld_score = result

            

            successful_channels.append(channel)

            embeddings_list.append(embedding)

            compression_ratios.append(comp_ratio)

            mtld_scores.append(mtld_score)

    

    if not successful_channels:

        print("\nError: No channels were successfully processed!")

        return

    

    print(f"\nSuccessfully processed {len(successful_channels)}/{len(args.channels)} channels")

    

    # Convert to numpy arrays

    embeddings_matrix = np.array(embeddings_list)

    compression_ratios = np.array(compression_ratios)

    mtld_scores = np.array(mtld_scores)

    

    # UMAP needs at least 2 samples; stop early with a friendly message

    if len(successful_channels) < 2:

        print("\nOnly one channel was processed; skipping UMAP/plotting (need >=2).")

        print("Results:")

        for i, channel in enumerate(successful_channels):

            print(f"{channel}:")

            print(f"  Compression Ratio: {compression_ratios[i]:.4f}")

            print(f"  MTLD Score: {mtld_scores[i]:.2f}")

        return


    # Reduce dimensions with UMAP

    print("Reducing dimensions with UMAP...")

    n_channels = len(successful_channels)

    n_neighbors = min(15, max(5, n_channels // 2))

    

    reducer = umap.UMAP(

        n_components=2,

        metric='cosine',

        n_neighbors=n_neighbors,

        min_dist=0.1,

        random_state=42, 

        n_epochs=200

    )

    

    reduced_embeddings = reducer.fit_transform(embeddings_matrix)

    print("UMAP complete!\n")

    

    # Determine coloring option

    coloring = None

    if args.compression_coloring:

        coloring = 'compression'

    elif args.mtld_coloring:

        coloring = 'mtld'

    

    # Visualize results

    print("Creating visualization...")

    visualize_results(

        reduced_embeddings,

        successful_channels,

        compression_ratios=compression_ratios,

        mtld_scores=mtld_scores,

        coloring=coloring

    )

    

    # Print summary

    print("\n-Summary-")

    for i, channel in enumerate(successful_channels):

        print(f"{channel}:")

        print(f"  Compression Ratio: {compression_ratios[i]:.4f}")

        print(f"  MTLD Score: {mtld_scores[i]:.2f}")


if __name__ == "__main__":
    main()




    # Example command:

    # python3 src/mapper.py animalogic bogxd mkbhd eons --date-from 20200101 --duration-from 8:00 --exclude-live --token-limit 400000 --min-words 400000 --compression-coloring
