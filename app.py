import os
import io
import base64
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'kmeans-jabar-2024-secret')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads') 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {'csv'}
CLUSTER_LABELS = {
    0: ('Sangat Rendah', '#2ecc71', 'success'),
    1: ('Rendah', '#3498db', 'info'),
    2: ('Sedang', '#f39c12', 'warning'),
    3: ('Tinggi', '#e74c3c', 'danger'),
    4: ('Sangat Tinggi', '#8e44ad', 'dark'),
}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#f8f9fa', edgecolor='none')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)

    # Drop rows with missing critical values
    required_cols = ['nama_kabupaten_kota', 'jumlah_cuaca_ekstrem', 'tahun']
    df = df.dropna(subset=required_cols)
    n_missing = n_before - n_dupes - len(df)

    # Ensure numeric
    df['jumlah_cuaca_ekstrem'] = pd.to_numeric(df['jumlah_cuaca_ekstrem'], errors='coerce')
    df['tahun'] = pd.to_numeric(df['tahun'], errors='coerce')
    df = df.dropna(subset=['jumlah_cuaca_ekstrem', 'tahun'])

    # Clean region names
    df['nama_kabupaten_kota'] = df['nama_kabupaten_kota'].str.strip().str.title()

    return df, n_dupes, n_missing


def build_feature_matrix(df):
    """Aggregate per kabupaten/kota: total, mean, max, std kejadian."""
    agg = df.groupby('nama_kabupaten_kota')['jumlah_cuaca_ekstrem'].agg(
        total='sum',
        rata_rata='mean',
        maksimum='max',
        std_dev='std'
    ).reset_index()
    agg['std_dev'] = agg['std_dev'].fillna(0)
    return agg


def run_elbow_silhouette(X_scaled, k_range=(2, 8)):
    inertias, scores, k_values = [], [], []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        s = silhouette_score(X_scaled, labels)
        scores.append(s)
        k_values.append(k)
    return k_values, inertias, scores


def find_optimal_k(k_values, inertias, scores):
    """Find optimal K via elbow (second derivative) + silhouette."""
    # Elbow: largest drop (second derivative)
    if len(inertias) >= 3:
        deltas = np.diff(inertias)
        second_deriv = np.diff(deltas)
        elbow_idx = np.argmax(np.abs(second_deriv)) + 1
        elbow_k = k_values[elbow_idx]
    else:
        elbow_k = k_values[0]

    # Silhouette: highest score
    sil_k = k_values[np.argmax(scores)]

    # Vote: if they agree, use that; else prefer silhouette
    optimal = elbow_k if elbow_k == sil_k else sil_k
    return optimal, elbow_k, sil_k


def assign_cluster_labels(agg_df, n_clusters):
    """Assign human-readable rawan labels based on cluster mean total."""
    cluster_means = agg_df.groupby('cluster')['total'].mean().sort_values()
    rank_map = {cluster_id: rank for rank, cluster_id in enumerate(cluster_means.index)}

    # Evenly distribute labels across 5 levels
    step = 5 / n_clusters
    label_map = {}
    for cluster_id, rank in rank_map.items():
        level = min(int(rank * step), 4)
        label_map[cluster_id] = CLUSTER_LABELS[level]
    return label_map


def plot_elbow(k_values, inertias, optimal_k, elbow_k):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    ax.plot(k_values, inertias, 'o-', color='#1a6e3c', linewidth=2.5,
            markersize=7, markerfacecolor='#ffffff', markeredgewidth=2, markeredgecolor='#1a6e3c')
    ax.axvline(x=optimal_k, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8,
               label=f'K Optimal = {optimal_k}')
    ax.set_xlabel('Jumlah Klaster (K)', fontsize=11, labelpad=8)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11, labelpad=8)
    ax.set_title('Elbow Method — Penentuan Jumlah Klaster Optimal', fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_silhouette(k_values, scores, optimal_k):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    colors = ['#1a5276' if k == optimal_k else '#85c1e9' for k in k_values]
    bars = ax.bar(k_values, scores, color=colors, width=0.55, edgecolor='white', linewidth=1.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(y=max(scores), color='#c0392b', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Skor Tertinggi = {max(scores):.3f}')
    ax.set_xlabel('Jumlah Klaster (K)', fontsize=11, labelpad=8)
    ax.set_ylabel('Silhouette Score', fontsize=11, labelpad=8)
    ax.set_title('Silhouette Score — Kualitas Klaster per Nilai K', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(k_values)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_cluster_bar(agg_df, label_map):
    df_plot = agg_df.copy()
    df_plot['label'] = df_plot['cluster'].map(lambda c: label_map[c][0])
    df_plot['color'] = df_plot['cluster'].map(lambda c: label_map[c][1])
    df_plot = df_plot.sort_values('total', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot) * 0.32)))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    bars = ax.barh(df_plot['nama_kabupaten_kota'], df_plot['total'],
                   color=df_plot['color'], edgecolor='white', linewidth=0.8, height=0.7)
    for bar, val in zip(bars, df_plot['total']):
        ax.text(bar.get_width() + max(df_plot['total']) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{int(val)}', va='center', fontsize=8)
    ax.set_xlabel('Total Kejadian Bencana Cuaca Ekstrem', fontsize=11, labelpad=8)
    ax.set_title('Distribusi Total Kejadian per Kabupaten/Kota\nBerdasarkan Klaster Kerawanan', fontsize=12, fontweight='bold', pad=12)
    # Legend
    unique_labels = {}
    for c, info in label_map.items():
        unique_labels[info[0]] = info[1]
    patches = [mpatches.Patch(color=col, label=lbl) for lbl, col in unique_labels.items()]
    ax.legend(handles=patches, loc='lower right', fontsize=9, framealpha=0.8)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_scatter(agg_df, label_map):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    for cluster_id, info in label_map.items():
        subset = agg_df[agg_df['cluster'] == cluster_id]
        ax.scatter(subset['rata_rata'], subset['maksimum'],
                   c=info[1], label=f"Klaster {cluster_id}: {info[0]}",
                   s=90, edgecolors='white', linewidth=1.2, alpha=0.9, zorder=3)
    ax.set_xlabel('Rata-rata Kejadian per Tahun', fontsize=11, labelpad=8)
    ax.set_ylabel('Kejadian Maksimum', fontsize=11, labelpad=8)
    ax.set_title('Scatter Plot K-Means: Rata-rata vs Maksimum Kejadian', fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig_to_base64(fig)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        flash('Tidak ada file yang dipilih.', 'danger')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('Pilih file CSV terlebih dahulu.', 'danger')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Hanya file format CSV yang diterima.', 'danger')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # ── 1. Load & Clean ──────────────────────────────────────────────
        df, n_dupes, n_missing = load_and_clean_data(filepath)
        n_rows = len(df)
        n_regions = df['nama_kabupaten_kota'].nunique()
        year_range = f"{int(df['tahun'].min())} – {int(df['tahun'].max())}"

        # ── 2. Feature Engineering ───────────────────────────────────────
        agg = build_feature_matrix(df)
        features = ['total', 'rata_rata', 'maksimum', 'std_dev']
        X = agg[features].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # ── 3. Elbow + Silhouette ─────────────────────────────────────────
        k_max = min(8, n_regions - 1)
        k_min = 2
        k_values, inertias, sil_scores = run_elbow_silhouette(X_scaled, (k_min, k_max))

        # Override K if user supplied
        user_k = request.form.get('n_clusters', '').strip()
        if user_k and user_k.isdigit() and k_min <= int(user_k) <= k_max:
            optimal_k = int(user_k)
            elbow_k, sil_k = optimal_k, optimal_k
            k_source = f'Manual (K = {optimal_k})'
        else:
            optimal_k, elbow_k, sil_k = find_optimal_k(k_values, inertias, sil_scores)
            k_source = f'Otomatis (Elbow={elbow_k}, Silhouette={sil_k}) → K={optimal_k}'

        final_silhouette = sil_scores[k_values.index(optimal_k)]

        # ── 4. Final KMeans ───────────────────────────────────────────────
        km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        agg['cluster'] = km_final.fit_predict(X_scaled)
        label_map = assign_cluster_labels(agg, optimal_k)
        agg['tingkat_kerawanan'] = agg['cluster'].map(lambda c: label_map[c][0])
        agg['badge_class'] = agg['cluster'].map(lambda c: label_map[c][2])

        # ── 5. Plots ──────────────────────────────────────────────────────
        img_elbow = plot_elbow(k_values, inertias, optimal_k, elbow_k)
        img_silhouette = plot_silhouette(k_values, sil_scores, optimal_k)
        img_bar = plot_cluster_bar(agg, label_map)
        img_scatter = plot_scatter(agg, label_map)

        # ── 6. Table Data ─────────────────────────────────────────────────
        table_df = agg[['nama_kabupaten_kota', 'total', 'rata_rata', 'maksimum',
                         'std_dev', 'cluster', 'tingkat_kerawanan', 'badge_class']].copy()
        table_df = table_df.sort_values('tingkat_kerawanan')
        table_data = table_df.to_dict(orient='records')

        # Cluster summary
        cluster_summary = []
        for cluster_id, info in label_map.items():
            subset = agg[agg['cluster'] == cluster_id]
            cluster_summary.append({
                'cluster': cluster_id,
                'label': info[0],
                'color': info[1],
                'badge': info[2],
                'count': len(subset),
                'regions': ', '.join(sorted(subset['nama_kabupaten_kota'].tolist()))
            })
        cluster_summary.sort(key=lambda x: x['cluster'])

        return render_template('results.html',
            filename=filename,
            n_rows=n_rows,
            n_dupes=n_dupes,
            n_missing=n_missing,
            n_regions=n_regions,
            year_range=year_range,
            optimal_k=optimal_k,
            k_source=k_source,
            final_silhouette=round(final_silhouette, 4),
            img_elbow=img_elbow,
            img_silhouette=img_silhouette,
            img_bar=img_bar,
            img_scatter=img_scatter,
            table_data=table_data,
            cluster_summary=cluster_summary,
        )

    except Exception as e:
        flash(f'Terjadi kesalahan saat memproses data: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.errorhandler(413)
def too_large(e):
    flash('Ukuran file terlalu besar. Maksimum 16 MB.', 'danger')
    return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
