import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
import numpy as np
from collections import Counter

# Set research paper style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette - professional and accessible
COLORS = {
    'primary': '#2c3e50',
    'secondary': '#3498db', 
    'accent': '#e74c3c',
    'success': '#27ae60',
    'warning': '#f39c12',
    'neutral': '#95a5a6'
}
PALETTE = sns.color_palette(['#2c3e50', '#3498db', '#27ae60', '#e74c3c', '#9b59b6', '#f39c12'])

# Try to import optional libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Sankey diagrams will be skipped.")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("Warning: WordCloud not installed. Keyword analysis will be skipped.")

# Configuration
CSV_FILES = [
    "/home/nrbarman/projects/chef-doeuvre/turpan/Qwen-2.5-7B-Instruct_gender_classification_merged.csv",
    "/home/nrbarman/projects/chef-doeuvre/turpan/Ministral-8B-Instruct-2410_gender_classification_merged.csv",
    "/home/nrbarman/projects/chef-doeuvre/turpan/Llama-3.1-8B-Instruct_gender_classification_merged.csv",
    "/home/nrbarman/projects/chef-doeuvre/turpan/Gemma-3-12B-Instruct_gender_classification_merged.csv",
    "/home/nrbarman/projects/chef-doeuvre/turpan/DeepSeek-R1-Distill-Qwen-7B_gender_classification_merged.csv",
    "/home/nrbarman/projects/chef-doeuvre/turpan/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit_gender_classification_merged.csv",
    "/home/nrbarman/projects/chef-doeuvre/turpan/DeepSeek-R1-Distill-Qwen-1.5B_gender_classification_merged.csv",
    "/home/nrbarman/projects/chef-doeuvre/turpan/Llama-3.2-1B-Instruct_gender_classification_merged.csv"
]

OUTPUT_DIR = "/home/nrbarman/projects/chef-doeuvre/turpan/plots"

def load_and_prep_data(filepath):
    """Loads CSV and renames continent columns to ethnicity."""
    try:
        df = pd.read_csv(filepath)
        # Rename columns
        rename_map = {
            'original_continent': 'original_ethnicity',
            'predicted_continent': 'predicted_ethnicity',
            'continent_keywords': 'ethnicity_keywords',
            'continent_reasoning': 'ethnicity_reasoning'
        }
        df.rename(columns=rename_map, inplace=True)
        # Fill NaNs and ensure strings
        cols_to_clean = ['original_gender', 'predicted_gender', 'original_ethnicity', 'predicted_ethnicity']
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip().str.lower()
        
        # Grouping Logic
        # Gender: keep male, female, else Other
        valid_genders = ['male', 'female']
        df['predicted_gender'] = df['predicted_gender'].apply(lambda x: x if x in valid_genders else 'other')
        df['original_gender'] = df['original_gender'].apply(lambda x: x if x in valid_genders else 'other')
        
        # Ethnicity: keep original ethnicities, else Other
        valid_ethnicities = set(df['original_ethnicity'].unique())
        df['predicted_ethnicity'] = df['predicted_ethnicity'].apply(lambda x: x if x in valid_ethnicities else 'other')

        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_model_name(filepath):
    """Extracts model name from filepath."""
    basename = os.path.basename(filepath)
    return basename.replace("_gender_classification_merged.csv", "")

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Generates and saves a confusion matrix with research paper styling."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create clean heatmap without annotations
    sns.heatmap(cm, annot=False, cmap='Blues', 
                xticklabels=[l.title() for l in labels], 
                yticklabels=[l.title() for l in labels],
                ax=ax, cbar_kws={'label': 'Count', 'shrink': 0.8},
                linewidths=0.5, linecolor='white')
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontweight='medium')
    ax.set_ylabel('True Label', fontweight='medium')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, facecolor='white', edgecolor='none')
    plt.close()

def plot_bar_distribution(df, column, title, filename):
    """Generates and saves a bar chart with research paper styling."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get value counts and sort
    counts = df[column].value_counts()
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(counts)), counts.values, color=PALETTE[0], 
                   edgecolor='white', linewidth=0.5, height=0.7)
    
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([str(l).title() for l in counts.index])
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, counts.values)):
        ax.text(val + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:,}', va='center', fontsize=9, color=COLORS['primary'])
    
    ax.set_xlabel('Count', fontweight='medium')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.invert_yaxis()
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, facecolor='white', edgecolor='none')
    plt.close()

def plot_sankey(df, source_col, target_col, title, filename):
    """Generates and saves a Sankey diagram using Plotly with research styling."""
    if not PLOTLY_AVAILABLE:
        return

    # Aggregate data
    flow = df.groupby([source_col, target_col]).size().reset_index(name='count')
    
    # Create labels with prefix to distinguish source/target
    sources = ['True: ' + str(s).title() for s in flow[source_col].unique().tolist()]
    targets = ['Pred: ' + str(t).title() for t in flow[target_col].unique().tolist()]
    all_labels = sources + targets
    
    source_map = {'True: ' + str(s).title(): i for i, s in enumerate(flow[source_col].unique())}
    target_map = {'Pred: ' + str(t).title(): i + len(sources) for i, t in enumerate(flow[target_col].unique())}
    
    source_indices = [source_map['True: ' + str(s).title()] for s in flow[source_col]]
    target_indices = [target_map['Pred: ' + str(t).title()] for t in flow[target_col]]
    values = flow['count'].tolist()
    
    # Color nodes
    node_colors = ['#2c3e50'] * len(sources) + ['#3498db'] * len(targets)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="white", width=1),
            label=all_labels,
            color=node_colors,
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color='rgba(52, 152, 219, 0.3)'
        ))])
    
    fig.update_layout(
        title_text=title, 
        font=dict(size=12, family='serif'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=1200,
        height=800
    )
    # Save as static image if possible, else HTML
    try:
        fig.write_image(filename, scale=3)
    except:
        fig.write_html(filename.replace('.png', '.html'))

def generate_wordcloud(text_series, title, filename, filter_words=None):
    """Generates and saves a word cloud with research styling."""
    if not WORDCLOUD_AVAILABLE:
        return
    
    text = " ".join(str(x) for x in text_series.dropna())
    if not text.strip():
        return
    
    # Filter out unwanted words
    if filter_words:
        for word in filter_words:
            text = text.replace(word, '')
            text = text.replace(word.lower(), '')
            text = text.replace(word.upper(), '')

    wc = WordCloud(
        width=1200, 
        height=600, 
        background_color='white',
        colormap='Blues',
        max_words=100,
        min_font_size=10,
        max_font_size=80,
        prefer_horizontal=0.7,
        random_state=42
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontweight='bold', pad=15, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, facecolor='white', edgecolor='none')
    plt.close()

def plot_intersectional_bias(df, gender_col, ethnicity_col, true_gender, true_ethnicity, title, filename):
    """Heatmap of accuracy for Gender-Ethnicity pairs with research styling."""
    df = df.copy()
    # Create a combined column for True and Predicted
    df['True_Intersection'] = df[true_gender].str.title() + " - " + df[true_ethnicity].str.title()
    df['Pred_Intersection'] = df[gender_col].str.title() + " - " + df[ethnicity_col].str.title()
    
    # Calculate accuracy for each group
    groups = df['True_Intersection'].unique()
    accuracies = {}
    counts = {}
    for group in groups:
        subset = df[df['True_Intersection'] == group]
        if len(subset) > 0:
            acc = accuracy_score(subset['True_Intersection'], subset['Pred_Intersection'])
            accuracies[group] = acc
            counts[group] = len(subset)
    
    if not accuracies:
        return

    # Convert to DataFrame for heatmap
    acc_df = pd.DataFrame(list(accuracies.items()), columns=['Group', 'Accuracy'])
    acc_df['Count'] = acc_df['Group'].map(counts)
    
    # Split group back to Gender/Ethnicity for matrix
    acc_df[['Gender', 'Ethnicity']] = acc_df['Group'].str.split(' - ', expand=True)
    pivot_table = acc_df.pivot(index='Gender', columns='Ethnicity', values='Accuracy')
    count_pivot = acc_df.pivot(index='Gender', columns='Ethnicity', values='Count')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create mask for NaN values
    mask = pivot_table.isna()
    
    sns.heatmap(pivot_table, annot=False, cmap='RdYlGn', 
                vmin=0, vmax=1, ax=ax, mask=mask,
                cbar_kws={'label': 'Accuracy', 'shrink': 0.8},
                linewidths=0.5, linecolor='white')
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Ethnicity', fontweight='medium')
    ax.set_ylabel('Gender', fontweight='medium')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, facecolor='white', edgecolor='none')
    plt.close()

def plot_per_class_metrics(y_true, y_pred, labels, title, filename):
    """Plot precision, recall, and F1 for each class with research styling."""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    # Create DataFrame for plotting
    metrics_df = pd.DataFrame({
        'Class': [l.title() for l in labels],
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color=PALETTE[0], edgecolor='white')
    bars2 = ax.bar(x, recall, width, label='Recall', color=PALETTE[1], edgecolor='white')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color=PALETTE[2], edgecolor='white')
    
    ax.set_ylabel('Score', fontweight='medium')
    ax.set_xlabel('Class', fontweight='medium')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([l.title() for l in labels], rotation=45, ha='right')
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, facecolor='white', edgecolor='none')
    plt.close()
    
    return metrics_df

def plot_comparison_side_by_side(df, original_col, predicted_col, title, filename):
    """Side-by-side comparison of original vs predicted distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original distribution
    orig_counts = df[original_col].value_counts()
    axes[0].barh(range(len(orig_counts)), orig_counts.values, color=PALETTE[0], 
                 edgecolor='white', linewidth=0.5, height=0.7)
    axes[0].set_yticks(range(len(orig_counts)))
    axes[0].set_yticklabels([str(l).title() for l in orig_counts.index])
    axes[0].set_xlabel('Count', fontweight='medium')
    axes[0].set_title('True Distribution', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Predicted distribution
    pred_counts = df[predicted_col].value_counts()
    axes[1].barh(range(len(pred_counts)), pred_counts.values, color=PALETTE[1], 
                 edgecolor='white', linewidth=0.5, height=0.7)
    axes[1].set_yticks(range(len(pred_counts)))
    axes[1].set_yticklabels([str(l).title() for l in pred_counts.index])
    axes[1].set_xlabel('Count', fontweight='medium')
    axes[1].set_title('Predicted Distribution', fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    fig.suptitle(title, fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, facecolor='white', edgecolor='none')
    plt.close()

def plot_model_agreement(model_preds, title, filename):
    """Heatmap of agreement between models."""
    models = list(model_preds.keys())
    n_models = len(models)
    agreement_matrix = np.zeros((n_models, n_models))
    
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                # Align indices if needed, assuming same order/index in CSVs
                # For simplicity, assuming all CSVs have same rows in same order
                # If not, we'd need to merge on index/song_title
                # Here we assume simple list comparison
                p1 = model_preds[m1]
                p2 = model_preds[m2]
                agreement = np.mean(np.array(p1) == np.array(p2))
                agreement_matrix[i, j] = agreement
                
    plt.figure(figsize=(10, 8))
    sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='Purples', xticklabels=models, yticklabels=models)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_accuracies = []
    # Store predictions as Series with index for alignment
    model_gender_preds = {}
    model_ethnicity_preds = {}

    for filepath in CSV_FILES:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        model_name = get_model_name(filepath)
        print(f"Processing {model_name}...")
        
        model_dir = os.path.join(OUTPUT_DIR, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        df = load_and_prep_data(filepath)
        if df is None:
            continue
        
        # Set index to 'index' column if available for alignment
        if 'index' in df.columns:
            df.set_index('index', inplace=True)

        # Store predictions
        model_gender_preds[model_name] = df['predicted_gender']
        model_ethnicity_preds[model_name] = df['predicted_ethnicity']

        # 1. Confusion Matrices
        # Gender
        gender_labels = sorted(df['original_gender'].unique().astype(str))
        plot_confusion_matrix(df['original_gender'], df['predicted_gender'], gender_labels, 
                              f'Gender Confusion Matrix - {model_name}', 
                              os.path.join(model_dir, 'gender_cm.png'))
        
        # Ethnicity
        ethnicity_labels = sorted(df['original_ethnicity'].unique().astype(str))
        plot_confusion_matrix(df['original_ethnicity'], df['predicted_ethnicity'], ethnicity_labels, 
                              f'Ethnicity Confusion Matrix - {model_name}', 
                              os.path.join(model_dir, 'ethnicity_cm.png'))

        # 2. Side-by-side comparison
        plot_comparison_side_by_side(df, 'original_gender', 'predicted_gender',
                                     f'Gender: True vs Predicted - {model_name}',
                                     os.path.join(model_dir, 'gender_comparison.png'))
        plot_comparison_side_by_side(df, 'original_ethnicity', 'predicted_ethnicity',
                                     f'Ethnicity: True vs Predicted - {model_name}',
                                     os.path.join(model_dir, 'ethnicity_comparison.png'))

        # 3. Accuracy
        gender_acc = accuracy_score(df['original_gender'], df['predicted_gender'])
        ethnicity_acc = accuracy_score(df['original_ethnicity'], df['predicted_ethnicity'])
        all_accuracies.append({
            'Model': model_name,
            'Gender Accuracy': gender_acc,
            'Ethnicity Accuracy': ethnicity_acc
        })
        
        # 3b. Per-class metrics
        plot_per_class_metrics(df['original_gender'], df['predicted_gender'], gender_labels,
                               f'Gender Classification Metrics - {model_name}',
                               os.path.join(model_dir, 'gender_metrics.png'))
        plot_per_class_metrics(df['original_ethnicity'], df['predicted_ethnicity'], ethnicity_labels,
                               f'Ethnicity Classification Metrics - {model_name}',
                               os.path.join(model_dir, 'ethnicity_metrics.png'))

        # 4. Sankey Diagrams
        plot_sankey(df, 'original_gender', 'predicted_gender', f'Gender Flow - {model_name}',
                    os.path.join(model_dir, 'gender_sankey.png'))
        plot_sankey(df, 'original_ethnicity', 'predicted_ethnicity', f'Ethnicity Flow - {model_name}',
                    os.path.join(model_dir, 'ethnicity_sankey.png'))

        # 5. Keyword Analysis
        if 'gender_keywords' in df.columns:
            generate_wordcloud(df['gender_keywords'], f'Gender Keywords - {model_name}',
                               os.path.join(model_dir, 'gender_keywords_wc.png'),
                               filter_words=['GENDER_REASONING', 'gender_reasoning'])
        if 'ethnicity_keywords' in df.columns:
            generate_wordcloud(df['ethnicity_keywords'], f'Ethnicity Keywords - {model_name}',
                               os.path.join(model_dir, 'ethnicity_keywords_wc.png'),
                               filter_words=['CONTINENT_REASONING', 'continent_reasoning', 'ETHNICITY_REASONING', 'ethnicity_reasoning'])

        # 6. Intersectional Bias
        plot_intersectional_bias(df, 'predicted_gender', 'predicted_ethnicity', 
                                 'original_gender', 'original_ethnicity',
                                 f'Intersectional Accuracy - {model_name}',
                                 os.path.join(model_dir, 'intersectional_bias.png'))

    # Aggregate Plots
    print("Generating aggregate plots...")
    acc_df = pd.DataFrame(all_accuracies)
    
    # Accuracy Comparison - Publication quality
    if not acc_df.empty:
        fig, ax = plt.subplots(figsize=(16, 7))
        
        # Use full model names
        model_labels = acc_df['Model'].tolist()
        
        # Create grouped bar chart
        n_models = len(acc_df)
        x = np.arange(n_models)
        width = 0.35
        
        gender_acc = acc_df['Gender Accuracy'].values
        ethnicity_acc = acc_df['Ethnicity Accuracy'].values
        
        bars1 = ax.bar(x - width/2, gender_acc, width, label='Gender', color=PALETTE[0], edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + width/2, ethnicity_acc, width, label='Ethnicity', color=PALETTE[1], edgecolor='white', linewidth=0.5)
        
        ax.set_ylabel('Accuracy', fontweight='medium')
        ax.set_xlabel('Model', fontweight='medium')
        ax.set_title('Model Accuracy Comparison: Gender vs Ethnicity Classification', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.legend(frameon=False, loc='upper right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'), facecolor='white', edgecolor='none')
        plt.close()
        
        # Save accuracy table as CSV
        acc_df.to_csv(os.path.join(OUTPUT_DIR, 'accuracy_summary.csv'), index=False)
        print(f"\nAccuracy Summary:\n{acc_df.to_string(index=False)}")

    # Model Agreement
    def align_and_calculate_agreement(preds_dict, title, filename):
        models = list(preds_dict.keys())
        n_models = len(models)
        agreement_matrix = np.zeros((n_models, n_models))
        
        # Find common index
        common_index = None
        for m in models:
            if common_index is None:
                common_index = preds_dict[m].index
            else:
                common_index = common_index.intersection(preds_dict[m].index)
        
        if common_index is None or len(common_index) == 0:
            print(f"No common indices found for {title}")
            return

        print(f"Calculating agreement on {len(common_index)} common samples for {title}")

        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    p1 = preds_dict[m1].loc[common_index]
                    p2 = preds_dict[m2].loc[common_index]
                    agreement = np.mean(p1 == p2)
                    agreement_matrix[i, j] = agreement
        
        # Use full model names
        model_labels = models
                    
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(agreement_matrix, dtype=bool), k=1)
        
        sns.heatmap(agreement_matrix, annot=False, cmap='Purples', 
                    xticklabels=model_labels, yticklabels=model_labels, ax=ax,
                    vmin=0.5, vmax=1.0, cbar_kws={'label': 'Agreement', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white')
        
        ax.set_title(title, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filename, facecolor='white', edgecolor='none')
        plt.close()

    if model_gender_preds:
        align_and_calculate_agreement(model_gender_preds, 'Gender Prediction Agreement', 
                             os.path.join(OUTPUT_DIR, 'gender_agreement.png'))
    if model_ethnicity_preds:
        align_and_calculate_agreement(model_ethnicity_preds, 'Ethnicity Prediction Agreement', 
                             os.path.join(OUTPUT_DIR, 'ethnicity_agreement.png'))

    print("Done!")

if __name__ == "__main__":
    main()
