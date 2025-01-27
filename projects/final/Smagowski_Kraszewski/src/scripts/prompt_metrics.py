import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

def analyze_classification_results(df):
    """
    Analyze binary classification results and generate comprehensive metrics and visualizations.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with columns true_label, predicted_label, model, prompt_method
    """
    # Create empty dictionaries to store results
    results = {
        'model': [],
        'prompt_method': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    # Calculate metrics for each combination of model and prompt method
    for model in df['model'].unique():
        for method in df['prompt_method'].unique():
            mask = (df['model'] == model) & (df['prompt_method'] == method)
            subset = df[mask]
            
            if len(subset) > 0:
                y_true = subset['true_label']
                y_pred = subset['predicted_label']
                y_pred_proba = y_pred.astype(float)  # Assuming predicted_label contains probabilities
                
                # Calculate metrics
                results['model'].append(model)
                results['prompt_method'].append(method)
                results['accuracy'].append(accuracy_score(y_true, y_pred.round()))
                results['precision'].append(precision_score(y_true, y_pred.round()))
                results['recall'].append(recall_score(y_true, y_pred.round()))
                results['f1'].append(f1_score(y_true, y_pred.round()))
                
                # Calculate ROC and AUC
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                results['auc'].append(auc(fpr, tpr))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create visualizations
    create_metric_plots(results_df)
    create_roc_curves(df)
    
    return results_df

def create_metric_plots(results_df):
    """Create bar plots for F1 and Accuracy scores using matplotlib"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get unique models and prompt methods
    models = results_df['model'].unique()
    methods = results_df['prompt_method'].unique()
    
    # Set bar width and positions
    bar_width = 0.35
    x = np.arange(len(methods))
    
    # Plot F1 scores
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        ax1.bar(x + i*bar_width, model_data['f1'], bar_width, 
                label=model, alpha=0.8)
    
    ax1.set_title('F1 Scores by Model and Prompt Method', pad=20)
    ax1.set_xlabel('Prompt Method')
    ax1.set_ylabel('F1 Score')
    ax1.set_xticks(x + bar_width/2)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Accuracy scores
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        ax2.bar(x + i*bar_width, model_data['accuracy'], bar_width,
                label=model, alpha=0.8)
    
    ax2.set_title('Accuracy Scores by Model and Prompt Method', pad=20)
    ax2.set_xlabel('Prompt Method')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks(x + bar_width/2)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def create_roc_curves(df):
    """Create ROC curves for all model and prompt method combinations"""
    plt.figure(figsize=(10, 8))
    
    # Define a color cycle for different combinations
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['model'].unique()) * len(df['prompt_method'].unique())))
    color_idx = 0
    
    # Plot ROC curve for each combination
    for model in df['model'].unique():
        for method in df['prompt_method'].unique():
            mask = (df['model'] == model) & (df['prompt_method'] == method)
            subset = df[mask]
            
            if len(subset) > 0:
                y_true = subset['true_label']
                y_pred_proba = subset['predicted_label'].astype(float)
                
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color=colors[color_idx],
                        label=f'{model} - {method} (AUC = {roc_auc:.2f})',
                        linewidth=2)
                color_idx += 1
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    
    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models and Prompt Methods')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def main():
    # Read the CSV file
    df = pd.read_csv('prompt_results/all_results_20250122_031114.csv')
    
    # Run analysis
    results_df = analyze_classification_results(df)
    
    # Display metrics summary
    print("\nMetrics Summary:")
    print(results_df.to_string(index=False))
    
    # Optional: Save results to CSV
    results_df.to_csv('classification_results.csv', index=False)

if __name__ == "__main__":
    main()