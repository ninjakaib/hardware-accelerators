#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys

def extract_design_features(design_name):
    """Extract design type, format, and other features from the design name."""
    features = {}
    
    # Extract design type (adder, multiplier, lmul)
    if "adder" in design_name:
        features["type"] = "adder"
    elif "multiplier" in design_name:
        features["type"] = "multiplier"
    elif "lmul" in design_name:
        features["type"] = "lmul"
    else:
        features["type"] = "other"
    
    # Extract format (fp32, fp8, bf16)
    if "fp32" in design_name:
        features["format"] = "fp32"
    elif "fp8" in design_name:
        features["format"] = "fp8"
    elif "bf16" in design_name:
        features["format"] = "bf16"
    else:
        features["format"] = "other"
    
    # Extract implementation style
    if "combinational" in design_name:
        features["style"] = "combinational"
    elif "pipelined" in design_name:
        features["style"] = "pipelined"
    elif "stage" in design_name:
        # Extract stage number
        stage_match = re.search(r'stage_(\d+)', design_name)
        if stage_match:
            features["style"] = f"stage_{stage_match.group(1)}"
        else:
            features["style"] = "staged"
    else:
        features["style"] = "other"
    
    # Extract optimization level
    if "fast" in design_name:
        features["optimization"] = "fast"
    elif "simple" in design_name:
        features["optimization"] = "simple"
    else:
        features["optimization"] = "standard"
    
    return features

def main():
    # Script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Reports directory
    reports_dir = os.path.join(script_dir, "reports")
    
    # CSV file path
    csv_path = os.path.join(reports_dir, "design_metrics.csv")
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' does not exist.")
        print("Please run the analyze_reports.py script first to generate the CSV file.")
        return 1
    
    # Load the DataFrame
    df = pd.read_csv(csv_path)
    print(f"Loaded data for {len(df)} designs.")
    
    # Extract design features
    feature_data = []
    for design in df['design']:
        features = extract_design_features(design)
        features['design'] = design
        feature_data.append(features)
    
    # Create features DataFrame and merge with metrics
    features_df = pd.DataFrame(feature_data)
    df = pd.merge(df, features_df, on='design')
    
    # Create output directory for plots
    plots_dir = os.path.join(reports_dir, "comparison_plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Set the style for plots
    sns.set(style="whitegrid")
    
    # 1. Compare area by design type and format
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='type', y='area', hue='format', data=df)
    plt.title('Area by Design Type and Format')
    plt.xlabel('Design Type')
    plt.ylabel('Area')
    plt.savefig(os.path.join(plots_dir, 'area_by_type_format.png'))
    print("Created plot: area_by_type_format.png")
    
    # 2. Compare area by implementation style
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='style', y='area', hue='type', data=df)
    plt.title('Area by Implementation Style and Design Type')
    plt.xlabel('Implementation Style')
    plt.ylabel('Area')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'area_by_style.png'))
    print("Created plot: area_by_style.png")
    
    # 3. Compare power consumption by design type and format
    if 'total_power' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='type', y='total_power', hue='format', data=df)
        plt.title('Power Consumption by Design Type and Format')
        plt.xlabel('Design Type')
        plt.ylabel('Total Power')
        plt.savefig(os.path.join(plots_dir, 'power_by_type_format.png'))
        print("Created plot: power_by_type_format.png")
    
    # 4. Compare timing (WNS) by design type and style
    if 'wns' in df.columns:
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='style', y='wns', hue='type', data=df)
        plt.title('Worst Negative Slack by Implementation Style and Design Type')
        plt.xlabel('Implementation Style')
        plt.ylabel('WNS')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'wns_by_style.png'))
        print("Created plot: wns_by_style.png")
    
    # 5. Compare optimization impact
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='type', y='area', hue='optimization', data=df)
    plt.title('Area by Design Type and Optimization Level')
    plt.xlabel('Design Type')
    plt.ylabel('Area')
    plt.savefig(os.path.join(plots_dir, 'area_by_optimization.png'))
    print("Created plot: area_by_optimization.png")
    
    # 6. Create a summary table by design type and format
    summary = df.groupby(['type', 'format']).agg({
        'area': ['mean', 'min', 'max', 'count'],
        'cell_count': ['mean', 'min', 'max'],
        'runtime_total_seconds': ['mean', 'min', 'max']
    }).reset_index()
    
    # Save summary to CSV
    summary_path = os.path.join(reports_dir, "design_summary_by_type_format.csv")
    summary.to_csv(summary_path)
    print(f"\nSaved summary statistics to: {summary_path}")
    
    # 7. Create a summary table by implementation style
    style_summary = df.groupby(['type', 'style']).agg({
        'area': ['mean', 'min', 'max', 'count'],
        'cell_count': ['mean', 'min', 'max'],
        'runtime_total_seconds': ['mean', 'min', 'max']
    }).reset_index()
    
    # Save style summary to CSV
    style_summary_path = os.path.join(reports_dir, "design_summary_by_style.csv")
    style_summary.to_csv(style_summary_path)
    print(f"Saved style summary statistics to: {style_summary_path}")
    
    # 8. Create a correlation heatmap
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
        print("Created plot: correlation_heatmap.png")
    
    print("\nComparison analysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 