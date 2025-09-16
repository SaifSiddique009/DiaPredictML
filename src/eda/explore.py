import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.utils.helpers import logger

def perform_eda(file_path, output_dir='results/eda_plots/'):
    """
    Perform EDA: stats, correlations, distributions, pairplots.
    Insights: Helps decide models (e.g., tree-based for non-linear relations, handle imbalance).
    
    Args:
        file_path (str): Path to CSV.
        output_dir (str): Where to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(file_path)
    
    # Basic Stats
    logger.info("Data Description:\n" + str(df.describe()))
    logger.info("Missing Values:\n" + str(df.isnull().sum()))  # No missings in this dataset
    
    # Class Balance
    sns.countplot(x='Outcome', data=df)
    plt.title('Class Distribution')
    plt.savefig(os.path.join(output_dir, 'class_dist.png'))
    plt.close()
    logger.info(f"Class imbalance: {df['Outcome'].value_counts(normalize=True)}")  # ~65% negative, 35% positive -> Need stratified split, class weights
    
    # Correlations
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'corr_matrix.png'))
    plt.close()
    logger.info("High correlations: Glucose, BMI with Outcome -> Good features for prediction")
    
    # Distributions
    df.hist(bins=20, figsize=(12, 10))
    plt.suptitle('Feature Distributions')
    plt.savefig(os.path.join(output_dir, 'distributions.png'))
    plt.close()
    logger.info("Distributions: Some skewed (e.g., Insulin) -> Scaling helps")
    
    # Pairplot (subset for speed)
    sns.pairplot(df[['Glucose', 'BMI', 'Age', 'Outcome']], hue='Outcome')
    plt.savefig(os.path.join(output_dir, 'pairplot.png'))
    plt.close()
    logger.info("Pairplot Insights: Non-linear separations -> Tree-based/ensemble models suitable; ANN for complex patterns")
    
    # EDA Conclusions for Model Selection:
    # - Binary classification, imbalanced -> Will use F1 as key metric, class weights.
    # - Numerical features, some correlations -> Baselines: LogisticReg (linear), DecisionTree (non-linear).
    # - Potential non-linearity -> Advanced: RF, XGBoost.
    # - Small dataset (768 samples) -> ANN feasible but prone to overfit; might need to use dropout/tuning.