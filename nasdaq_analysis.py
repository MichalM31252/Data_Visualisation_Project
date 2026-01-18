"""
NASDAQ Stock Market Analysis Script
Analyzes top 10 NASDAQ companies with:
1. Historical data retrieval for 2021
2. Daily trend visualization
3. K-means clustering by month
4. Bubble charts showing clustering results per month
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
TOP_10_NASDAQ_COMPANIES = [
    'AAPL',    # Apple
    'MSFT',    # Microsoft
    'AMZN',    # Amazon
    'NVDA',    # NVIDIA
    'GOOGL',   # Google/Alphabet
    'META',    # Meta (Facebook)
    'TSLA',    # Tesla
    'NFLX',    # Netflix
    'AVGO',    # Broadcom
    'ASML'     # ASML
]

START_DATE = '2021-01-01'
END_DATE = '2021-12-31'
OPTIMAL_K = 3  # Number of clusters

# ==================== SECTION A: FETCH DATA ====================
def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical daily stock data for given tickers.
    
    Parameters:
    -----------
    tickers : list
        List of stock symbols
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    
    Returns:
    --------
    dict : Dictionary with ticker as key and DataFrame as value
    """
    print("=" * 70)
    print("SECTION A: FETCHING HISTORICAL STOCK DATA")
    print("=" * 70)
    
    stock_data = {}
    
    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...", end=" ")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            stock_data[ticker] = data
            print(f"✓ {len(data)} records")
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    print(f"\nSuccessfully fetched data for {len(stock_data)} companies")
    return stock_data

# ==================== SECTION B: CREATE LINE CHARTS ====================
def create_trend_charts(stock_data):
    """
    Create individual line charts for each company showing adjusted close price trends.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary with ticker as key and DataFrame as value
    """
    print("\n" + "=" * 70)
    print("SECTION B: CREATING TREND CHARTS")
    print("=" * 70)
    
    # Create individual plots for each company
    fig, axes = plt.subplots(5, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (ticker, data) in enumerate(stock_data.items()):
        ax = axes[idx]
        ax.plot(data.index, data['Adj Close'], linewidth=2, color='steelblue')
        ax.set_title(f'{ticker} - Stock Price Trend (2021)', fontweight='bold', fontsize=11)
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Adjusted Close Price ($)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add statistics
        min_price = data['Adj Close'].min()
        max_price = data['Adj Close'].max()
        current_price = data['Adj Close'].iloc[-1]
        change_pct = ((current_price - data['Adj Close'].iloc[0]) / data['Adj Close'].iloc[0]) * 100
        
        stats_text = f'Min: ${min_price:.2f}\nMax: ${max_price:.2f}\nChange: {change_pct:+.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('01_trend_charts_all_companies.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_trend_charts_all_companies.png")
    plt.close()
    
    # Create a combined comparison chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for ticker, data in stock_data.items():
        # Normalize prices to 100 for comparison
        normalized = (data['Adj Close'] / data['Adj Close'].iloc[0]) * 100
        ax.plot(normalized.index, normalized, marker='o', markersize=4, 
                label=ticker, linewidth=2, alpha=0.7)
    
    ax.set_title('Normalized Stock Price Comparison (2021)\n(Base: 100 at 2021-01-01)', 
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Normalized Price (Base = 100)', fontsize=11)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('02_normalized_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_normalized_comparison.png")
    plt.close()

# ==================== SECTION C: MONTHLY K-MEANS CLUSTERING ====================
def get_monthly_last_prices(stock_data):
    """
    Extract the last trading day price for each company per month.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary with ticker as key and DataFrame as value
    
    Returns:
    --------
    dict : Dictionary with month as key and DataFrame (companies x prices) as value
    """
    print("\n" + "=" * 70)
    print("SECTION C: PREPARING DATA FOR K-MEANS CLUSTERING")
    print("=" * 70)
    
    monthly_prices = {}
    
    for ticker, data in stock_data.items():
        # Resample to get last business day of each month
        monthly_data = data['Adj Close'].resample('M').last()
        
        for date, price in monthly_data.items():
            month_str = date.strftime('%Y-%m')
            if month_str not in monthly_prices:
                monthly_prices[month_str] = {}
            monthly_prices[month_str][ticker] = price
    
    # Convert to DataFrames
    monthly_dfs = {}
    for month, prices in sorted(monthly_prices.items()):
        monthly_dfs[month] = pd.DataFrame(list(prices.items()), 
                                         columns=['Ticker', 'Price'])
    
    print(f"✓ Extracted prices for {len(monthly_dfs)} months")
    for month in sorted(monthly_dfs.keys()):
        print(f"  {month}: {len(monthly_dfs[month])} companies")
    
    return monthly_dfs

def perform_kmeans_clustering(monthly_prices_dfs, n_clusters=OPTIMAL_K):
    """
    Perform K-means clustering on monthly price data.
    
    Parameters:
    -----------
    monthly_prices_dfs : dict
        Dictionary with month as key and price DataFrame as value
    n_clusters : int
        Number of clusters
    
    Returns:
    --------
    dict : Dictionary with clustering results per month
    """
    print(f"\n✓ Performing K-means clustering with K={n_clusters}")
    
    clustering_results = {}
    
    for month, df in sorted(monthly_prices_dfs.items()):
        # Prepare features for clustering
        prices = df['Price'].values.reshape(-1, 1)
        
        # Standardize prices
        scaler = StandardScaler()
        prices_scaled = scaler.fit_transform(prices)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(prices_scaled)
        
        # Calculate silhouette score
        if len(df) > n_clusters:
            silhouette_avg = silhouette_score(prices_scaled, clusters)
        else:
            silhouette_avg = np.nan
        
        # Store results
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = clusters
        df_with_clusters['Price_Scaled'] = prices_scaled.flatten()
        df_with_clusters['Centroid'] = kmeans.cluster_centers_[clusters].flatten()
        
        clustering_results[month] = {
            'df': df_with_clusters,
            'scaler': scaler,
            'kmeans': kmeans,
            'silhouette': silhouette_avg,
            'original_prices': prices.flatten()
        }
        
        print(f"  {month}: Silhouette Score = {silhouette_avg:.3f}")
    
    return clustering_results

# ==================== SECTION D: BUBBLE CHARTS ====================
def create_bubble_charts(clustering_results):
    """
    Create bubble charts for each month showing clustering results.
    Each bubble represents a cluster, with size proportional to number of companies.
    
    Parameters:
    -----------
    clustering_results : dict
        Dictionary with clustering results per month
    """
    print("\n" + "=" * 70)
    print("SECTION D: CREATING BUBBLE CHARTS FOR CLUSTERING")
    print("=" * 70)
    
    # Define colors for clusters
    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    fig_count = 0
    rows_per_fig = 3
    cols_per_fig = 4
    plots_per_fig = rows_per_fig * cols_per_fig
    
    months = sorted(clustering_results.keys())
    total_months = len(months)
    num_figs = (total_months + plots_per_fig - 1) // plots_per_fig
    
    for fig_num in range(num_figs):
        fig, axes = plt.subplots(rows_per_fig, cols_per_fig, 
                                 figsize=(16, 12))
        axes = axes.flatten()
        
        start_idx = fig_num * plots_per_fig
        end_idx = min(start_idx + plots_per_fig, total_months)
        
        for plot_idx, month_idx in enumerate(range(start_idx, end_idx)):
            month = months[month_idx]
            ax = axes[plot_idx]
            
            results = clustering_results[month]
            df = results['df']
            
            # Group by cluster
            for cluster in sorted(df['Cluster'].unique()):
                cluster_data = df[df['Cluster'] == cluster]
                companies = cluster_data['Ticker'].values
                prices = cluster_data['Price'].values
                
                # Calculate bubble properties
                bubble_size = len(companies) * 300  # Size based on number of companies
                
                # X-axis: average price of cluster
                x = prices.mean()
                
                # Y-axis: cluster ID (for visual separation)
                y = cluster
                
                # Plot bubble
                ax.scatter(x, y, s=bubble_size, alpha=0.6, 
                          color=cluster_colors[cluster % len(cluster_colors)],
                          edgecolors='black', linewidth=2)
                
                # Add text label with company names and count
                label_text = f"C{cluster}\n({len(companies)})"
                ax.text(x, y, label_text, ha='center', va='center',
                       fontweight='bold', fontsize=9)
            
            ax.set_title(f'{month} - K-Means Clustering\n(Silhouette: {results["silhouette"]:.3f})',
                        fontweight='bold', fontsize=11)
            ax.set_xlabel('Average Price ($)', fontsize=10)
            ax.set_ylabel('Cluster', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.5, max(df['Cluster']) + 0.5)
        
        # Hide unused subplots
        for plot_idx in range(end_idx - start_idx, plots_per_fig):
            axes[plot_idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'03_bubble_charts_part_{fig_num+1}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 03_bubble_charts_part_{fig_num+1}.png")
        plt.close()
    
    # Create detailed bubble charts with actual company positions
    print("\n✓ Creating detailed scatter plots by cluster...")
    
    for month_idx, month in enumerate(sorted(clustering_results.keys())):
        if month_idx % 3 == 0:  # 3 months per figure for detail
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig_months = []
        
        results = clustering_results[month]
        df = results['df']
        ax = axes[month_idx % 3]
        fig_months.append(month)
        
        # Create scatter plot
        colors_list = [cluster_colors[c % len(cluster_colors)] 
                      for c in df['Cluster']]
        
        scatter = ax.scatter(df['Price'], df['Cluster'], 
                           s=300, c=df['Cluster'], cmap='viridis',
                           alpha=0.6, edgecolors='black', linewidth=1.5)
        
        # Add company labels
        for idx, row in df.iterrows():
            ax.annotate(row['Ticker'], 
                       (row['Price'], row['Cluster']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')
        
        ax.set_title(f'{month} - Company Distribution\nby Cluster and Price',
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('Stock Price ($)', fontsize=10)
        ax.set_ylabel('Cluster', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_yticks(sorted(df['Cluster'].unique()))
        
        if (month_idx + 1) % 3 == 0 or month_idx == len(clustering_results) - 1:
            plt.tight_layout()
            plt.savefig(f'04_detailed_clusters_{fig_months[0]}_{fig_months[-1]}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Saved: 04_detailed_clusters_{fig_months[0]}_{fig_months[-1]}.png")
            plt.close()

# ==================== CLUSTERING SUMMARY ====================
def print_clustering_summary(clustering_results):
    """
    Print detailed summary of clustering results.
    
    Parameters:
    -----------
    clustering_results : dict
        Dictionary with clustering results per month
    """
    print("\n" + "=" * 70)
    print("CLUSTERING SUMMARY BY MONTH")
    print("=" * 70)
    
    for month in sorted(clustering_results.keys()):
        results = clustering_results[month]
        df = results['df']
        
        print(f"\n{month}:")
        print(f"  Silhouette Score: {results['silhouette']:.4f}")
        
        for cluster in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster]
            companies = ', '.join(cluster_data['Ticker'].values)
            min_price = cluster_data['Price'].min()
            max_price = cluster_data['Price'].max()
            avg_price = cluster_data['Price'].mean()
            
            print(f"  Cluster {cluster}:")
            print(f"    Companies: {companies}")
            print(f"    Price Range: ${min_price:.2f} - ${max_price:.2f}")
            print(f"    Average Price: ${avg_price:.2f}")

# ==================== MAIN EXECUTION ====================
def main():
    """
    Main execution function orchestrating the entire analysis.
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "NASDAQ STOCK MARKET ANALYSIS - 2021".center(68) + "║")
    print("║" + "K-Means Clustering & Visualization".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        # Step A: Fetch data
        stock_data = fetch_stock_data(TOP_10_NASDAQ_COMPANIES, START_DATE, END_DATE)
        
        if not stock_data:
            print("Error: Could not fetch any stock data. Exiting.")
            return
        
        # Step B: Create trend charts
        create_trend_charts(stock_data)
        
        # Step C: Prepare for clustering and perform K-means
        monthly_prices_dfs = get_monthly_last_prices(stock_data)
        clustering_results = perform_kmeans_clustering(monthly_prices_dfs, OPTIMAL_K)
        
        # Step D: Create bubble charts
        create_bubble_charts(clustering_results)
        
        # Print summary
        print_clustering_summary(clustering_results)
        
        # Final summary
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  1. 01_trend_charts_all_companies.png - Individual company trends")
        print("  2. 02_normalized_comparison.png - Normalized price comparison")
        print("  3. 03_bubble_charts_part_*.png - Monthly clustering bubbles")
        print("  4. 04_detailed_clusters_*.png - Detailed scatter plots by cluster")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
