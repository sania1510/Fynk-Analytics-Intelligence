"""
Test: Upload CSV and Generate Real Data Insights
Shows actual trends: Revenue growth, Profit changes, Units sold trends
"""

import asyncio
import pandas as pd
from pathlib import Path
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from src.data.loader import DataLoader
from src.analytics.analyzer import AnalyticsAnalyzer
from src.analytics.insights import InsightsGenerator


def calculate_insights(df, schema):
    """Extract and calculate real data insights"""
    
    insights = []
    
    # Calculate revenue trend
    if 'revenue' in df.columns and 'date' in df.columns:
        df_sorted = df.sort_values('date')
        first_revenue = df_sorted['revenue'].iloc[0]
        last_revenue = df_sorted['revenue'].iloc[-1]
        revenue_change = last_revenue - first_revenue
        revenue_pct = (revenue_change / first_revenue * 100) if first_revenue > 0 else 0
        
        if revenue_change > 0:
            insights.append(f"Revenue INCREASED by ${revenue_change:,.0f} ({revenue_pct:+.1f}%) - from ${first_revenue:,.0f} to ${last_revenue:,.0f}")
        else:
            insights.append(f"Revenue DECREASED by ${abs(revenue_change):,.0f} ({revenue_pct:.1f}%) - from ${first_revenue:,.0f} to ${last_revenue:,.0f}")
    
    # Calculate profit trend
    if 'profit' in df.columns and 'date' in df.columns:
        df_sorted = df.sort_values('date')
        first_profit = df_sorted['profit'].iloc[0]
        last_profit = df_sorted['profit'].iloc[-1]
        profit_change = last_profit - first_profit
        profit_pct = (profit_change / first_profit * 100) if first_profit > 0 else 0
        total_profit = df['profit'].sum()
        
        if profit_change > 0:
            insights.append(f"Profit INCREASED by ${profit_change:,.0f} ({profit_pct:+.1f}%) - Total profit: ${total_profit:,.0f}")
        else:
            insights.append(f"Profit DECREASED by ${abs(profit_change):,.0f} ({abs(profit_pct):.1f}%) - Total profit: ${total_profit:,.0f}")
    
    # Calculate units sold trend
    if 'units_sold' in df.columns and 'date' in df.columns:
        df_sorted = df.sort_values('date')
        first_units = df_sorted['units_sold'].iloc[0]
        last_units = df_sorted['units_sold'].iloc[-1]
        units_change = last_units - first_units
        units_pct = (units_change / first_units * 100) if first_units > 0 else 0
        total_units = df['units_sold'].sum()
        
        if units_change > 0:
            insights.append(f"Units Sold INCREASED by {units_change} units ({units_pct:+.1f}%) - Total units: {total_units}")
        else:
            insights.append(f"Units Sold DECREASED by {abs(units_change)} units ({units_pct:.1f}%) - Total units: {total_units}")
    
    # Regional performance
    if 'revenue' in df.columns and 'region' in df.columns:
        region_revenue = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
        top_region = region_revenue.idxmax()
        top_revenue = region_revenue.max()
        insights.append(f"Top performing region: {top_region} with ${top_revenue:,.0f} in revenue")
    
    # Product performance
    if 'revenue' in df.columns and 'product' in df.columns:
        product_revenue = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
        top_product = product_revenue.idxmax()
        top_prod_revenue = product_revenue.max()
        insights.append(f"Best selling product: {top_product} with ${top_prod_revenue:,.0f} in revenue")
    
    # Profit margin trend
    if 'profit' in df.columns and 'revenue' in df.columns:
        df['margin'] = (df['profit'] / df['revenue'] * 100)
        avg_margin = df['margin'].mean()
        insights.append(f"Average profit margin: {avg_margin:.1f}%")
    
    # Average conversion rate
    if 'conversion_rate' in df.columns:
        avg_conversion = df['conversion_rate'].mean()
        insights.append(f"Average conversion rate: {avg_conversion:.1%}")
    
    return insights


async def test_csv_insights():
    """Load CSV file, analyze it, and generate real insights"""
    
    print("="*80)
    print("CSV UPLOAD & DATA INSIGHTS TEST")
    print("="*80)
    
    # Step 1: Load CSV File
    print("\n[STEP 1] LOADING CSV FILE")
    print("-"*80)
    
    csv_path = Path("sample_data/sales_data.csv")
    
    if not csv_path.exists():
        print(f"ERROR: CSV file not found at {csv_path}")
        return
    
    print(f"SUCCESS: CSV file found: {csv_path}")
    
    # Load the CSV
    loader = DataLoader()
    result = await loader.load_csv(
        str(csv_path), 
        dataset_id="sales_test",
        auto_detect=True
    )
    
    if not result.get('success'):
        print(f"ERROR: {result.get('error')}")
        return
    
    print(f"SUCCESS: CSV loaded")
    print(f"   Rows: {result['rows']}, Columns: {result['columns']}")
    
    # Get the dataframe and schema from loader
    df = loader.data_store.get(result['dataset_id'])
    schema = loader.schema_store.get(result['dataset_id'])
    
    if df is not None:
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Schema metrics: {[m.name for m in schema.metrics]}")
    
    # Step 2: Calculate Real Data Insights
    print("\n[STEP 2] DATA TREND ANALYSIS")
    print("-"*80)
    
    real_insights = calculate_insights(df, schema)
    
    print("\nKEY INSIGHTS:")
    for i, insight in enumerate(real_insights, 1):
        print(f"  {i}. {insight}")
    
    # Step 3: Run Analytics
    print("\n[STEP 3] RUNNING DETAILED ANALYTICS")
    print("-"*80)
    
    analyzer = AnalyticsAnalyzer(enable_ml=False)
    
    # 3a. Time Series Analysis
    print("\n[TIME SERIES: Revenue Trend]")
    try:
        ts_result = await analyzer.time_series_analysis(
            df=df,
            schema=schema,
            metric_names=['revenue'],
            granularity='day'
        )
        print(f"SUCCESS: {len(ts_result.get('data', []))} days analyzed")
        
        # Show summary statistics
        summary = ts_result.get('summary', {}).get('revenue', {})
        print(f"  Total Revenue: ${summary.get('total', 0):,.0f}")
        print(f"  Daily Average: ${summary.get('avg', 0):,.0f}")
        print(f"  Min: ${summary.get('min', 0):,.0f}, Max: ${summary.get('max', 0):,.0f}")
        print(f"  Trend: {summary.get('direction', 'Unknown')}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        ts_result = None
    
    # 3b. Profit Analysis
    print("\n[PROFIT ANALYSIS]")
    try:
        profit_result = await analyzer.time_series_analysis(
            df=df,
            schema=schema,
            metric_names=['profit'],
            granularity='day'
        )
        print(f"SUCCESS: {len(profit_result.get('data', []))} days analyzed")
        
        summary = profit_result.get('summary', {}).get('profit', {})
        print(f"  Total Profit: ${summary.get('total', 0):,.0f}")
        print(f"  Daily Average: ${summary.get('avg', 0):,.0f}")
        print(f"  Trend: {summary.get('direction', 'Unknown')}")
        
    except Exception as e:
        print(f"ERROR: {e}")
    
    # 3c. Regional Performance
    print("\n[REGIONAL BREAKDOWN]")
    try:
        regional = await analyzer.dimension_breakdown(
            df=df,
            schema=schema,
            metric_names=['revenue', 'profit'],
            dimension_name='region'
        )
        print(f"SUCCESS: {len(regional.get('data', []))} regions analyzed")
        
        for region_data in regional.get('data', [])[:4]:
            region = region_data.get('region', 'Unknown')
            revenue = region_data.get('revenue', 0)
            profit = region_data.get('profit', 0)
            print(f"  {region}: Revenue=${revenue:,.0f}, Profit=${profit:,.0f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
    
    # 3d. Product Performance
    print("\n[PRODUCT BREAKDOWN]")
    try:
        products = await analyzer.dimension_breakdown(
            df=df,
            schema=schema,
            metric_names=['revenue', 'units_sold'],
            dimension_name='product'
        )
        print(f"SUCCESS: {len(products.get('data', []))} products analyzed")
        
        for prod_data in products.get('data', []):
            product = prod_data.get('product', 'Unknown')
            revenue = prod_data.get('revenue', 0)
            units = prod_data.get('units_sold', 0)
            print(f"  {product}: Revenue=${revenue:,.0f}, Units={units}")
        
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Step 4: Generate AI Insights (if available)
    print("\n[STEP 4] GENERATING AI INSIGHTS")
    print("-"*80)
    
    insights_gen = InsightsGenerator()
    
    if ts_result:
        print("\n[TIME SERIES INSIGHTS]")
        try:
            ai_insights = await insights_gen.generate_insights(
                ts_result,
                insight_type='detailed'
            )
            
            if ai_insights.get('insight_type') == 'fallback':
                print("  (Using fallback - AI not available)")
            
            for i, text in enumerate(ai_insights.get('insights', [])[:2], 1):
                preview = text[:75] + "..." if len(text) > 75 else text
                print(f"  {i}. {preview}")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nSUMMARY:")
    print(f"  Total rows analyzed: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Total revenue: ${df['revenue'].sum():,.0f}")
    print(f"  Total profit: ${df['profit'].sum():,.0f}")
    print(f"  Total units sold: {df['units_sold'].sum()}")
    print(f"  Overall profit margin: {(df['profit'].sum() / df['revenue'].sum() * 100):.1f}%")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_csv_insights())
