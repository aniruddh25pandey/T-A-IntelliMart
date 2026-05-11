"""
T&A IntelliMart - Flask Backend API (Restored & Stable)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from datetime import datetime
from textblob import TextBlob
import random
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

MODELS_DIR = 'models'
DATASET_PATH = '../DATASET/dataset_with_sentiment.csv'
DB_PATH = 'reviews.db'

# Load ML models
try:
    discount_model = joblib.load(f'{MODELS_DIR}/discount_model.pkl')
    price_model = joblib.load(f'{MODELS_DIR}/price_model.pkl')
    le_category = joblib.load(f'{MODELS_DIR}/le_category.pkl')
    le_vendor = joblib.load(f'{MODELS_DIR}/le_vendor.pkl')
    le_sentiment = joblib.load(f'{MODELS_DIR}/le_sentiment.pkl')
    le_price_cat = joblib.load(f'{MODELS_DIR}/le_price_cat.pkl')
    le_rating_cat = joblib.load(f'{MODELS_DIR}/le_rating_cat.pkl')
    features = joblib.load(f'{MODELS_DIR}/features.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH)
    # Clean column names and category strings
    df.columns = df.columns.str.strip()
    if 'Category' in df.columns:
        df['Category'] = df['Category'].str.strip()
    print(f"Dataset loaded: {len(df)} rows")
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.DataFrame()

# Main categories
MAIN_CATEGORIES = ['Electronics', 'Grocery', 'Snacks', 'Beverages', 'Fruits & Vegetables', 
                   'Fashion', 'Home & Living', 'Beauty', 'Books', 'Clothing']

if not df.empty:
    df_clean = df[df['Category'].isin(MAIN_CATEGORIES)].copy()
else:
    df_clean = pd.DataFrame()

# Product name templates
PRODUCT_NAMES = {
    'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Smartwatch', 'Tablet', 'Camera', 'Speaker', 'Gaming Console', 'Keyboard', 'Mouse'],
    'Grocery': ['Rice Bag', 'Wheat Flour', 'Cooking Oil', 'Sugar Pack', 'Salt Container', 'Tea Bags', 'Coffee Pack', 'Biscuits', 'Dal Pack', 'Spices Set'],
    'Snacks': ['Potato Chips', 'Cookies Pack', 'Chocolate Bar', 'Namkeen Mix', 'Popcorn', 'Wafers', 'Crackers', 'Nut Mix', 'Candy Pack', 'Puffs'],
    'Beverages': ['Cola Bottle', 'Juice Pack', 'Energy Drink', 'Coffee', 'Tea Pack', 'Smoothie', 'Mineral Water', 'Soda Can', 'Lemonade', 'Iced Tea'],
    'Fruits & Vegetables': ['Fresh Apples', 'Bananas', 'Oranges', 'Mangoes', 'Tomatoes', 'Potatoes', 'Onions', 'Carrots', 'Spinach', 'Broccoli'],
    'Fashion': ['Stylish Shirt', 'Denim Jeans', 'Summer Dress', 'Leather Jacket', 'Sneakers', 'Formal Shoes', 'Handbag', 'Wallet', 'Sunglasses', 'Wrist Watch'],
    'Home & Living': ['Sofa Set', 'Dining Table', 'Bed Sheet Set', 'Pillow Set', 'Curtains', 'Wall Clock', 'Table Lamp', 'Flower Vase', 'Cushion Cover', 'Floor Rug'],
    'Beauty': ['Face Wash', 'Moisturizer', 'Lipstick', 'Perfume', 'Shampoo', 'Conditioner', 'Hair Oil', 'Body Lotion', 'Sunscreen', 'Face Mask'],
    'Books': ['Fiction Novel', 'Self Help Book', 'Cookbook', 'Biography', 'History Book', 'Science Fiction', 'Mystery Novel', 'Comic Book', 'Dictionary', 'Atlas'],
    'Clothing': ['T-Shirt', 'Hoodie', 'Jacket', 'Sweater', 'Shorts', 'Skirt', 'Blouse', 'Pants', 'Leggings', 'Pajamas']
}

def get_product_name(row, index):
    product_name = str(row.get('product_name', '')).strip()
    category = row.get('Category', 'Product')
    vendor = row.get('vendors', 'Store')
    
    bad_names = ['nan', 'nan_missing', 'none', '', 'null']
    if not product_name or product_name.lower() in bad_names or product_name == category:
        templates = PRODUCT_NAMES.get(category, ['Product'])
        template_name = templates[index % len(templates)]
        product_name = f"{template_name} by {vendor}"
    
    if len(product_name) > 80:
        product_name = product_name[:80] + '...'
    return product_name

def generate_sample_products(category):
    templates = PRODUCT_NAMES.get(category, ['Product A', 'Product B'])
    vendors_list = ['Amazon', 'Flipkart', 'Croma', 'Reliance Digital', 'Myntra', 'Nykaa', 'BigBasket', 'Zepto']
    price_ranges = {
        'Electronics': (500, 50000), 'Grocery': (50, 2000), 'Snacks': (20, 500),
        'Beverages': (30, 800), 'Fruits & Vegetables': (50, 1000), 'Fashion': (200, 5000),
        'Home & Living': (500, 25000), 'Beauty': (100, 3000), 'Books': (100, 1500), 'Clothing': (300, 4000)
    }
    min_price, max_price = price_ranges.get(category, (100, 5000))
    products = []
    for i, template in enumerate(templates):
        vendor = random.choice(vendors_list)
        original_price = round(random.uniform(min_price, max_price), 2)
        discount = round(random.uniform(5, 35), 2)
        final_price = round(original_price * (1 - discount/100), 2)
        rating = round(random.uniform(3.5, 5.0), 1)
        reviews = random.randint(50, 5000)
        products.append({
            'id': i, 'name': f"{template} by {vendor}", 'category': category,
            'original_price': original_price, 'final_price': final_price,
            'discount': discount, 'rating': rating, 'reviews': reviews,
            'vendor': vendor, 'sentiment': 'POSITIVE' if rating >= 4 else 'NEUTRAL'
        })
    return products

# Database Initialization
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            username TEXT NOT NULL,
            rating INTEGER NOT NULL,
            review_text TEXT NOT NULL,
            review_date TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

AUTHORIZED_USERS = {
    'admin': 'admin', 'aniruddh': 'admin123', 'ishu': 'admin123',
    'customer': 'customer', 'test': 'test'
}

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def analyze_sentiment(text):
    if not text or text.strip() == '':
        return {'label': 'NEUTRAL', 'polarity': 0.0}
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1: label = 'POSITIVE'
    elif polarity < -0.1: label = 'NEGATIVE'
    else: label = 'NEUTRAL'
    return {'label': label, 'polarity': round(polarity, 4)}

def predict_optimal_price(category, vendor, original_price, rating, review_count, 
                         sentiment_label='POSITIVE', sentiment_polarity=0.5):
    try:
        price_cat = 'Budget' if original_price <= 500 else 'Mid-Range' if original_price <= 2000 else 'Premium' if original_price <= 10000 else 'Luxury'
        rating_cat = 'Poor' if rating <= 2 else 'Average' if rating <= 3.5 else 'Good' if rating <= 4.5 else 'Excellent'
        
        cat_encoded = le_category.transform([category])[0]
        vendor_encoded = le_vendor.transform([vendor])[0]
        sent_encoded = le_sentiment.transform([sentiment_label])[0]
        price_cat_encoded = le_price_cat.transform([price_cat])[0]
        rating_cat_encoded = le_rating_cat.transform([rating_cat])[0]
        
        input_data = pd.DataFrame([[
            cat_encoded, vendor_encoded, original_price, rating, review_count,
            sent_encoded, sentiment_polarity, price_cat_encoded, rating_cat_encoded
        ]], columns=features)
        
        predicted_discount = float(discount_model.predict(input_data)[0])
        predicted_price = float(price_model.predict(input_data)[0])
        
        return {
            'predicted_discount': round(predicted_discount, 2),
            'predicted_price': round(predicted_price, 2),
            'savings': round(original_price - predicted_price, 2),
            'price_category': price_cat,
            'rating_category': rating_cat
        }
    except Exception as e:
        return {'error': str(e)}

# ============================================================
# BASIC API ENDPOINTS
# ============================================================

@app.route('/')
def home():
    return jsonify({'status': 'online', 'service': 'T&A IntelliMart API', 'version': '2.0'})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').lower()
    password = data.get('password', '')
    role = data.get('role', 'customer')
    
    if username in AUTHORIZED_USERS:
        if role == 'supplier' or role == 'admin':
            if AUTHORIZED_USERS[username] == password:
                return jsonify({'success': True, 'message': 'Admin login successful', 'user': username, 'role': 'admin'})
            else:
                return jsonify({'success': False, 'message': 'Invalid password'}), 401
        else:
            return jsonify({'success': True, 'message': 'Customer login successful', 'user': username, 'role': 'customer'})
    else:
        return jsonify({'success': False, 'message': 'User not authorized'}), 401

@app.route('/api/categories')
def get_categories():
    category_data = []
    for cat in MAIN_CATEGORIES:
        count = len(df_clean[df_clean['Category'] == cat])
        if count == 0: count = 10 # Fallback count
        category_data.append({'name': cat, 'count': count})
    return jsonify({'success': True, 'categories': category_data, 'total': len(category_data)})

@app.route('/api/vendors')
def get_vendors():
    if df_clean.empty: return jsonify({'success': True, 'vendors': [], 'total': 0})
    vendors = df_clean['vendors'].value_counts().to_dict()
    vendor_data = [{'name': k, 'count': int(v)} for k, v in vendors.items()]
    return jsonify({'success': True, 'vendors': vendor_data, 'total': len(vendor_data)})

@app.route('/api/stats')
def get_stats():
    if df_clean.empty: return jsonify({'success': True, 'total_products': 0})
    return jsonify({
        'success': True,
        'total_products': len(df_clean),
        'total_categories': df_clean['Category'].nunique(),
        'total_vendors': df_clean['vendors'].nunique(),
        'avg_price': round(df_clean['Final_Price'].mean(), 2),
        'avg_rating': round(df_clean['Rating'].mean(), 2),
        'avg_discount': round(df_clean['Discount_Percentage'].mean(), 2)
    })

# ============================================================
# PRODUCT & GRAPH ENDPOINTS
# ============================================================

@app.route('/api/products/<category>')
def get_products_by_category(category):
    if df_clean.empty:
        return jsonify({'success': True, 'category': category, 'products': generate_sample_products(category)})
        
    category_products = df_clean[df_clean['Category'] == category]
    
    if len(category_products) == 0:
        return jsonify({
            'success': True,
            'category': category,
            'total_products': 10,
            'products': generate_sample_products(category)
        })
    
    top_products = category_products.nlargest(20, 'Rating').reset_index(drop=True)
    products_list = []
    for idx, row in top_products.iterrows():
        product_name = get_product_name(row, idx)
        products_list.append({
            'id': int(idx),
            'name': product_name,
            'category': row['Category'],
            'original_price': round(float(row['Original_Price']), 2),
            'final_price': round(float(row['Final_Price']), 2),
            'discount': round(float(row['Discount_Percentage']), 2),
            'rating': round(float(row['Rating']), 2),
            'reviews': int(row['Review_Count']),
            'vendor': row['vendors'],
            'sentiment': row['Sentiment_Label']
        })
    return jsonify({
        'success': True,
        'category': category,
        'total_products': len(category_products),
        'products': products_list
    })

@app.route('/api/graphs/<category>')
def get_graphs_data(category):
    if df_clean.empty:
         return jsonify({'success': False, 'message': 'No data available'}), 404

    category_data = df_clean[df_clean['Category'] == category]
    if len(category_data) == 0:
        return jsonify({'success': False, 'message': 'Category not found'}), 404
    
    top_products = category_data.nlargest(10, 'Rating').reset_index(drop=True)
    price_vs_product = {
        'products': [get_product_name(row, i)[:30] for i, row in top_products.iterrows()],
        'prices': top_products['Final_Price'].round(2).tolist()
    }
    
    sample_data = category_data.sample(min(100, len(category_data)))
    discount_vs_price = {
        'prices': sample_data['Final_Price'].round(2).tolist(),
        'discounts': sample_data['Discount_Percentage'].round(2).tolist(),
        'vendors': sample_data['vendors'].tolist()
    }
    rating_vs_price = {
        'prices': sample_data['Final_Price'].round(2).tolist(),
        'ratings': sample_data['Rating'].round(2).tolist(),
        'vendors': sample_data['vendors'].tolist()
    }
    
    vendor_prices = category_data.groupby('vendors').agg({
        'Final_Price': 'mean',
        'Discount_Percentage': 'mean'
    }).reset_index()
    prices_by_vendor = {
        'vendors': vendor_prices['vendors'].tolist(),
        'avg_prices': vendor_prices['Final_Price'].round(2).tolist(),
        'avg_discounts': vendor_prices['Discount_Percentage'].round(2).tolist()
    }
    
    return jsonify({
        'success': True,
        'category': category,
        'total_products': len(category_data),
        'price_vs_product': price_vs_product,
        'discount_vs_price': discount_vs_price,
        'rating_vs_price': rating_vs_price,
        'prices_by_vendor': prices_by_vendor
    })

@app.route('/api/recommendations/<category>')
def get_recommendations(category):
    if df_clean.empty:
        return jsonify({'success': False, 'message': 'No data available'}), 404
        
    category_data = df_clean[df_clean['Category'] == category]
    if len(category_data) == 0:
        return jsonify({'success': False, 'message': 'Category not found'}), 404
    
    top_products = category_data.nlargest(10, 'Review_Count').reset_index(drop=True)
    recommendations = []
    for idx, row in top_products.iterrows():
        try:
            prediction = predict_optimal_price(
                category=row['Category'],
                vendor=row['vendors'],
                original_price=float(row['Original_Price']),
                rating=float(row['Rating']),
                review_count=int(row['Review_Count']),
                sentiment_label=row['Sentiment_Label'],
                sentiment_polarity=float(row['Sentiment_Polarity'])
            )
            current_price = float(row['Final_Price'])
            optimal_price = prediction['predicted_price']
            
            if optimal_price < current_price * 0.95:
                action = 'DECREASE'
                impact = round((current_price - optimal_price) / current_price * 100, 2)
            elif optimal_price > current_price * 1.05:
                action = 'INCREASE'
                impact = round((optimal_price - current_price) / current_price * 100, 2)
            else:
                action = 'MAINTAIN'
                impact = 0
            
            product_name = get_product_name(row, idx)
            recommendations.append({
                'product_name': product_name[:50],
                'vendor': row['vendors'],
                'current_price': round(current_price, 2),
                'optimal_price': optimal_price,
                'current_discount': round(float(row['Discount_Percentage']), 2),
                'optimal_discount': prediction['predicted_discount'],
                'rating': round(float(row['Rating']), 2),
                'reviews': int(row['Review_Count']),
                'action': action,
                'impact_percentage': impact
            })
        except Exception:
            continue
    
    return jsonify({'success': True, 'category': category, 'recommendations': recommendations})

@app.route('/api/insights/<category>')
def get_insights(category):
    if df_clean.empty:
        return jsonify({'success': False, 'message': 'No data available'}), 404
        
    category_data = df_clean[df_clean['Category'] == category]
    if len(category_data) == 0:
        return jsonify({'success': False, 'message': 'Category not found'}), 404
    
    avg_price = category_data['Final_Price'].mean()
    avg_discount = category_data['Discount_Percentage'].mean()
    avg_rating = category_data['Rating'].mean()
    positive_pct = (category_data['Sentiment_Label'] == 'POSITIVE').sum() / len(category_data) * 100
    top_product = category_data.nlargest(1, 'Rating').iloc[0]
    best_vendor = category_data.groupby('vendors')['Rating'].mean().idxmax()
    best_vendor_rating = category_data.groupby('vendors')['Rating'].mean().max()
    
    top_product_name = get_product_name(top_product, 0)
    
    insights = {
        'category': category,
        'total_products': len(category_data),
        'avg_price': round(avg_price, 2),
        'avg_discount': round(avg_discount, 2),
        'avg_rating': round(avg_rating, 2),
        'positive_sentiment_pct': round(positive_pct, 2),
        'best_vendor': best_vendor,
        'best_vendor_rating': round(best_vendor_rating, 2),
        'top_product': {
            'name': top_product_name[:50],
            'rating': round(float(top_product['Rating']), 2),
            'price': round(float(top_product['Final_Price']), 2)
        },
        'insights_messages': [
            f'Average price in {category}: ₹{avg_price:,.2f}',
            f'Standard discount offered: {avg_discount:.1f}%',
            f'Category average rating: {avg_rating:.2f}/5',
            f'{positive_pct:.1f}% customers are happy with {category} products',
            f'Best vendor in {category}: {best_vendor} ({best_vendor_rating:.2f} stars)'
        ]
    }
    return jsonify({'success': True, 'insights': insights})

# ============================================================
# SEARCH & PRODUCT DETAIL ENDPOINTS
# ============================================================

@app.route('/api/search/products', methods=['GET'])
def search_products():
    if df_clean.empty:
        return jsonify({'success': True, 'products': []})
        
    query = request.args.get('q', '').strip().lower()
    if not query or len(query) < 2:
        return jsonify({'success': False, 'message': 'Please enter at least 2 characters'}), 400
    
    matching_products = df_clean[
        df_clean['product_name'].astype(str).str.lower().str.contains(query, na=False, regex=False)
    ].reset_index(drop=True)
    
    if len(matching_products) == 0:
        return jsonify({'success': True, 'query': query, 'total_found': 0, 'products': []})
    
    results = matching_products.head(20)
    products_list = []
    for idx, row in results.iterrows():
        product_name = get_product_name(row, idx)
        products_list.append({
            'name': product_name,
            'category': row['Category'],
            'original_price': round(float(row['Original_Price']), 2),
            'final_price': round(float(row['Final_Price']), 2),
            'discount': round(float(row['Discount_Percentage']), 2),
            'rating': round(float(row['Rating']), 2),
            'reviews': int(row['Review_Count']),
            'vendor': row['vendors'],
            'sentiment': row['Sentiment_Label']
        })
    
    return jsonify({
        'success': True,
        'query': query,
        'total_found': len(matching_products),
        'showing': len(products_list),
        'products': products_list
    })

@app.route('/api/product/suggestions', methods=['GET'])
def product_suggestions():
    if df_clean.empty:
        return jsonify({'success': True, 'suggestions': []})

    query = request.args.get('q', '').strip().lower()
    if not query or len(query) < 2:
        return jsonify({'success': True, 'suggestions': []})
    
    matching = df_clean[
        df_clean['product_name'].astype(str).str.lower().str.contains(query, na=False, regex=False)
    ].head(10).reset_index(drop=True)
    
    suggestions = []
    seen = set()
    for idx, row in matching.iterrows():
        name = get_product_name(row, idx)
        if name not in seen:
            seen.add(name)
            suggestions.append({'name': name[:60], 'category': row['Category']})
    
    return jsonify({'success': True, 'suggestions': suggestions})

@app.route('/api/product/insights', methods=['POST'])
def get_product_insights():
    if df_clean.empty:
         return jsonify({'success': False, 'message': 'No data loaded'}), 400

    data = request.json
    product_name = data.get('product_name', '').strip()
    if not product_name:
        return jsonify({'success': False, 'message': 'Product name is required'}), 400
    
    product_match = df_clean[df_clean['product_name'].astype(str) == product_name]
    if len(product_match) == 0:
        product_match = df_clean[
            df_clean['product_name'].astype(str).str.lower().str.contains(product_name.lower(), na=False, regex=False)
        ]
    if len(product_match) == 0:
        return jsonify({'success': False, 'message': 'Product not found'}), 404
    
    product = product_match.iloc[0]
    category = product['Category']
    category_products = df_clean[df_clean['Category'] == category]
    avg_category_price = category_products['Final_Price'].mean()
    product_price = float(product['Final_Price'])
    price_comparison = ((product_price - avg_category_price) / avg_category_price * 100)
    
    display_name = get_product_name(product, 0)
    
    try:
        prediction = predict_optimal_price(
            category=product['Category'],
            vendor=product['vendors'],
            original_price=float(product['Original_Price']),
            rating=float(product['Rating']),
            review_count=int(product['Review_Count']),
            sentiment_label=product['Sentiment_Label'],
            sentiment_polarity=float(product['Sentiment_Polarity'])
        )
    except Exception as e:
        prediction = {'error': str(e)}
    
    price_range = 0.3
    min_price = product_price * (1 - price_range)
    max_price = product_price * (1 + price_range)
    similar_products = category_products[
        (category_products['Final_Price'] >= min_price) &
        (category_products['Final_Price'] <= max_price) &
        (category_products['product_name'] != product['product_name'])
    ].nlargest(5, 'Rating').reset_index(drop=True)
    
    similar_list = []
    for idx, sim in similar_products.iterrows():
        sim_name = get_product_name(sim, idx)
        similar_list.append({
            'name': sim_name[:60],
            'vendor': sim['vendors'],
            'price': round(float(sim['Final_Price']), 2),
            'rating': round(float(sim['Rating']), 2),
            'discount': round(float(sim['Discount_Percentage']), 2)
        })
    
    current_price = product_price
    if 'predicted_price' in prediction:
        optimal_price = prediction['predicted_price']
        if optimal_price < current_price * 0.95:
            action = 'DECREASE'
            impact = round((current_price - optimal_price) / current_price * 100, 2)
            action_message = f'Reduce price by {impact}%'
        elif optimal_price > current_price * 1.05:
            action = 'INCREASE'
            impact = round((optimal_price - current_price) / current_price * 100, 2)
            action_message = f'Can increase price by {impact}%'
        else:
            action = 'MAINTAIN'
            impact = 0
            action_message = 'Current pricing is optimal'
    else:
        action = 'MAINTAIN'
        impact = 0
        action_message = 'Pricing analysis unavailable'
    
    best_vendor_in_cat = category_products.groupby('vendors')['Rating'].mean().idxmax()
    best_vendor_rating = round(category_products.groupby('vendors')['Rating'].mean().max(), 2)
    
    return jsonify({
        'success': True,
        'product': {
            'name': display_name,
            'category': product['Category'],
            'vendor': product['vendors'],
            'original_price': round(float(product['Original_Price']), 2),
            'current_price': round(current_price, 2),
            'discount': round(float(product['Discount_Percentage']), 2),
            'rating': round(float(product['Rating']), 2),
            'reviews': int(product['Review_Count']),
            'sentiment': product['Sentiment_Label'],
            'sentiment_score': round(float(product['Sentiment_Polarity']), 4)
        },
        'category_analysis': {
            'category': category,
            'total_in_category': len(category_products),
            'avg_category_price': round(avg_category_price, 2),
            'price_vs_category': round(price_comparison, 2),
            'position': 'ABOVE AVERAGE' if price_comparison > 0 else 'BELOW AVERAGE',
            'avg_rating': round(category_products['Rating'].mean(), 2),
            'avg_discount': round(category_products['Discount_Percentage'].mean(), 2),
            'best_vendor': best_vendor_in_cat,
            'best_vendor_rating': best_vendor_rating
        },
        'ml_prediction': {
            'optimal_price': prediction.get('predicted_price', 0),
            'optimal_discount': prediction.get('predicted_discount', 0),
            'action': action,
            'impact_percentage': impact,
            'action_message': action_message,
            'savings': prediction.get('savings', 0)
        },
        'similar_products': similar_list,
        'insights': [
            f'Priced {abs(price_comparison):.1f}% {"above" if price_comparison > 0 else "below"} average',
            f'Sold by: {product["vendors"]}',
            f'Rating: {float(product["Rating"]):.2f}/5',
            f'Sentiment: {product["Sentiment_Label"]}',
            f'ML: {action_message}',
            f'Best vendor: {best_vendor_in_cat}',
            f'{len(similar_products)} similar products'
        ]
    })

# ============================================================
# REVIEW ENDPOINTS
# ============================================================

@app.route('/api/reviews/add', methods=['POST'])
def add_review():
    data = request.json
    product_name = data.get('product_name', '').strip()
    category = data.get('category', '').strip()
    username = data.get('username', 'Anonymous').strip()
    rating = int(data.get('rating', 5))
    review_text = data.get('review_text', '').strip()
    
    if not product_name or not category or not review_text:
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    
    if rating < 1 or rating > 5:
        return jsonify({'success': False, 'message': 'Rating must be 1-5'}), 400
    
    review_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO reviews (product_name, category, username, rating, review_text, review_date)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (product_name, category, username, rating, review_text, review_date))
        conn.commit()
        review_id = cursor.lastrowid
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Review added successfully!',
            'review_id': review_id
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/reviews/all', methods=['GET'])
def get_all_reviews():
    try:
        limit = int(request.args.get('limit', 20))
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM reviews ORDER BY review_date DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        reviews = [dict(row) for row in rows]
        return jsonify({'success': True, 'total': len(reviews), 'reviews': reviews})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'reviews': []}), 500

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("T&A IntelliMart Backend Ready!")
    print("API running at: http://localhost:5000")
    print("=" * 80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)