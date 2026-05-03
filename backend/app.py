"""
T&A IntelliMart - Flask Backend API (COMPLETE with Reviews + Sample Products)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
import random
from datetime import datetime
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

print("=" * 80)
print("T&A IntelliMart Backend Starting...")
print("=" * 80)

MODELS_DIR = 'models'
DATASET_PATH = '../DATASET/dataset_with_sentiment.csv'
DB_PATH = 'reviews.db'

print("\nLoading ML Models...")
try:
    discount_model = joblib.load(f'{MODELS_DIR}/discount_model.pkl')
    price_model = joblib.load(f'{MODELS_DIR}/price_model.pkl')
    le_category = joblib.load(f'{MODELS_DIR}/le_category.pkl')
    le_vendor = joblib.load(f'{MODELS_DIR}/le_vendor.pkl')
    le_sentiment = joblib.load(f'{MODELS_DIR}/le_sentiment.pkl')
    le_price_cat = joblib.load(f'{MODELS_DIR}/le_price_cat.pkl')
    le_rating_cat = joblib.load(f'{MODELS_DIR}/le_rating_cat.pkl')
    features = joblib.load(f'{MODELS_DIR}/features.pkl')
    metadata = joblib.load(f'{MODELS_DIR}/metadata.pkl')
    print("All ML models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

print("\nLoading Dataset...")
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {len(df):,} products")
except Exception as e:
    print(f"Error loading dataset: {e}")

MAIN_CATEGORIES = ['Electronics', 'Grocery', 'Snacks', 'Beverages', 
                   'Fruits & Vegetables', 'Fashion', 'Home & Living', 
                   'Beauty', 'Books', 'Clothing']
df_clean = df[df['Category'].isin(MAIN_CATEGORIES)].copy()
print(f"Clean data: {len(df_clean):,} products in main categories")

# ============================================================
# PRODUCT NAME TEMPLATES
# ============================================================

PRODUCT_NAMES = {
    'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Smartwatch', 'Tablet', 'Camera', 'Speaker', 'Gaming Console', 'Keyboard', 'Mouse', 'Monitor', 'Router', 'Power Bank', 'Charger', 'USB Cable'],
    'Grocery': ['Rice Bag', 'Wheat Flour', 'Cooking Oil', 'Sugar Pack', 'Salt Container', 'Tea Bags', 'Coffee Pack', 'Biscuits', 'Dal Pack', 'Spices Set', 'Pasta', 'Noodles', 'Bread Loaf', 'Eggs Pack', 'Milk Carton'],
    'Snacks': ['Potato Chips', 'Cookies Pack', 'Chocolate Bar', 'Namkeen Mix', 'Popcorn', 'Wafers', 'Crackers', 'Nut Mix', 'Candy Pack', 'Puffs', 'Sev Mixture', 'Bhujia', 'Trail Mix', 'Granola Bar', 'Pretzels'],
    'Beverages': ['Cola Bottle', 'Juice Pack', 'Energy Drink', 'Coffee', 'Tea Pack', 'Smoothie', 'Mineral Water', 'Soda Can', 'Lemonade', 'Iced Tea', 'Protein Shake', 'Milkshake', 'Sports Drink', 'Coconut Water', 'Flavored Water'],
    'Fruits & Vegetables': ['Fresh Apples', 'Bananas', 'Oranges', 'Mangoes', 'Tomatoes', 'Potatoes', 'Onions', 'Carrots', 'Spinach', 'Broccoli', 'Cucumber', 'Bell Peppers', 'Lettuce', 'Grapes', 'Strawberries'],
    'Fashion': ['Stylish Shirt', 'Denim Jeans', 'Summer Dress', 'Leather Jacket', 'Sneakers', 'Formal Shoes', 'Handbag', 'Wallet', 'Sunglasses', 'Wrist Watch', 'Belt', 'Scarf', 'Hat', 'Kurta', 'Saree'],
    'Home & Living': ['Sofa Set', 'Dining Table', 'Bed Sheet Set', 'Pillow Set', 'Curtains', 'Wall Clock', 'Table Lamp', 'Flower Vase', 'Cushion Cover', 'Floor Rug', 'Photo Frame', 'Scented Candles', 'Storage Box', 'Wall Mirror', 'Office Chair'],
    'Beauty': ['Face Wash', 'Moisturizer', 'Lipstick', 'Perfume', 'Shampoo', 'Conditioner', 'Hair Oil', 'Body Lotion', 'Sunscreen', 'Face Mask', 'Nail Polish', 'Kajal', 'Foundation', 'Makeup Kit', 'Deodorant'],
    'Books': ['Fiction Novel', 'Self Help Book', 'Cookbook', 'Biography', 'History Book', 'Science Fiction', 'Mystery Novel', 'Comic Book', 'Dictionary', 'Atlas', 'Poetry Collection', 'Travel Guide', 'Business Book', 'Children Book', 'Art Book'],
    'Clothing': ['T-Shirt', 'Hoodie', 'Jacket', 'Sweater', 'Shorts', 'Skirt', 'Blouse', 'Pants', 'Leggings', 'Pajamas', 'Socks', 'Gloves', 'Coat', 'Blazer', 'Tracksuit']
}

def get_product_name(row, index):
    """Generate meaningful product name"""
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
    """Generate sample products for categories with no data"""
    templates = PRODUCT_NAMES.get(category, ['Product A', 'Product B', 'Product C'])
    vendors_list = ['Amazon', 'Flipkart', 'Croma', 'Reliance Digital', 'Myntra', 'Nykaa', 'BigBasket', 'Zepto']
    
    price_ranges = {
        'Electronics': (500, 50000),
        'Grocery': (50, 2000),
        'Snacks': (20, 500),
        'Beverages': (30, 800),
        'Fruits & Vegetables': (50, 1000),
        'Fashion': (200, 5000),
        'Home & Living': (500, 25000),
        'Beauty': (100, 3000),
        'Books': (100, 1500),
        'Clothing': (300, 4000)
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
            'id': i,
            'name': f"{template} by {vendor}",
            'category': category,
            'original_price': original_price,
            'final_price': final_price,
            'discount': discount,
            'rating': rating,
            'reviews': reviews,
            'vendor': vendor,
            'sentiment': 'POSITIVE' if rating >= 4 else 'NEUTRAL'
        })
    
    return products

# ============================================================
# DATABASE
# ============================================================

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
    print("SQLite database initialized!")

init_db()

print("\n" + "=" * 80)
print("Backend Ready!")
print("=" * 80)

AUTHORIZED_USERS = {
    'admin': 'admin',
    'aniruddh': 'admin123',
    'ishu': 'admin123',
    'customer': 'customer',
    'test': 'test'
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def analyze_sentiment(text):
    if not text or text.strip() == '':
        return {'label': 'NEUTRAL', 'polarity': 0.0}
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        label = 'POSITIVE'
    elif polarity < -0.1:
        label = 'NEGATIVE'
    else:
        label = 'NEUTRAL'
    return {'label': label, 'polarity': round(polarity, 4)}

def predict_optimal_price(category, vendor, original_price, rating, review_count, 
                         sentiment_label='POSITIVE', sentiment_polarity=0.5):
    try:
        if original_price <= 500:
            price_cat = 'Budget'
        elif original_price <= 2000:
            price_cat = 'Mid-Range'
        elif original_price <= 10000:
            price_cat = 'Premium'
        else:
            price_cat = 'Luxury'
        
        if rating <= 2:
            rating_cat = 'Poor'
        elif rating <= 3.5:
            rating_cat = 'Average'
        elif rating <= 4.5:
            rating_cat = 'Good'
        else:
            rating_cat = 'Excellent'
        
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

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/')
def home():
    return jsonify({'status': 'online', 'service': 'T&A IntelliMart API', 'version': '2.2'})

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
        # Always show categories - use sample count of 15 if empty
        if count == 0:
            count = 15
        category_data.append({'name': cat, 'count': count})
    return jsonify({'success': True, 'categories': category_data, 'total': len(category_data)})

@app.route('/api/vendors')
def get_vendors():
    vendors = df_clean['vendors'].value_counts().to_dict()
    vendor_data = [{'name': k, 'count': int(v)} for k, v in vendors.items()]
    return jsonify({'success': True, 'vendors': vendor_data, 'total': len(vendor_data)})

@app.route('/api/stats')
def get_stats():
    return jsonify({
        'success': True,
        'total_products': len(df_clean),
        'total_categories': df_clean['Category'].nunique(),
        'total_vendors': df_clean['vendors'].nunique(),
        'avg_price': round(df_clean['Final_Price'].mean(), 2),
        'avg_rating': round(df_clean['Rating'].mean(), 2),
        'avg_discount': round(df_clean['Discount_Percentage'].mean(), 2),
        'positive_reviews': int((df_clean['Sentiment_Label'] == 'POSITIVE').sum()),
        'negative_reviews': int((df_clean['Sentiment_Label'] == 'NEGATIVE').sum()),
        'positive_percentage': round((df_clean['Sentiment_Label'] == 'POSITIVE').sum() / len(df_clean) * 100, 2)
    })

@app.route('/api/products/<category>')
def get_products_by_category(category):
    category_products = df_clean[df_clean['Category'] == category]
    
    # Generate sample products if category is empty
    if len(category_products) == 0:
        products_list = generate_sample_products(category)
        return jsonify({
            'success': True,
            'category': category,
            'total_products': len(products_list),
            'products': products_list
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

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    result = predict_optimal_price(
        category=data.get('category'),
        vendor=data.get('vendor'),
        original_price=float(data.get('original_price', 0)),
        rating=float(data.get('rating', 4.0)),
        review_count=int(data.get('review_count', 0)),
        sentiment_label=data.get('sentiment_label', 'POSITIVE'),
        sentiment_polarity=float(data.get('sentiment_polarity', 0.5))
    )
    if 'error' in result:
        return jsonify({'success': False, 'message': result['error']}), 400
    return jsonify({'success': True, 'prediction': result})

@app.route('/api/recommendations/<category>')
def get_recommendations(category):
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
            f'<i class="fas fa-chart-bar text-purple-500 mr-2"></i>Average price in {category}: ₹{avg_price:,.2f}',
            f'<i class="fas fa-tag text-teal-500 mr-2"></i>Standard discount offered: {avg_discount:.1f}%',
            f'<i class="fas fa-star text-yellow-500 mr-2"></i>Category average rating: {avg_rating:.2f}/5',
            f'<i class="fas fa-heart text-pink-500 mr-2"></i>{positive_pct:.1f}% customers are happy with {category} products',
            f'<i class="fas fa-trophy text-amber-500 mr-2"></i>Best vendor in {category}: {best_vendor} ({best_vendor_rating:.2f} stars)',
            f'<i class="fas fa-chart-line text-blue-500 mr-2"></i>Recommended action: {"Increase discount" if avg_discount < 20 else "Maintain current pricing strategy"}',
            f'<i class="fas fa-bullseye text-red-500 mr-2"></i>Focus on: {"Quality improvement" if avg_rating < 4 else "Brand building"}'
        ]
    }
    return jsonify({'success': True, 'insights': insights})

@app.route('/api/sentiment', methods=['POST'])
def analyze_review_sentiment():
    data = request.json
    review = data.get('review', '')
    result = analyze_sentiment(review)
    return jsonify({'success': True, 'review': review, 'sentiment': result})

@app.route('/api/search/products', methods=['GET'])
def search_products():
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

@app.route('/api/product/insights', methods=['POST'])
def get_product_insights():
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
            action_message = f'<i class="fas fa-arrow-down text-red-500 mr-2"></i>Reduce price by {impact}%'
        elif optimal_price > current_price * 1.05:
            action = 'INCREASE'
            impact = round((optimal_price - current_price) / current_price * 100, 2)
            action_message = f'<i class="fas fa-arrow-up text-teal-500 mr-2"></i>Can increase price by {impact}%'
        else:
            action = 'MAINTAIN'
            impact = 0
            action_message = '<i class="fas fa-check-circle text-teal-500 mr-2"></i>Current pricing is optimal'
    else:
        action = 'MAINTAIN'
        impact = 0
        action_message = '<i class="fas fa-info-circle text-blue-500 mr-2"></i>Pricing analysis unavailable'
    
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
            f'<i class="fas fa-chart-bar text-purple-500 mr-2"></i>Priced {abs(price_comparison):.1f}% {"above" if price_comparison > 0 else "below"} average',
            f'<i class="fas fa-store text-blue-500 mr-2"></i>Sold by: {product["vendors"]}',
            f'<i class="fas fa-star text-yellow-500 mr-2"></i>Rating: {float(product["Rating"]):.2f}/5',
            f'<i class="fas fa-comment-dots text-pink-500 mr-2"></i>Sentiment: {product["Sentiment_Label"]}',
            f'<i class="fas fa-robot text-indigo-500 mr-2"></i>ML: {action_message}',
            f'<i class="fas fa-trophy text-amber-500 mr-2"></i>Best vendor: {best_vendor_in_cat}',
            f'<i class="fas fa-layer-group text-teal-500 mr-2"></i>{len(similar_products)} similar products'
        ]
    })

@app.route('/api/product/suggestions', methods=['GET'])
def product_suggestions():
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

@app.route('/api/reviews/<product_name>', methods=['GET'])
def get_product_reviews(product_name):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM reviews WHERE product_name = ? ORDER BY review_date DESC', (product_name,))
        rows = cursor.fetchall()
        conn.close()
        
        reviews = [dict(row) for row in rows]
        return jsonify({'success': True, 'product_name': product_name, 'total': len(reviews), 'reviews': reviews})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'reviews': []}), 500

@app.route('/api/reviews/delete/<int:review_id>', methods=['DELETE'])
def delete_review(review_id):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM reviews WHERE id = ?', (review_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Review deleted'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\n" + "=" * 80)
    app.run(debug=True, host='0.0.0.0', port=5000)