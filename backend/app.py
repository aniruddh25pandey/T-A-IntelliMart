"""
T&A IntelliMart - Flask Backend API
Complete backend with ML predictions, product search, and business logic
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ============================================================
# LOAD MODELS AND DATA
# ============================================================

print("=" * 80)
print("T&A IntelliMart Backend Starting...")
print("=" * 80)

MODELS_DIR = 'models'
DATASET_PATH = '../DATASET/dataset_with_sentiment.csv'

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
    """Analyze sentiment of text"""
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
    """Predict optimal price and discount using ML models"""
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

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'service': 'T&A IntelliMart API',
        'version': '2.0',
        'endpoints': [
            '/api/login', '/api/categories', '/api/vendors', '/api/stats',
            '/api/products/<category>', '/api/graphs/<category>', '/api/predict',
            '/api/recommendations/<category>', '/api/insights/<category>',
            '/api/sentiment', '/api/search/products', '/api/product/insights',
            '/api/product/suggestions'
        ]
    })

@app.route('/api/login', methods=['POST'])
def login():
    """User authentication endpoint"""
    data = request.json
    username = data.get('username', '').lower()
    password = data.get('password', '')
    role = data.get('role', 'customer')
    
    if username in AUTHORIZED_USERS:
        if role == 'supplier' or role == 'admin':
            if AUTHORIZED_USERS[username] == password:
                return jsonify({
                    'success': True,
                    'message': 'Admin login successful',
                    'user': username,
                    'role': 'admin'
                })
            else:
                return jsonify({'success': False, 'message': 'Invalid password'}), 401
        else:
            return jsonify({
                'success': True,
                'message': 'Customer login successful',
                'user': username,
                'role': 'customer'
            })
    else:
        return jsonify({'success': False, 'message': 'User not authorized'}), 401

@app.route('/api/categories')
def get_categories():
    """Get all available categories with product counts"""
    category_data = []
    for cat in MAIN_CATEGORIES:
        count = len(df_clean[df_clean['Category'] == cat])
        if count > 0:
            category_data.append({'name': cat, 'count': count})
    
    return jsonify({
        'success': True,
        'categories': category_data,
        'total': len(category_data)
    })

@app.route('/api/vendors')
def get_vendors():
    """Get all vendors with product counts"""
    vendors = df_clean['vendors'].value_counts().to_dict()
    vendor_data = [{'name': k, 'count': int(v)} for k, v in vendors.items()]
    return jsonify({
        'success': True,
        'vendors': vendor_data,
        'total': len(vendor_data)
    })

@app.route('/api/stats')
def get_stats():
    """Get overall statistics"""
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
    """Get top products in a specific category"""
    category_products = df_clean[df_clean['Category'] == category]
    
    if len(category_products) == 0:
        return jsonify({'success': False, 'message': 'Category not found'}), 404
    
    top_products = category_products.nlargest(20, 'Rating')
    
    products_list = []
    for _, row in top_products.iterrows():
        products_list.append({
            'name': str(row['product_name'])[:60],
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
    """Get data for 4 different graphs for a category"""
    category_data = df_clean[df_clean['Category'] == category]
    
    if len(category_data) == 0:
        return jsonify({'success': False, 'message': 'Category not found'}), 404
    
    # Graph 1: Price vs Product Name (Top 10)
    top_products = category_data.nlargest(10, 'Rating')
    price_vs_product = {
        'products': [str(p)[:30] for p in top_products['product_name'].tolist()],
        'prices': top_products['Final_Price'].round(2).tolist()
    }
    
    # Graph 2 & 3: Sample data for scatter plots
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
    
    # Graph 4: Prices and Discounts by Vendor
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
    """Predict optimal price and discount for custom input"""
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
    """Get ML-powered price recommendations for top products in category"""
    category_data = df_clean[df_clean['Category'] == category]
    
    if len(category_data) == 0:
        return jsonify({'success': False, 'message': 'Category not found'}), 404
    
    top_products = category_data.nlargest(10, 'Review_Count')
    
    recommendations = []
    for _, row in top_products.iterrows():
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
            
            recommendations.append({
                'product_name': str(row['product_name'])[:50],
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
        except Exception as e:
            continue
    
    return jsonify({
        'success': True,
        'category': category,
        'recommendations': recommendations
    })

@app.route('/api/insights/<category>')
def get_insights(category):
    """Get business insights for a category"""
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
            'name': str(top_product['product_name'])[:50],
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
    """Analyze sentiment of customer review text"""
    data = request.json
    review = data.get('review', '')
    result = analyze_sentiment(review)
    return jsonify({'success': True, 'review': review, 'sentiment': result})

@app.route('/api/search/products', methods=['GET'])
def search_products():
    """Search products by name"""
    query = request.args.get('q', '').strip().lower()
    
    if not query or len(query) < 2:
        return jsonify({
            'success': False,
            'message': 'Please enter at least 2 characters'
        }), 400
    
    matching_products = df_clean[
        df_clean['product_name'].str.lower().str.contains(query, na=False, regex=False)
    ]
    
    if len(matching_products) == 0:
        return jsonify({
            'success': True,
            'query': query,
            'total_found': 0,
            'products': [],
            'message': f'No products found matching "{query}"'
        })
    
    results = matching_products.head(20)
    
    products_list = []
    for _, row in results.iterrows():
        products_list.append({
            'name': str(row['product_name'])[:80],
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
    """Get detailed insights for a specific product"""
    data = request.json
    product_name = data.get('product_name', '').strip()
    
    if not product_name:
        return jsonify({'success': False, 'message': 'Product name is required'}), 400
    
    # Try exact match first
    product_match = df_clean[df_clean['product_name'] == product_name]
    
    # If no exact match, try partial match
    if len(product_match) == 0:
        product_match = df_clean[
            df_clean['product_name'].str.lower().str.contains(
                product_name.lower(), na=False, regex=False
            )
        ]
    
    if len(product_match) == 0:
        return jsonify({'success': False, 'message': 'Product not found'}), 404
    
    product = product_match.iloc[0]
    category = product['Category']
    
    category_products = df_clean[df_clean['Category'] == category]
    
    avg_category_price = category_products['Final_Price'].mean()
    product_price = float(product['Final_Price'])
    price_comparison = ((product_price - avg_category_price) / avg_category_price * 100)
    
    # Get ML prediction
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
    
    # Find similar products
    price_range = 0.3
    min_price = product_price * (1 - price_range)
    max_price = product_price * (1 + price_range)
    
    similar_products = category_products[
        (category_products['Final_Price'] >= min_price) &
        (category_products['Final_Price'] <= max_price) &
        (category_products['product_name'] != product['product_name'])
    ].nlargest(5, 'Rating')
    
    similar_list = []
    for _, sim in similar_products.iterrows():
        similar_list.append({
            'name': str(sim['product_name'])[:60],
            'vendor': sim['vendors'],
            'price': round(float(sim['Final_Price']), 2),
            'rating': round(float(sim['Rating']), 2),
            'discount': round(float(sim['Discount_Percentage']), 2)
        })
    
    # Determine action based on ML prediction
    current_price = product_price
    if 'predicted_price' in prediction:
        optimal_price = prediction['predicted_price']
        if optimal_price < current_price * 0.95:
            action = 'DECREASE'
            impact = round((current_price - optimal_price) / current_price * 100, 2)
            action_message = f'<i class="fas fa-arrow-down text-red-500 mr-2"></i>Reduce price by {impact}% to boost sales'
        elif optimal_price > current_price * 1.05:
            action = 'INCREASE'
            impact = round((optimal_price - current_price) / current_price * 100, 2)
            action_message = f'<i class="fas fa-arrow-up text-teal-500 mr-2"></i>Can increase price by {impact}% (demand is high)'
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
            'name': str(product['product_name']),
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
            f'<i class="fas fa-chart-bar text-purple-500 mr-2"></i>This product is priced {abs(price_comparison):.1f}% {"above" if price_comparison > 0 else "below"} category average',
            f'<i class="fas fa-store text-blue-500 mr-2"></i>Sold by: {product["vendors"]}',
            f'<i class="fas fa-star text-yellow-500 mr-2"></i>Rating: {float(product["Rating"]):.2f}/5 ({"Excellent" if product["Rating"] > 4.5 else "Good" if product["Rating"] > 3.5 else "Average"})',
            f'<i class="fas fa-comment-dots text-pink-500 mr-2"></i>Customer sentiment: {product["Sentiment_Label"]} ({float(product["Sentiment_Polarity"]):.2f})',
            f'<i class="fas fa-robot text-indigo-500 mr-2"></i>ML Recommendation: {action_message}',
            f'<i class="fas fa-trophy text-amber-500 mr-2"></i>Best vendor in this category: {best_vendor_in_cat} ({best_vendor_rating} stars)',
            f'<i class="fas fa-layer-group text-teal-500 mr-2"></i>{len(similar_products)} similar products available in this category'
        ]
    })

@app.route('/api/product/suggestions', methods=['GET'])
def product_suggestions():
    """Auto-complete suggestions for product search"""
    query = request.args.get('q', '').strip().lower()
    
    if not query or len(query) < 2:
        return jsonify({'success': True, 'suggestions': []})
    
    matching = df_clean[
        df_clean['product_name'].str.lower().str.contains(query, na=False, regex=False)
    ].head(10)
    
    suggestions = []
    seen = set()
    for _, row in matching.iterrows():
        name = str(row['product_name'])[:60]
        if name not in seen:
            seen.add(name)
            suggestions.append({
                'name': name,
                'category': row['Category']
            })
    
    return jsonify({'success': True, 'suggestions': suggestions})

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("Frontend should point to: http://localhost:5000/api/*")
    print("\n" + "=" * 80)
    app.run(debug=True, host='0.0.0.0', port=5000)