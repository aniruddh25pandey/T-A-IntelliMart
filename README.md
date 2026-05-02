\# 🛍️ T&A IntelliMart - Smart E-commerce Pricing System

\### 🧠 AI-Powered E-commerce Pricing & Customer Intelligence Platform

!\[Python\](https://img.shields.io/badge/Python-3.13-blue)

!\[Flask\](https://img.shields.io/badge/Flask-3.0-black)

!\[ML\](https://img.shields.io/badge/ML-Random%20Forest-green)

!\[Accuracy\](https://img.shields.io/badge/Accuracy-99.63%25-brightgreen)

!\[Status\](https://img.shields.io/badge/Status-Complete-success)

\---

\## 📖 About The Project

\*\*T&A IntelliMart\*\* is a smart e-commerce pricing and customer intelligence system that helps both customers and sellers make better decisions using product data, customer reviews, and machine learning. The system analyzes various factors such as pricing, ratings, discounts, and customer feedback to generate intelligent pricing recommendations and real-time business insights.

\---

| Metric | Value |
| --- | --- |
| 📦 Total Products Analyzed | 188,301 |
| 🎯 ML Model Accuracy | 99.63% |
| 😊 Positive Sentiment | 92.13% |
| 🏪 Active Vendors | 13 |
| 📁 Product Categories | 10+ |
| 📝 Reviews Analyzed | 188,301 |
| 🌲 ML Algorithm | Random Forest |
| ⏱️ Training Time | ~3 minutes |

\## 🎯 Project Highlights

**✨ Key Features**

**👤 Customer Panel**

*   🏠 Beautiful Dashboard with pink aesthetic theme
*   🛒 Browse Products by 10+ categories
*   💰 Smart Price Recommendations powered by ML
*   🏷️ Best Deals Finder
*   ⭐ Customer Reviews Analysis
*   💡 Buy/Wait Insights based on market trends
*   🎭 Sentiment-based product recommendations

**👨‍💼 Admin Panel**

*   📊 Real-time Analytics Dashboard (Purple theme)
*   🔍 Product Search with live auto-complete
*   📈 4 Interactive Graphs (Price, Discount, Rating, Vendor)
*   🏪 Vendor Comparison & Best Vendor Analysis
*   🎭 Sentiment Analysis (POSITIVE/NEGATIVE/NEUTRAL)
*   🤖 ML-Powered Price Recommendations
*   💰 Win-Win Pricing Calculator
*   📊 Category-wise Business Insights

**🧠 Tech Stack**

| Category | Technology |
| --- | --- |
| Frontend | HTML5, CSS3, JavaScript |
| Styling | Tailwind CSS |
| Charts | Plotly.js |
| Icons | Font Awesome |
| Backend | Python 3.13, Flask 3.0 |
| API | RESTful with Flask-CORS |
| ML Framework | Scikit-learn |
| ML Algorithm | Random Forest (100 trees) |
| NLP | TextBlob |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Version Control | Git, GitHub |
| IDE | VS Code with Jupyter |

**📁 Project Structure**

T&A-IntelliMart/

│

├── 📁 backend/

│ ├── 🐍 app.py # Flask API server (14 endpoints)

│ ├── 📁 models/ # Trained ML models (.pkl files)

│ │ ├── discount\_model.pkl

│ │ ├── price\_model.pkl

│ │ ├── le\_category.pkl

│ │ ├── le\_vendor.pkl

│ │ ├── le\_sentiment.pkl

│ │ └── features.pkl

│ └── 📁 visualizations/ # Generated graph images

│

├── 📁 DATASET/

│ ├── 📊 ecommerce.csv # Original dataset (200k rows)

│ ├── 📊 cleaned\_dataset.csv # Cleaned data (188k rows)

│ └── 📊 dataset\_with\_sentiment.csv # With sentiment analysis

│

├── 📁 frontend/

│ ├── 🎨 login.html # Purple-themed login page

│ ├── 🎨 customer-dashboard.html # Pink aesthetic customer panel

│ └── 🎨 admin-dashboard.html # Purple professional admin panel

│

├── 📁 notebooks/

│ ├── 📓 01\_preprocessing.ipynb # Data cleaning (188k rows)

│ ├── 📓 02\_sentiment.ipynb # Sentiment analysis (92% positive)

│ ├── 📓 03\_training.ipynb # ML model training (99.63% accuracy)

│ └── 📓 04\_visualizations.ipynb # 9 professional graphs

│

├── 📄 .gitignore

└── 📄 README.md

**🚀 Getting Started**

**Prerequisites**

| Requirement | Version |
| --- | --- |
| Python | 3.8 or higher |
| VS Code | Latest |
| Git | Latest |
| Browser | Chrome/Firefox/Edge |

**Installation Steps**

**1\. Clone the repository:**

bash

Copy code

git clone https://github.com/aniruddh25pandey/T-A-IntelliMart.git

cd T-A-IntelliMart

**2\. Install Python libraries:**

bash

Copy code

pip install pandas numpy scikit-learn matplotlib seaborn textblob flask flask-cors joblib plotly nltk

**3\. Download NLP data:**

bash

Copy code

python -m textblob.download\_corpora

**4\. Start the Flask backend:**

bash

Copy code

cd backend

python app.py

**5\. Open the frontend:**

*   Navigate to **frontend/** folder
*   Open **login.html** in your browser
*   Login as:
    *   **Customer:** Any username → Click "Customer"
    *   **Admin:** username **ishu** or **aniruddh** → Click "Supplier"

**🎯 ML Model Performance**

| Metric | Value |
| --- | --- |
| Algorithm | Random Forest |
| Number of Trees | 100 |
| Max Depth | 20 |
| Training Data | 150,640 products (80%) |
| Testing Data | 37,661 products (20%) |
| Price Prediction Accuracy | 99.63% |
| Discount Prediction Accuracy | 98.75% |
| Training Time | ~3 minutes |
| Features Used | 10 |

**Features Used in Model**

| Feature | Type | Description |
| --- | --- | --- |
| Category | Categorical | Product category (encoded) |
| Vendor | Categorical | Seller name (encoded) |
| Original Price | Numerical | MRP of product |
| Rating | Numerical | Customer rating (1-5) |
| Review Count | Numerical | Number of reviews |
| Sentiment Label | Categorical | POSITIVE/NEGATIVE (encoded) |
| Sentiment Polarity | Numerical | Score (-1 to +1) |
| Price Category | Categorical | Budget/Mid/Premium/Luxury |
| Rating Category | Categorical | Poor/Average/Good/Excellent |

**🔌 API Endpoints**

| Endpoint | Method | Description |
| --- | --- | --- |
| / | GET | API status and info |
| /api/login | POST | User authentication |
| /api/categories | GET | Get all categories with counts |
| /api/vendors | GET | Get all vendors with counts |
| /api/stats | GET | Overall statistics |
| /api/products/<category> | GET | Products by category |
| /api/graphs/<category> | GET | Graph data for 4 charts |
| /api/predict | POST | ML price prediction |
| /api/recommendations/<category> | GET | ML recommendations |
| /api/insights/<category> | GET | Business insights |
| /api/sentiment | POST | Sentiment analysis |
| /api/search/products | GET | Product search |
| /api/product/insights | POST | Detailed product insights |
| /api/product/suggestions | GET | Auto-complete suggestions |

**📊 Dataset Information**

| Attribute | Details |
| --- | --- |
| Source | E-commerce product data |
| Original Size | 200,000 rows |
| Cleaned Size | 188,301 rows |
| Original Columns | 9 |
| Total Columns | 13 (after feature engineering) |
| Categories | 10+ (Electronics, Grocery, Fashion, etc.) |
| Vendors | 13 (Amazon, Flipkart, Zomato, etc.) |
| Price Range | ₹5 to ₹1,00,000+ |
| Rating Range | 0.0 to 5.0 stars |

**💡 Key Insights Discovered**

| Insight | Value |
| --- | --- |
| 😊 Customer Satisfaction | 92.13% positive |
| 💰 Most Expensive Category | Electronics (₹3,006 avg) |
| 💎 Most Affordable Category | Beauty (₹155 avg) |
| 🏆 Biggest Vendor | Flipkart (21,501 products) |
| 🥈 Second Biggest | Amazon (21,225 products) |
| 🏷️ Optimal Discount | 10-25% (sweet spot) |
| ⭐ Price-Rating Correlation | 0.07 (no strong link) |
| 💚 Best Rated Category | Electronics (4.2/5 avg) |

**🎨 Screenshots**

**🔐 Login Page**

Purple gradient theme with role-based access (Customer/Supplier)

**🛍️ Customer Dashboard**

Pink aesthetic with Gen-Z vibes, floating cards, and navigation

**👨‍💼 Admin Dashboard**

Professional purple theme with real-time analytics, 4 interactive graphs, and ML recommendations

**🎯 Business Impact**

This system helps:

| User Type | Benefits |
| --- | --- |
| 🛒 Online Sellers | Optimize pricing for maximum profit |
| 🏪 E-commerce Platforms | Understand customer behavior |
| 📊 Business Analysts | Make data-driven decisions |
| 👥 Customers | Get best deals and recommendations |
| 🎓 Students | Learn full-stack + ML development |

**👥 Team**

| Developer | Role | GitHub |
| --- | --- | --- |
| Tanisha Khare | Developer | @aniruddh25pandey |
| Aniruddh Pradeep Pandey | Developer | @TK404-1425 |

**Responsibilities**

| Member | Tasks |
| --- | --- |
| Aniruddh | Frontend, Backend, ML Pipeline, Integration |
| Tanisha | Documentation, ML Pipeline ,Testing, UI Enhancements |

**🚀 Future Enhancements**

*   \[ \] 📱 Mobile app version (React Native)
*   \[ \] 🗄️ Database integration (PostgreSQL/MongoDB)
*   \[ \] 🔐 Advanced authentication (OAuth, JWT)
*   \[ \] 📧 Email notifications for price drops
*   \[ \] 🌐 Deploy to cloud (AWS/Heroku/Vercel)
*   \[ \] 🤖 Chatbot for customer support
*   \[ \] 📊 Real-time data streaming
*   \[ \] 🛒 Shopping cart functionality
*   \[ \] 💳 Payment gateway integration
*   \[ \] 📦 Order tracking system
*   \[ \] 🌍 Multi-language support
*   \[ \] 🎨 Dark mode theme

**🎓 Academic Context**

This project demonstrates:

| Skill | Application |
| --- | --- |
| Data Preprocessing | 188k rows cleaned |
| Feature Engineering | 4 new features created |
| Sentiment Analysis | NLP with TextBlob |
| Machine Learning | Random Forest training |
| Model Evaluation | 99.63% accuracy achieved |
| Frontend Development | 3 responsive pages |
| Backend Development | 14 RESTful APIs |
| Data Visualization | 9+ professional charts |
| Version Control | Git/GitHub workflow |
| Project Management | Systematic development |

**📝 License**

This project is developed for **educational and portfolio purposes**.

**🙏 Acknowledgments**

*   📊 Dataset providers
*   🌐 Open-source community
*   🧠 Scikit-learn team
*   🔥 Flask development team
*   📈 Plotly team
*   👨‍🏫 Our mentors and supporters

**📧 Contact**

For queries or collaborations:

| Contact | Details |
| --- | --- |
| 📧 Email | aniruddhpandey8423@gmail.com |
| 💻 GitHub | @aniruddh25pandey / or @TK404-1425 |
| 🔗 Project Link | T-A-IntelliMart |

**⭐ If you find this project helpful, please give it a star!**

**Made with ❤️ using Python, Flask, and Machine Learning**