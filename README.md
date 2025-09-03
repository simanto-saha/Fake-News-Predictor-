# Fake News Predictor Using Classical Machine Learning Models 📰🔍

A comprehensive machine learning project that classifies news articles as real or fake using multiple algorithms and natural language processing techniques.

## 🌟 Features

- **Multiple ML Models**: Compares 6 different machine learning algorithms
- **Text Preprocessing**: Advanced text cleaning and feature extraction
- **Interactive Prediction**: Real-time news classification interface
- **Performance Visualization**: Comprehensive charts and confusion matrices
- **Model Persistence**: Save and load trained models
- **Exploratory Data Analysis**: In-depth data insights and visualizations

## 🤖 Machine Learning Models

1. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
2. **Logistic Regression** - Linear classification with probability output
3. **Random Forest** - Ensemble method using multiple decision trees
4. **Gradient Boosting** - Sequential learning with error correction
5. **AdaBoost** - Adaptive boosting for weak learner combination
6. **SGD Classifier** - Stochastic gradient descent optimization

## 📊 Dataset Requirements

Your CSV file should contain:
- `title`: News article headlines
- `text`: News article content
- `label`: Classification labels (0 = Real, 1 = Fake)

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas scikit-learn matplotlib seaborn numpy pickle-mixin
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. Place your dataset CSV file in the project directory

3. Update the file path in the code:
```python
read = pd.read_csv('your_dataset.csv')
```

### Usage

1. **Run the complete pipeline:**
```python
python fake_news_detection.py
```

2. **The system will:**
   - Load and preprocess your data
   - Train all 6 models
   - Display performance comparisons
   - Show visualizations
   - Save the best model
   - Launch interactive prediction interface

3. **Interactive Prediction:**
   - Enter news title (optional)
   - Enter news content
   - Get real-time classification results

## 📈 Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall correctness percentage
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Balanced precision-recall measure
- **Confusion Matrix**: Detailed classification breakdown

## 🔧 Code Structure

### Data Processing
```python
def clean_text(text):
    # Converts to lowercase and removes non-alphabetic characters
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text
```

### Feature Engineering
- Handles missing values with `fillna("")`
- Combines title and text: `content = title + " " + text`
- TF-IDF vectorization with 1000 max features
- Removes English stop words

### Model Training
- 80/20 train-test split with stratification
- Trains all models with identical data
- Comprehensive performance evaluation
- Automatic best model selection

### Visualization
- Model performance comparison charts
- Text length distribution analysis
- Label distribution visualization
- Confusion matrices for all models

## 📁 File Structure

```
fake-news-detection/
│
├── fake_news_detection.py          # Main script
├── README.md                       # This file
├── requirements.txt                # Dependencies
├── data/
│   └── your_dataset.csv           # Your news dataset
├── models/
│   └── best_model_*.pkl           # Saved trained models
└── visualizations/
    └── performance_charts.png      # Generated plots
```

## 🎯 Example Output

```
=== MODEL COMPARISON SUMMARY ===
                    accuracy  precision    recall  f1_score
Naive Bayes           0.8234     0.8156    0.8234    0.8192
Logistic Regression   0.8567     0.8534    0.8567    0.8548
Random Forest         0.8789     0.8745    0.8789    0.8765
Gradient Boosting     0.8812     0.8798    0.8812    0.8803
AdaBoost             0.8723     0.8687    0.8723    0.8703
SGD Classifier       0.8456     0.8423    0.8456    0.8438

Best Model: Gradient Boosting
F1 Score: 0.8803
```

## 🔍 Interactive Prediction Example

```
Options:
1. Enter news article for prediction
2. Exit

Enter your choice (1, 2): 1

Enter news title (optional): Breaking News Alert
Enter news content: Scientists discover new method for renewable energy...

----------------------------------------
PREDICTION RESULTS:
----------------------------------------
Prediction: Real
Confidence: 87.23%
Probability Breakdown:
  Real News: 87.23%
  Fake News: 12.77%
----------------------------------------
```

## ⚠️ Important Notes

- **SGD Classifier**: Uses log_loss for probability support
- **Model Compatibility**: Code handles models with/without probability predictions
- **Error Handling**: Comprehensive error handling for edge cases
- **Memory Usage**: Large datasets may require additional memory optimization

## 📊 Data Insights

The system provides exploratory data analysis including:
- Class distribution (Real vs Fake ratio)
- Text length analysis by category
- Word count distributions
- Statistical summaries and outlier detection

## 🛠️ Customization

### Adding New Models
```python
models = {
    'Your Model': YourClassifier(parameters),
    # Add other models...
}
```

### Adjusting Text Preprocessing
```python
def clean_text(text):
    # Add your custom preprocessing steps
    text = your_custom_function(text)
    return text
```

### Modifying Visualization
```python
# Customize colors, chart types, or add new plots
colors = ['your', 'custom', 'colors']
```

## 📝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scikit-learn for machine learning algorithms
- Pandas for data manipulation
- Matplotlib/Seaborn for visualizations
- The open-source community for inspiration

## 📞 Contact

Simanto Saha - simanto.saha@g.bracu.ac.bd

Project Link: [https://github.com/simanto-saha/Fake-News-Predictor-.git]
---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
