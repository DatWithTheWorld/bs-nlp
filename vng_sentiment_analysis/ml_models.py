"""
Machine Learning Models for VNG App Reviews Sentiment Analysis
- TF-IDF Feature Extraction
- Naive Bayes, Logistic Regression, SVM, Random Forest
- Cross-Validation (StratifiedKFold k=5)
- Evaluation metrics
"""

import os
import json
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ['Negative', 'Neutral', 'Positive']


def create_tfidf_features(X_train, X_test, max_features=10000):
    """Create TF-IDF features from text."""
    print("\n[TF-IDF] Creating features...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigram + bigram
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"   Feature shape: {X_train_tfidf.shape}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")

    return X_train_tfidf, X_test_tfidf, vectorizer


def get_models():
    """Get dictionary of ML models."""
    return {
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs',
            random_state=42, n_jobs=-1
        ),
        'SVM (Linear)': LinearSVC(
            max_iter=2000, C=1.0, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=None,
            random_state=42, n_jobs=-1
        ),
    }


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model."""
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    # Try ROC AUC (need probability estimates)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_test)
            # Convert to probabilities-like scores
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            y_proba = scaler.fit_transform(y_scores)
        else:
            y_proba = None

        if y_proba is not None:
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
            roc_auc = float(roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro'))
            metrics['roc_auc'] = roc_auc
    except Exception:
        pass

    return {
        'model_name': model_name,
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred.tolist(),
    }


def cross_validate_models(X_train_tfidf, y_train, n_folds=5):
    """Perform cross-validation for all models."""
    print(f"\n[Cross-Validation] StratifiedKFold k={n_folds}")
    print("-" * 60)

    models = get_models()
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_results = {}

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    for name, model in models.items():
        print(f"\n   {name}...")
        start_time = time.time()

        try:
            scores = cross_validate(
                model, X_train_tfidf, y_train,
                cv=cv, scoring=scoring,
                return_train_score=True, n_jobs=-1
            )

            elapsed = time.time() - start_time

            cv_results[name] = {
                'test_accuracy': scores['test_accuracy'].tolist(),
                'test_precision': scores['test_precision_macro'].tolist(),
                'test_recall': scores['test_recall_macro'].tolist(),
                'test_f1': scores['test_f1_macro'].tolist(),
                'train_accuracy': scores['train_accuracy'].tolist(),
                'mean_test_accuracy': float(scores['test_accuracy'].mean()),
                'std_test_accuracy': float(scores['test_accuracy'].std()),
                'mean_test_f1': float(scores['test_f1_macro'].mean()),
                'std_test_f1': float(scores['test_f1_macro'].std()),
                'time_seconds': elapsed,
            }

            print(f"     Accuracy: {scores['test_accuracy'].mean():.4f} (+/- {scores['test_accuracy'].std():.4f})")
            print(f"     F1 Score: {scores['test_f1_macro'].mean():.4f} (+/- {scores['test_f1_macro'].std():.4f})")
            print(f"     Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"     ERROR: {e}")
            cv_results[name] = {'error': str(e)}

    return cv_results


def train_and_evaluate_all(X_train, X_test, y_train, y_test, output_dir):
    """Train all ML models and evaluate."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("MACHINE LEARNING PIPELINE")
    print("=" * 60)

    # Step 1: TF-IDF
    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(X_train, X_test)

    # Save vectorizer
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    # Step 1.5: SMOTE Balancing
    print("\n   [SMOTE] Cân bằng dữ liệu tự động...")
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)
        print("   Đã áp dụng SMOTE thành công!")
    except ImportError:
        print("   [WARNING] Chưa cài đặt `imbalanced-learn`. Bỏ qua SMOTE, sử dụng raw features.")
        X_train_resampled, y_train_resampled = X_train_tfidf, y_train

    # Step 2: Cross-Validation
    cv_results = cross_validate_models(X_train_resampled, y_train_resampled)

    # Step 3: Train and Evaluate on Test Set
    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION ON TEST SET")
    print("=" * 60)

    models = get_models()
    all_results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        start_time = time.time()

        model.fit(X_train_resampled, y_train_resampled)
        train_time = time.time() - start_time

        result = evaluate_model(model, X_test_tfidf, y_test, name)
        result['train_time'] = train_time
        result['cv_results'] = cv_results.get(name, {})

        all_results[name] = result
        trained_models[name] = model

        print(f"   Accuracy:  {result['metrics']['accuracy']:.4f}")
        print(f"   F1 Macro:  {result['metrics']['f1_macro']:.4f}")
        print(f"   Precision: {result['metrics']['precision_macro']:.4f}")
        print(f"   Recall:    {result['metrics']['recall_macro']:.4f}")
        if 'roc_auc' in result['metrics']:
            print(f"   ROC AUC:   {result['metrics']['roc_auc']:.4f}")
        print(f"   Train time: {train_time:.2f}s")

    # Step 4: Get feature importance / top words
    print("\n[Feature Importance] Top words per class...")
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = {}

    # From Logistic Regression coefficients
    lr_model = trained_models.get('Logistic Regression')
    if lr_model is not None and hasattr(lr_model, 'coef_'):
        for i, class_name in enumerate(CLASS_NAMES):
            if i < lr_model.coef_.shape[0]:
                top_indices = lr_model.coef_[i].argsort()[-20:][::-1]
                top_words = [(feature_names[j], float(lr_model.coef_[i][j])) for j in top_indices]
                feature_importance[class_name] = top_words
                print(f"   {class_name}: {', '.join([w for w, _ in top_words[:10]])}")

    # Step 5: Save results
    print("\n[Saving Results]...")

    # Save ML results
    results_save = {}
    for name, result in all_results.items():
        r = dict(result)
        r.pop('predictions', None)  # Don't save full predictions to JSON
        results_save[name] = r

    with open(os.path.join(output_dir, 'ml_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_save, f, ensure_ascii=False, indent=2, default=str)

    # Save CV results
    with open(os.path.join(output_dir, 'cv_results.json'), 'w', encoding='utf-8') as f:
        json.dump(cv_results, f, ensure_ascii=False, indent=2, default=str)

    # Save feature importance
    with open(os.path.join(output_dir, 'feature_importance.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_importance, f, ensure_ascii=False, indent=2)

    # Save trained models
    for name, model in trained_models.items():
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        with open(os.path.join(output_dir, f'model_{safe_name}.pkl'), 'wb') as f:
            pickle.dump(model, f)

    # Print summary table
    print("\n" + "=" * 60)
    print("ML RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    for name, result in all_results.items():
        m = result['metrics']
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
              f"{m['precision_macro']:>10.4f} {m['recall_macro']:>10.4f}")

    return all_results, cv_results, trained_models, vectorizer, feature_importance


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE, 'output')

    data = np.load(os.path.join(OUTPUT_DIR, 'train_test_split.npz'), allow_pickle=True)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    train_and_evaluate_all(X_train, X_test, y_train, y_test, OUTPUT_DIR)
