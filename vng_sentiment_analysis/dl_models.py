"""
Deep Learning Models for VNG App Reviews Sentiment Analysis
- LSTM (Bidirectional) and CNN (Conv1D) models
- Keras/TensorFlow based
- Cross-validation support
"""

import sys
sys.path.insert(0, r'C:\tf_pkg')

import os
import json
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

CLASS_NAMES = ['Negative', 'Neutral', 'Positive']


def check_tensorflow():
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf
        print(f"   TensorFlow version: {tf.__version__}")
        return True
    except ImportError:
        print("   [WARNING] TensorFlow not available. Skipping DL models.")
        return False


def prepare_sequences(X_train, X_test, max_words=15000, max_len=100):
    """Tokenize and pad text sequences for DL models."""
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    print("\n[Sequence Preparation]")

    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    print(f"   Vocab size: {vocab_size}")
    print(f"   Max sequence length: {max_len}")
    print(f"   Train shape: {X_train_pad.shape}")
    print(f"   Test shape: {X_test_pad.shape}")

    return X_train_pad, X_test_pad, tokenizer, vocab_size


def build_lstm_model(vocab_size, max_len, num_classes=3, embedding_dim=128):
    """Build Bidirectional LSTM model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, Bidirectional, LSTM, Dense, Dropout,
        SpatialDropout1D, GlobalMaxPooling1D
    )

    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_cnn_model(vocab_size, max_len, num_classes=3, embedding_dim=128):
    """Build CNN (Conv1D) model for text classification."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout,
        SpatialDropout1D, BatchNormalization
    )

    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        SpatialDropout1D(0.3),
        Conv1D(128, 5, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, 3, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_dl_model(model, X_train, y_train, X_val, y_val, model_name,
                   epochs=20, batch_size=64):
    """Train a DL model and return history."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print(f"\n--- Training {model_name} ---")
    print(f"   Epochs: {epochs}, Batch size: {batch_size}")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def evaluate_dl_model(model, X_test, y_test, model_name):
    """Evaluate a DL model."""
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    print(f"\n   {model_name} Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"   Precision: {metrics['precision_macro']:.4f}")
    print(f"   Recall:    {metrics['recall_macro']:.4f}")

    return {
        'model_name': model_name,
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred.tolist(),
    }


def cross_validate_dl(X_train_text, y_train, build_fn, model_name,
                      max_words=15000, max_len=100, n_folds=5,
                      epochs=15, batch_size=64):
    """Cross-validate a DL model."""
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    print(f"\n[DL Cross-Validation] {model_name} (k={n_folds})")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = {'accuracy': [], 'f1_macro': []}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_text, y_train)):
        print(f"   Fold {fold+1}/{n_folds}...")

        X_fold_train = X_train_text[train_idx]
        X_fold_val = X_train_text[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]

        # Tokenize
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_fold_train)

        X_ft = pad_sequences(tokenizer.texts_to_sequences(X_fold_train),
                             maxlen=max_len, padding='post')
        X_fv = pad_sequences(tokenizer.texts_to_sequences(X_fold_val),
                             maxlen=max_len, padding='post')

        vocab_size = min(max_words, len(tokenizer.word_index) + 1)

        # Build and train
        model = build_fn(vocab_size, max_len)
        model.fit(X_ft, y_fold_train, validation_data=(X_fv, y_fold_val),
                  epochs=epochs, batch_size=batch_size, verbose=0,
                  callbacks=[__import__('tensorflow').keras.callbacks.EarlyStopping(
                      monitor='val_loss', patience=3, restore_best_weights=True)])

        # Evaluate
        y_pred = np.argmax(model.predict(X_fv, verbose=0), axis=1)
        acc = accuracy_score(y_fold_val, y_pred)
        f1 = f1_score(y_fold_val, y_pred, average='macro', zero_division=0)
        fold_scores['accuracy'].append(float(acc))
        fold_scores['f1_macro'].append(float(f1))
        print(f"     Accuracy: {acc:.4f}, F1: {f1:.4f}")

        # Clean up memory
        del model
        import gc
        gc.collect()

    fold_scores['mean_accuracy'] = float(np.mean(fold_scores['accuracy']))
    fold_scores['std_accuracy'] = float(np.std(fold_scores['accuracy']))
    fold_scores['mean_f1'] = float(np.mean(fold_scores['f1_macro']))
    fold_scores['std_f1'] = float(np.std(fold_scores['f1_macro']))

    print(f"   CV Accuracy: {fold_scores['mean_accuracy']:.4f} (+/- {fold_scores['std_accuracy']:.4f})")
    print(f"   CV F1:       {fold_scores['mean_f1']:.4f} (+/- {fold_scores['std_f1']:.4f})")

    return fold_scores


def train_and_evaluate_dl(X_train, X_test, y_train, y_test, output_dir,
                          max_words=15000, max_len=100, epochs=20, batch_size=64):
    """Train and evaluate all DL models."""
    os.makedirs(output_dir, exist_ok=True)

    if not check_tensorflow():
        print("TensorFlow not available. Skipping DL pipeline.")
        return {}, {}

    print("\n" + "=" * 60)
    print("DEEP LEARNING PIPELINE")
    print("=" * 60)

    # Step 1: Prepare sequences
    X_train_pad, X_test_pad, tokenizer, vocab_size = prepare_sequences(
        X_train, X_test, max_words, max_len
    )

    # Save tokenizer
    with open(os.path.join(output_dir, 'dl_tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

    # Step 2: Split train into train/val for training
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_pad, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # Step 3: Train models
    dl_models = {
        'BiLSTM': (build_lstm_model, {}),
        'CNN': (build_cnn_model, {}),
    }

    all_results = {}
    all_histories = {}
    dl_cv_results = {}

    for name, (build_fn, kwargs) in dl_models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        # Build model
        model = build_fn(vocab_size, max_len, **kwargs)
        model.summary()

        # Train
        history = train_dl_model(model, X_tr, y_tr, X_val, y_val,
                                 name, epochs=epochs, batch_size=batch_size)

        # Evaluate on test set
        result = evaluate_dl_model(model, X_test_pad, y_test, name)
        all_results[name] = result

        # Save history
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        all_histories[name] = hist

        # Save model
        safe_name = name.lower().replace(' ', '_')
        model.save(os.path.join(output_dir, f'dl_model_{safe_name}.keras'))

        # Cross-validation
        print(f"\n[Running CV for {name}...]")
        cv_result = cross_validate_dl(
            X_train, y_train, build_fn, name,
            max_words=max_words, max_len=max_len,
            n_folds=5, epochs=15, batch_size=batch_size
        )
        dl_cv_results[name] = cv_result

    # Step 4: Save results
    print("\n[Saving DL Results]...")

    results_save = {}
    for name, result in all_results.items():
        r = dict(result)
        r.pop('predictions', None)
        results_save[name] = r

    with open(os.path.join(output_dir, 'dl_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_save, f, ensure_ascii=False, indent=2, default=str)

    with open(os.path.join(output_dir, 'dl_histories.json'), 'w', encoding='utf-8') as f:
        json.dump(all_histories, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'dl_cv_results.json'), 'w', encoding='utf-8') as f:
        json.dump(dl_cv_results, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("DL RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 60)
    for name, result in all_results.items():
        m = result['metrics']
        print(f"{name:<20} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
              f"{m['precision_macro']:>10.4f} {m['recall_macro']:>10.4f}")

    return all_results, all_histories, dl_cv_results


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE, 'output')

    data = np.load(os.path.join(OUTPUT_DIR, 'train_test_split.npz'), allow_pickle=True)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    train_and_evaluate_dl(X_train, X_test, y_train, y_test, OUTPUT_DIR)
