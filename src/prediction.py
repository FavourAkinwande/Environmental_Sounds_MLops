def evaluate_model(model, X_test, y_test_cat, le, print_samples=5):
    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred_labels = y_pred_probs.argmax(axis=1)
    y_true_labels = y_test_cat.argmax(axis=1)

    # Convert to string labels
    y_pred_str_labels = le.inverse_transform(y_pred_labels)
    y_true_str_labels = le.inverse_transform(y_true_labels)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))

    # Individual Metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # : Show a few prediction samples
    print(f"\nSample predictions (first {print_samples}):")
    for i in range(min(print_samples, len(y_true_labels))):
        print(f"True: {y_true_str_labels[i]} | Predicted: {y_pred_str_labels[i]}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }, y_true_labels, y_pred_labels
    
metrics, y_true_labels, y_pred_labels = evaluate_model(model, X_test, y_test_cat, le)

from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


# Making Single sample prediction
single_sample = X_test[0].reshape(1, -1)
predicted_label = predict_single_sample(model, single_sample, le)
