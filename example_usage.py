"""
Example usage of the Fake News Detection system
Demonstrates various ways to use the model
"""

from train_model import FakeNewsModelTrainer
import json

def example_1_basic_prediction():
    """Example 1: Basic prediction"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Prediction")
    print("="*70)
    
    trainer = FakeNewsModelTrainer()
    trainer.load_model()
    
    news_samples = [
        "New study shows that chocolate can improve your health and IQ",
        "Government secretly implanting microchips through vaccines",
        "Stock market reaches new high as economic growth continues",
        "Scientists discover that water has memory and can be programmed",
    ]
    
    for news in news_samples:
        result = trainer.predict(news)
        print(f"\nNews: {news[:60]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")


def example_2_batch_processing():
    """Example 2: Batch processing multiple news items"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Processing")
    print("="*70)
    
    trainer = FakeNewsModelTrainer()
    trainer.load_model()
    
    news_articles = [
        {"title": "Breaking News 1", "content": "Scientists discover cure for cancer"},
        {"title": "Breaking News 2", "content": "Aliens landed in the White House"},
        {"title": "Breaking News 3", "content": "Central bank raises interest rates"},
    ]
    
    results = []
    for article in news_articles:
        result = trainer.predict(article['content'])
        results.append({
            'title': article['title'],
            'prediction': result['prediction'],
            'confidence': result['confidence']
        })
        print(f"\n{article['title']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2f}")
    
    # Count fake vs real
    fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
    real_count = sum(1 for r in results if r['prediction'] == 'REAL')
    
    print(f"\n\nSummary:")
    print(f"  Fake News: {fake_count}")
    print(f"  Real News: {real_count}")


def example_3_confidence_based_filtering():
    """Example 3: Filter results by confidence threshold"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Confidence-Based Filtering")
    print("="*70)
    
    trainer = FakeNewsModelTrainer()
    trainer.load_model()
    
    news_list = [
        "New climate report shows rising temperatures",
        "Shocking: Elvis spotted alive in grocery store",
        "Tech company announces new product launch",
        "Government covers up UFO encounters",
    ]
    
    confidence_threshold = 0.7
    
    print(f"Filtering results with confidence > {confidence_threshold}")
    
    high_confidence = []
    low_confidence = []
    
    for news in news_list:
        result = trainer.predict(news)
        if result['confidence'] >= confidence_threshold:
            high_confidence.append((news, result))
        else:
            low_confidence.append((news, result))
    
    print(f"\n\nHIGH CONFIDENCE PREDICTIONS ({len(high_confidence)}):")
    for news, result in high_confidence:
        print(f"\n  News: {news[:50]}...")
        print(f"  Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
    
    print(f"\n\nLOW CONFIDENCE PREDICTIONS ({len(low_confidence)}):")
    for news, result in low_confidence:
        print(f"\n  News: {news[:50]}...")
        print(f"  Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")


def example_4_export_results_json():
    """Example 4: Export results to JSON"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Export Results to JSON")
    print("="*70)
    
    trainer = FakeNewsModelTrainer()
    trainer.load_model()
    
    news_list = [
        "Scientists announce major breakthrough",
        "Celebrity caught in shocking scandal",
        "New economic policy introduced by government",
    ]
    
    results = []
    for news in news_list:
        result = trainer.predict(news)
        results.append({
            'news': news,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'label': result['label']
        })
    
    # Export to JSON
    json_output = json.dumps(results, indent=2)
    print("\nJSON Output:")
    print(json_output)
    
    # Save to file
    with open('predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to 'predictions.json'")


def example_5_accuracy_testing():
    """Example 5: Test model accuracy with known samples"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Model Accuracy Testing")
    print("="*70)
    
    trainer = FakeNewsModelTrainer()
    trainer.load_model()
    
    # Test samples with known labels
    test_samples = [
        ("Climate change is causing rising sea levels", 0, "Real"),
        ("The moon landing was faked by NASA", 1, "Fake"),
        ("Economic indicators show growth", 0, "Real"),
        ("5G towers cause COVID-19", 1, "Fake"),
    ]
    
    correct = 0
    incorrect = 0
    
    print("\nTesting predictions on known samples:")
    for text, expected_label, label_name in test_samples:
        result = trainer.predict(text)
        is_correct = result['label'] == expected_label
        
        if is_correct:
            correct += 1
            status = "✓ CORRECT"
        else:
            incorrect += 1
            status = "✗ INCORRECT"
        
        print(f"\n{status}")
        print(f"  Text: {text[:50]}...")
        print(f"  Expected: {label_name}")
        print(f"  Predicted: {result['prediction']} (Confidence: {result['confidence']:.2f})")
    
    accuracy = (correct / len(test_samples)) * 100
    print(f"\n\nAccuracy on test samples: {accuracy:.1f}% ({correct}/{len(test_samples)})")


def example_6_real_world_scenario():
    """Example 6: Real-world scenario - News feed analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Real-World Scenario - News Feed Analysis")
    print("="*70)
    
    trainer = FakeNewsModelTrainer()
    trainer.load_model()
    
    # Simulating a news feed
    news_feed = [
        {"source": "Tech News", "title": "New smartphone released", "content": "Latest smartphone features AI chip"},
        {"source": "Health News", "title": "Miracle cure found", "content": "One weird trick doctors hate"},
        {"source": "Business News", "title": "Quarterly earnings up", "content": "Company reports 25% revenue growth"},
        {"source": "Conspiracy", "title": "Government cover-up", "content": "Secret alien base discovered"},
    ]
    
    print("\nAnalyzing news feed:\n")
    
    fake_news_count = 0
    real_news_count = 0
    
    for article in news_feed:
        result = trainer.predict(article['content'])
        
        if result['label'] == 1:
            fake_news_count += 1
            emoji = "⚠️"
        else:
            real_news_count += 1
            emoji = "✓"
        
        print(f"{emoji} {article['source']}: {article['title']}")
        print(f"   Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
    
    print(f"\n\nFeed Analysis Summary:")
    print(f"  Real News: {real_news_count}")
    print(f"  Fake News: {fake_news_count}")
    print(f"  Total Articles: {len(news_feed)}")
    print(f"  Fake News Percentage: {(fake_news_count/len(news_feed)*100):.1f}%")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("FAKE NEWS DETECTION - EXAMPLE USAGE")
    print("="*70)
    
    try:
        # Run examples
        example_1_basic_prediction()
        example_2_batch_processing()
        example_3_confidence_based_filtering()
        example_4_export_results_json()
        example_5_accuracy_testing()
        example_6_real_world_scenario()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")
        
    except FileNotFoundError:
        print("\nError: Model files not found!")
        print("Please run 'python train_model.py' first to train the model.")


if __name__ == "__main__":
    main()
