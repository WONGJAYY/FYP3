"""
Evaluator Module for E-Commerce Recommender System
Provides evaluation metrics including RMSE for recommendation quality assessment.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import os


class RecommenderEvaluator:
    """
    Evaluator for the content-based recommender system.
    Provides RMSE and other evaluation metrics.
    """
    
    def __init__(self, recommender, data_dir='processed_data'):
        """
        Initialize the evaluator.
        
        Args:
            recommender: ContentBasedRecommender instance
            data_dir: Directory containing processed data
        """
        self.recommender = recommender
        self.data_dir = data_dir
        self.products = None
        self._load_data()
    
    def _load_data(self):
        """Load product data with ratings."""
        with open(os.path.join(self.data_dir, 'products.pkl'), 'rb') as f:
            self.products = pickle.load(f)
    
    def calculate_rmse(self, actual_ratings, predicted_ratings):
        """
        Calculate Root Mean Square Error between actual and predicted ratings.
        
        RMSE = sqrt(1/n * Σ(actual - predicted)²)
        
        Args:
            actual_ratings: Array of actual ratings
            predicted_ratings: Array of predicted ratings
            
        Returns:
            RMSE value
        """
        return np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    
    def predict_rating_from_similar(self, product_id, n_neighbors=5):
        """
        Predict rating for a product based on similar products' ratings.
        Uses weighted average of similar products' ratings.
        
        Args:
            product_id: ID of the product
            n_neighbors: Number of similar products to consider
            
        Returns:
            Predicted rating (weighted average)
        """
        recommendations = self.recommender.recommend(product_id, n_recommendations=n_neighbors)
        
        if not recommendations:
            return None
        
        # Weighted average: rating * similarity_score
        total_weight = 0
        weighted_sum = 0
        
        for rec in recommendations:
            rating = rec['product'].get('avg_rating', 0)
            similarity = rec['similarity_score']
            
            if rating > 0 and similarity > 0:
                weighted_sum += rating * similarity
                total_weight += similarity
        
        if total_weight == 0:
            return None
            
        return weighted_sum / total_weight
    
    def evaluate_rating_prediction(self, test_size=0.2, n_neighbors=5):
        """
        Evaluate rating prediction performance using RMSE.
        
        Args:
            test_size: Fraction of products to use for testing
            n_neighbors: Number of neighbors for prediction
            
        Returns:
            Dictionary with evaluation results
        """
        # Get products with ratings
        products_with_ratings = self.products[self.products['avg_rating'] > 0].copy()
        
        if len(products_with_ratings) < 10:
            return {'error': 'Not enough products with ratings for evaluation'}
        
        # Split into train/test
        train_products, test_products = train_test_split(
            products_with_ratings, 
            test_size=test_size, 
            random_state=42
        )
        
        actual_ratings = []
        predicted_ratings = []
        
        for _, row in test_products.iterrows():
            product_id = row['id']
            actual_rating = row['avg_rating']
            
            predicted_rating = self.predict_rating_from_similar(product_id, n_neighbors)
            
            if predicted_rating is not None:
                actual_ratings.append(actual_rating)
                predicted_ratings.append(predicted_rating)
        
        if len(actual_ratings) == 0:
            return {'error': 'Could not generate predictions'}
        
        # Calculate metrics
        rmse = self.calculate_rmse(actual_ratings, predicted_ratings)
        mae = np.mean(np.abs(np.array(actual_ratings) - np.array(predicted_ratings)))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(actual_ratings),
            'n_test_products': len(test_products),
            'actual_ratings': actual_ratings,
            'predicted_ratings': predicted_ratings
        }
    
    def evaluate_similarity_quality(self, n_samples=50, n_recommendations=5):
        """
        Evaluate recommendation quality based on category/brand consistency.
        
        Args:
            n_samples: Number of products to sample
            n_recommendations: Number of recommendations per product
            
        Returns:
            Dictionary with evaluation metrics
        """
        products = self.recommender.get_all_products()
        
        if len(products) < n_samples:
            n_samples = len(products)
        
        # Sample products
        sample_indices = np.random.choice(len(products), size=n_samples, replace=False)
        
        category_matches = []
        brand_matches = []
        avg_similarities = []
        
        for idx in sample_indices:
            product = products[idx]
            recommendations = self.recommender.recommend(product['id'], n_recommendations)
            
            if not recommendations:
                continue
            
            source_brand = product.get('brand', '').lower()
            
            brand_match_count = 0
            similarities = []
            
            for rec in recommendations:
                rec_brand = rec['product'].get('brand', '').lower()
                if source_brand and rec_brand and source_brand == rec_brand:
                    brand_match_count += 1
                similarities.append(rec['similarity_score'])
            
            brand_matches.append(brand_match_count / len(recommendations))
            avg_similarities.append(np.mean(similarities))
        
        return {
            'avg_brand_consistency': np.mean(brand_matches) if brand_matches else 0,
            'avg_similarity_score': np.mean(avg_similarities) if avg_similarities else 0,
            'n_samples': n_samples,
            'n_recommendations': n_recommendations
        }
    
    def get_full_evaluation_report(self):
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print("Running RMSE evaluation...")
        rating_eval = self.evaluate_rating_prediction()
        
        print("Running similarity quality evaluation...")
        similarity_eval = self.evaluate_similarity_quality()
        
        return {
            'rating_prediction': rating_eval,
            'similarity_quality': similarity_eval
        }


def evaluate_recommender(data_dir='processed_data'):
    """
    Helper function to evaluate the recommender system.
    
    Returns:
        Evaluation results dictionary
    """
    from recommender import load_recommender
    
    recommender = load_recommender(data_dir)
    evaluator = RecommenderEvaluator(recommender, data_dir)
    
    return evaluator.get_full_evaluation_report()


if __name__ == '__main__':
    # Run evaluation
    results = evaluate_recommender()
    
    print("\n" + "="*50)
    print("RECOMMENDER SYSTEM EVALUATION RESULTS")
    print("="*50)
    
    if 'error' not in results['rating_prediction']:
        print(f"\n[Rating Prediction Metrics]")
        print(f"   RMSE: {results['rating_prediction']['rmse']:.4f}")
        print(f"   MAE:  {results['rating_prediction']['mae']:.4f}")
        print(f"   Predictions made: {results['rating_prediction']['n_predictions']}")
    else:
        print(f"\nRating Prediction: {results['rating_prediction']['error']}")
    
    print(f"\n[Similarity Quality Metrics]")
    print(f"   Brand Consistency: {results['similarity_quality']['avg_brand_consistency']:.2%}")
    print(f"   Avg Similarity Score: {results['similarity_quality']['avg_similarity_score']:.4f}")
