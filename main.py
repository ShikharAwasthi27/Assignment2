import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Tuple
import json

class CourseRecommendationEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the recommendation engine with sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.courses_df = None
        self.index = None
        self.embeddings = None
        
    def load_data(self, csv_path: str):
        """Load course data from CSV file."""
        self.courses_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.courses_df)} courses")
        
    def create_embeddings(self):
        """Create embeddings for all course descriptions."""
        if self.courses_df is None:
            raise ValueError("Please load data first using load_data()")
            
        # Combine title and description for richer embeddings
        course_texts = (self.courses_df['title'].fillna('') + ' ' + 
                       self.courses_df['description'].fillna('')).tolist()
        
        print("Creating embeddings...")
        self.embeddings = self.model.encode(course_texts)
        print(f"Created embeddings with shape: {self.embeddings.shape}")
        
    def build_index(self):
        """Build FAISS index for fast similarity search."""
        if self.embeddings is None:
            raise ValueError("Please create embeddings first using create_embeddings()")
            
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings.astype(np.float32))
        self.index.add(self.embeddings.astype(np.float32))
        
        print("FAISS index built successfully")
        
    def recommend_courses(self, profile: str, completed_ids: List[str] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend courses based on user profile.
        
        Args:
            profile: User's interests and background description
            completed_ids: List of completed course IDs to exclude
            top_k: Number of recommendations to return
            
        Returns:
            List of (course_id, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Please build index first using build_index()")
            
        # Create embedding for user profile
        profile_embedding = self.model.encode([profile])
        faiss.normalize_L2(profile_embedding.astype(np.float32))
        
        # Search for similar courses
        scores, indices = self.index.search(profile_embedding.astype(np.float32), 
                                          min(top_k + len(completed_ids or []), len(self.courses_df)))
        
        recommendations = []
        completed_set = set(completed_ids or [])
        
        for score, idx in zip(scores[0], indices[0]):
            course_id = str(self.courses_df.iloc[idx]['course_id'])
            
            # Skip completed courses
            if course_id not in completed_set:
                recommendations.append((course_id, float(score)))
                
            if len(recommendations) >= top_k:
                break
                
        return recommendations
    
    def get_course_info(self, course_id: str) -> dict:
        """Get course information by ID."""
        course = self.courses_df[self.courses_df['course_id'] == int(course_id)]
        if len(course) == 0:
            return None
        
        course = course.iloc[0]
        return {
            'course_id': course['course_id'],
            'title': course['title'],
            'description': course['description']
        }
    
    def save_model(self, path: str):
        """Save the trained model and index."""
        model_data = {
            'courses_df': self.courses_df,
            'embeddings': self.embeddings,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save FAISS index separately
        faiss.write_index(self.index, path.replace('.pkl', '.faiss'))
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a pre-trained model and index."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.courses_df = model_data['courses_df']
        self.embeddings = model_data['embeddings']
        
        # Load FAISS index
        self.index = faiss.read_index(path.replace('.pkl', '.faiss'))
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    engine = CourseRecommendationEngine()
    
    # Load and process data
    engine.load_data('assignment2data.csv')
    engine.create_embeddings()
    engine.build_index()
    
    # Save the model
    engine.save_model('course_model.pkl')
    
    # Test recommendation
    profile = "I've completed Python Programming for Data Science and enjoy data visualization. What should I take next?"
    recommendations = engine.recommend_courses(profile, completed_ids=['101'], top_k=5)
    
    print("\nRecommendations:")
    for course_id, score in recommendations:
        course_info = engine.get_course_info(course_id)
        print(f"Course ID: {course_id}, Score: {score:.3f}")
        print(f"Title: {course_info['title']}")
        print(f"Description: {course_info['description'][:100]}...")
        print("-" * 50)
