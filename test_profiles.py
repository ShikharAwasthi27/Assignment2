"""
Test profiles for evaluating the Course Recommendation Engine
"""

from main import CourseRecommendationEngine
import pandas as pd

# Test profiles as specified in the requirements
TEST_PROFILES = [
    {
        "name": "Data Visualization Enthusiast",
        "profile": "I've completed the 'Python Programming for Data Science' course and enjoy data visualization. What should I take next?",
        "completed_ids": ["101"]
    },
    {
        "name": "Azure & DevOps Learner", 
        "profile": "I know Azure basics and want to manage containers and build CI/CD pipelines. Recommend courses.",
        "completed_ids": ["205", "301"]
    },
    {
        "name": "ML to Neural Networks",
        "profile": "My background is in ML fundamentals; I'd like to specialize in neural networks and production workflows.",
        "completed_ids": ["401", "402"]
    },
    {
        "name": "Microservices & Kubernetes",
        "profile": "I want to learn to build and deploy microservices with Kubernetes—what courses fit best?",
        "completed_ids": ["302"]
    },
    {
        "name": "Blockchain Beginner",
        "profile": "I'm interested in blockchain and smart contracts but have no prior experience. Which courses do you suggest?",
        "completed_ids": []
    }
]

def evaluate_recommendations():
    """Evaluate the recommendation engine with test profiles."""
    
    # Initialize engine
    engine = CourseRecommendationEngine()
    
    try:
        # Try to load existing model
        engine.load_model('course_model.pkl')
        print("Loaded existing model")
    except FileNotFoundError:
        print("Training new model...")
        engine.load_data('assignment2data.csv')
        engine.create_embeddings()
        engine.build_index()
        engine.save_model('course_model.pkl')
    
    results = []
    
    print("=" * 80)
    print("COURSE RECOMMENDATION ENGINE EVALUATION")
    print("=" * 80)
    
    for i, test_case in enumerate(TEST_PROFILES, 1):
        print(f"\nTest Profile {i}: {test_case['name']}")
        print("-" * 60)
        print(f"Query: {test_case['profile']}")
        print(f"Completed Course IDs: {test_case['completed_ids']}")
        
        # Get recommendations
        recommendations = engine.recommend_courses(
            profile=test_case['profile'],
            completed_ids=test_case['completed_ids'],
            top_k=5
        )
        
        print("\nRecommendations:")
        test_results = []
        
        for j, (course_id, score) in enumerate(recommendations, 1):
            course_info = engine.get_course_info(course_id)
            print(f"{j}. Course ID: {course_id} (Score: {score:.3f})")
            print(f"   Title: {course_info['title']}")
            print(f"   Description: {course_info['description'][:120]}...")
            
            test_results.append({
                'rank': j,
                'course_id': course_id,
                'score': score,
                'title': course_info['title']
            })
        
        results.append({
            'profile_name': test_case['name'],
            'query': test_case['profile'],
            'completed_ids': test_case['completed_ids'],
            'recommendations': test_results
        })
        
        print("\n" + "=" * 80)
    
    return results

def save_results_to_csv(results, filename='evaluation_results.csv'):
    """Save evaluation results to CSV for analysis."""
    rows = []
    
    for result in results:
        for rec in result['recommendations']:
            rows.append({
                'profile_name': result['profile_name'],
                'query': result['query'],
                'completed_ids': str(result['completed_ids']),
                'rank': rec['rank'],
                'course_id': rec['course_id'],
                'score': rec['score'],
                'title': rec['title']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    results = evaluate_recommendations()
    save_results_to_csv(results)
