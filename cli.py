from main import CourseRecommendationEngine

def run_test():
    engine = CourseRecommendationEngine()
    engine.load_data('assignment2data.csv')
    engine.create_embeddings()
    engine.build_index()
    
    # Test recommendation
    profile = "I've completed Python and enjoy data visualization"
    recs = engine.recommend_courses(profile, completed_ids=['101'])
    
    for course_id, score in recs:
        info = engine.get_course_info(course_id)
        print(f"{course_id}: {info['title']} (Score: {score:.3f})")

if __name__ == "__main__":
    run_test()
