from typing import List, Dict
from sqlalchemy.orm import Session
from app.models.learning_resource import LearningResource
from app.services.recommendation_service import RecommendationService
from app.data.sample_courses import SAMPLE_COURSES

class CourseIngestionService:
    def __init__(self):
        self.recommendation_service = RecommendationService()
    
    def ingest_courses(self, db: Session):
        """Ingest sample courses into the database"""
        try:
            # First, check if we already have courses
            existing_count = db.query(LearningResource).count()
            if existing_count > 0:
                print(f"Found {existing_count} existing courses, skipping ingestion")
                return
            
            print("Starting course ingestion...")
            
            for course in SAMPLE_COURSES:
                # Extract additional skills from description
                extracted_skills = self.recommendation_service.extract_skills(
                    course["description"]
                )
                
                # Combine predefined and extracted skills
                all_skills = list(set(course["skills"] + extracted_skills))
                
                # Create learning resource
                resource = LearningResource(
                    title=course["title"],
                    description=course["description"],
                    platform=course["platform"],
                    skills=all_skills,
                    url=course["url"],
                    cost=course["cost"],
                    duration_hours=course["duration_hours"]
                )
                
                db.add(resource)
                print(f"Added course: {course['title']}")
            
            db.commit()
            final_count = db.query(LearningResource).count()
            print(f"Course ingestion complete. Total courses: {final_count}")
            
        except Exception as e:
            print(f"Error during course ingestion: {str(e)}")
            db.rollback()
            raise
