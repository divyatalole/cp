import os
import sys
sys.path.append(os.path.dirname(__file__))

from app.models.base import Base
from app.models.learning_resource import LearningResource
from app.data.sample_courses import SAMPLE_COURSES
from app.services.course_ingestion_service import CourseIngestionService
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./app.db"  # Using the same DB as our app

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

Base.metadata.create_all(bind=engine)

print("Starting course seeding...")
ingestion_service = CourseIngestionService(model_name="all-MiniLM-L6-v2")

# Delete existing courses
session.query(LearningResource).delete()
session.commit()
print("Cleared existing courses")

for course in SAMPLE_COURSES:
    try:
        print(f"Processing course: {course['title']}")
        course_with_embedding = ingestion_service.process_course(course)
        db_course = LearningResource(**course_with_embedding)
        session.merge(db_course)
        print(f"✅ Added course: {course['title']}")
    except Exception as e:
        print(f"❌ Error processing course {course['title']}: {str(e)}")

session.commit()
print("\n✅ Courses seeded successfully!")

# Print summary
course_count = session.query(LearningResource).count()
print(f"Total courses in database: {course_count}")

session.close()
