from sqlalchemy import Column, Integer, String, Float, JSON
from .base import Base

class LearningResource(Base):
    __tablename__ = "learning_resources"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(String)
    platform = Column(String)
    skills = Column(JSON)  # List of skills as JSON array
    url = Column(String)
    cost = Column(Float)
    duration_hours = Column(Float)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "platform": self.platform,
            "skills": self.skills,
            "url": self.url,
            "cost": self.cost,
            "duration_hours": self.duration_hours
        }
