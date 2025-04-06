from typing import List, Dict
import numpy as np
from sqlalchemy.orm import Session
from app.models.learning_resource import LearningResource

class RecommendationService:
    def __init__(self):
        # Initialize skill keywords mapping to our assessment dimensions
        self.skill_keywords = {
            "algorithm_knowledge": ["algorithms", "data structures", "complexity", "problem solving"],
            "coding_proficiency": ["programming", "coding", "software development", "implementation"],
            "system_design": ["system design", "architecture", "scalability", "distributed systems"],
            "debugging": ["debugging", "troubleshooting", "error handling", "testing"],
            "testing_qa": ["testing", "quality assurance", "test automation", "unit testing"],
            "devops": ["devops", "ci/cd", "deployment", "infrastructure", "cloud"],
            "security": ["security", "authentication", "authorization", "encryption"],
            "communication": ["documentation", "technical writing", "collaboration", "teamwork"]
        }
        
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using keyword matching"""
        text = text.lower()
        detected_skills = []
        
        for skill, keywords in self.skill_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected_skills.append(skill)
                
        return detected_skills
        
    def get_recommendations(
        self,
        db: Session,
        weak_skills: List[str],
        limit: int = 5
    ) -> List[Dict]:
        """Get course recommendations based on weak skills"""
        # Get all learning resources
        resources = db.query(LearningResource).all()
        
        # Score each resource based on skill overlap
        scored_resources = []
        for resource in resources:
            # Extract skills from resource description and title
            resource_skills = self.extract_skills(f"{resource.title} {resource.description}")
            
            # Calculate skill overlap score
            skill_overlap = len(set(resource_skills) & set(weak_skills)) / len(weak_skills) if weak_skills else 0
            
            if skill_overlap > 0:  # Only include relevant courses
                scored_resources.append({
                    "resource": resource,
                    "score": skill_overlap
                })
        
        # Sort by score and get top recommendations
        scored_resources.sort(key=lambda x: x["score"], reverse=True)
        top_resources = scored_resources[:limit]
        
        # Convert to response format
        recommendations = []
        for item in top_resources:
            resource = item["resource"]
            recommendations.append({
                **resource.to_dict(),
                "relevance_score": round(item["score"] * 100, 2)
            })
            
        return recommendations
