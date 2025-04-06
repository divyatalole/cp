from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.models.base import get_db
from app.models.assessment import Question as QuestionModel, AssessmentSession, UserResponse
from app.services.recommendation_service import RecommendationService
from app.schemas.recommendations import CourseRecommendation, RecommendationResponse

router = APIRouter()
recommendation_service = RecommendationService()

@router.get("/api/recommendations/{session_id}", response_model=RecommendationResponse)
async def get_recommendations(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get personalized course recommendations based on assessment results"""
    try:
        # Get session and verify completion
        session = db.query(AssessmentSession).filter(
            AssessmentSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        if session.status != "completed":
            raise HTTPException(status_code=400, detail="Assessment not completed")
        
        # Get user's responses and identify weak skills
        responses = db.query(UserResponse).filter(
            UserResponse.session_id == session_id
        ).all()
        
        if not responses:
            raise HTTPException(status_code=404, detail="No responses found")
        
        # Calculate skill scores and identify weak areas
        skill_scores = {}
        for response in responses:
            question = db.query(QuestionModel).filter(
                QuestionModel.id == response.question_id
            ).first()
            
            if question.skill_dimension not in skill_scores:
                skill_scores[question.skill_dimension] = {
                    "correct": 0,
                    "total": 0
                }
            
            skill_scores[question.skill_dimension]["total"] += 1
            if response.is_correct:
                skill_scores[question.skill_dimension]["correct"] += 1
        
        # Identify weak skills (score < 70%)
        weak_skills = [
            skill for skill, scores in skill_scores.items()
            if (scores["correct"] / scores["total"]) < 0.7
        ]
        
        if not weak_skills:
            return RecommendationResponse(
                session_id=session_id,
                weak_skills=[],
                recommendations=[]
            )
        
        # Get recommendations
        recommendations = recommendation_service.get_recommendations(
            db=db,
            weak_skills=weak_skills
        )
        
        return RecommendationResponse(
            session_id=session_id,
            weak_skills=weak_skills,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )
