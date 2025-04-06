import requests
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def map_skills(course_name: str, description: str) -> List[str]:
    """Map course content to relevant skills."""
    skills = set()
    
    # Basic skill mapping based on keywords
    skill_keywords = {
        'python': ['python', 'programming', 'coding'],
        'javascript': ['javascript', 'js', 'web development'],
        'data_science': ['data science', 'machine learning', 'ai', 'statistics'],
        'cloud_computing': ['aws', 'azure', 'cloud', 'devops'],
        'web_development': ['web', 'html', 'css', 'frontend'],
        'database': ['sql', 'database', 'mysql', 'postgresql'],
        'system_design': ['system design', 'architecture', 'scalability'],
        'algorithms': ['algorithm', 'data structure', 'leetcode'],
    }
    
    content = (course_name + ' ' + description).lower()
    
    for skill, keywords in skill_keywords.items():
        if any(keyword in content for keyword in keywords):
            skills.add(skill)
    
    # Ensure at least one skill is mapped
    if not skills:
        skills.add('general_programming')
    
    return list(skills)

def estimate_duration(description: str) -> float:
    """Estimate course duration in hours based on content."""
    # Basic estimation logic
    words = len(description.split())
    if words > 1000:
        return 40.0  # Comprehensive course
    elif words > 500:
        return 20.0  # Medium length course
    else:
        return 10.0  # Short course

def fetch_coursera_courses() -> List[Dict[str, Any]]:
    """Fetch courses from Coursera API and structure them for our system."""
    url = "https://api.coursera.org/api/courses.v1"
    params = {
        "limit": 50,
        "includes": "partnerIds,description",
        "fields": "name,description,slug,partnerIds",
        "q": "search",
        "query": "computer science programming"  # Focus on relevant courses
    }
    
    try:
        logger.info("Fetching courses from Coursera API...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        courses = response.json().get("elements", [])
        logger.info(f"Retrieved {len(courses)} courses from Coursera")
        
        result = []
        for course in courses:
            title = course.get("name", "").strip()
            description = course.get("description", "").strip()
            
            if not title or not description:
                continue
                
            course_data = {
                "title": title,
                "description": description,
                "platform": "Coursera",
                "skills": map_skills(title, description),
                "url": f"https://www.coursera.org/learn/{course['slug']}",
                "cost": 49.99,  # Default price for Coursera courses
                "duration_hours": estimate_duration(description),
                "relevance_score": 0  # Will be calculated during recommendation
            }
            result.append(course_data)
            
        return result
        
    except requests.RequestException as e:
        logger.error(f"Error fetching Coursera courses: {e}")
        return []

def save_courses(courses: List[Dict[str, Any]], output_file: str = "real_courses.json") -> None:
    """Save fetched courses to a JSON file."""
    output_path = Path(__file__).parent / output_file
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(courses, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(courses)} courses to {output_path}")
    except IOError as e:
        logger.error(f"Error saving courses to file: {e}")

if __name__ == "__main__":
    courses = fetch_coursera_courses()
    if courses:
        save_courses(courses)
    else:
        logger.error("No courses were fetched. Check the API response or connection.")
