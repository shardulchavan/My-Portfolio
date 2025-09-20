# chat_query.py
import google.generativeai as genai
from typing import List, Dict
from resume_data import RESUME_DATA
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat_query.log'),
        logging.StreamHandler()
    ]
)

class ChatQueryProcessor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                self.logger.info("Gemini API configured successfully")
            except Exception as e:
                self.logger.error(f"Failed to configure Gemini API: {str(e)}")
                self.model = None
        else:
            self.logger.warning("No API key provided - running in fallback mode")
            self.model = None
    
    def classify_query(self, message: str) -> Dict[str, any]:
        """Classify query type and extract key entities"""
        try:
            message_lower = message.lower()
            
            classification = {
                'type': 'general',
                'specific_company': None,
                'specific_technology': None,
                'intent': None
            }
            
            # Define query patterns
            patterns = {
                'experience': {
                    'keywords': ['experience', 'work', 'worked', 'role', 'position', 'job', 'employment'],
                    'intent': 'discussing_work_experience'
                },
                'company_specific': {
                    'keywords': ['at', 'with', 'for', 'in'],  # Usually followed by company name
                    'intent': 'asking_about_specific_company'
                },
                'skills': {
                    'keywords': ['skills', 'technologies', 'tech stack', 'proficient', 'expertise', 'knowledge', 'familiar'],
                    'intent': 'assessing_technical_skills'
                },
                'projects': {
                    'keywords': ['project', 'built', 'developed', 'created', 'implemented', 'designed', 'github'],
                    'intent': 'exploring_technical_projects'
                },
                'education': {
                    'keywords': ['education', 'degree', 'university', 'study', 'gpa', 'masters', 'bachelors'],
                    'intent': 'verifying_education'
                },
                'contact': {
                    'keywords': ['contact', 'email', 'phone', 'reach', 'connect', 'linkedin'],
                    'intent': 'requesting_contact_info'
                },
                'general_intro': {
                    'keywords': ['tell me about', 'who are you', 'introduce', 'yourself', 'background'],
                    'intent': 'general_introduction'
                }
            }
            
            # Check for each pattern
            max_score = 0
            for query_type, pattern in patterns.items():
                score = sum(1 for keyword in pattern['keywords'] if keyword in message_lower)
                if score > max_score:
                    max_score = score
                    classification['type'] = query_type
                    classification['intent'] = pattern['intent']
            
            # Check for specific company mentions
            for exp in RESUME_DATA['experiences']:
                company_lower = exp['company'].lower()
                if company_lower in message_lower or any(word in message_lower for word in company_lower.split()):
                    classification['specific_company'] = exp['company']
                    classification['type'] = 'company_specific'
                    break
            
            # Check for specific technology mentions
            all_techs = set()
            for exp in RESUME_DATA['experiences']:
                all_techs.update([tech.lower() for tech in exp['technologies']])
            for proj in RESUME_DATA['projects']:
                all_techs.update([tech.lower() for tech in proj['technologies']])
            
            for tech in all_techs:
                if tech in message_lower:
                    classification['specific_technology'] = tech
                    if classification['type'] == 'general':
                        classification['type'] = 'skills'
                    break
            
            # Additional classification for follow-up questions
            if any(word in message_lower for word in ['more', 'else', 'other', 'additionally']):
                classification['is_followup'] = True
            else:
                classification['is_followup'] = False
            
            self.logger.debug(f"Query classification result: {json.dumps(classification)}")
            return classification
            
        except Exception as e:
            self.logger.error(f"Error in classify_query: {str(e)}", exc_info=True)
            return {
                'type': 'general',
                'specific_company': None,
                'specific_technology': None,
                'intent': 'unknown'
            }
    
    def find_relevant_content(self, query: str) -> Dict[str, any]:
        """Find relevant content from resume based on query"""
        try:
            query_lower = query.lower()
            relevant_content = {
                "experiences": [],
                "projects": [],
                "skills": [],
                "education": [],
                "certifications": []
            }
            
            # Search experiences - check company names, roles, and achievements
            for exp in RESUME_DATA['experiences']:
                try:
                    # Check if company name or role is mentioned in query
                    if (exp['company'].lower() in query_lower or 
                        any(keyword in query_lower for keyword in exp['company'].lower().split()) or
                        exp['role'].lower() in query_lower or
                        any(keyword in query_lower for keyword in exp['role'].lower().split())):
                        relevant_content['experiences'].append(exp)
                        continue
                    
                    # Check if any technology used in this experience is mentioned
                    for tech in exp.get('technologies', []):
                        if tech.lower() in query_lower:
                            relevant_content['experiences'].append(exp)
                            break
                    
                    # Check achievements for relevant keywords
                    for achievement in exp.get('achievements', []):
                        if any(word in achievement.lower() for word in query_lower.split() if len(word) > 3):
                            if exp not in relevant_content['experiences']:
                                relevant_content['experiences'].append(exp)
                            break
                except Exception as e:
                    self.logger.error(f"Error processing experience {exp.get('company', 'Unknown')}: {str(e)}")
            
            # Search projects
            for project in RESUME_DATA['projects']:
                try:
                    # Check project name
                    if project['name'].lower() in query_lower:
                        relevant_content['projects'].append(project)
                        continue
                    
                    # Check technologies
                    for tech in project.get('technologies', []):
                        if tech.lower() in query_lower:
                            relevant_content['projects'].append(project)
                            break
                    
                    # Check description
                    if any(word in project.get('description', '').lower() for word in query_lower.split() if len(word) > 3):
                        if project not in relevant_content['projects']:
                            relevant_content['projects'].append(project)
                except Exception as e:
                    self.logger.error(f"Error processing project {project.get('name', 'Unknown')}: {str(e)}")
            
            # Search skills
            for category, skills in RESUME_DATA.get('skills', {}).items():
                try:
                    for skill in skills:
                        if skill.lower() in query_lower:
                            if category not in [s['category'] for s in relevant_content['skills']]:
                                relevant_content['skills'].append({
                                    'category': category,
                                    'skills': skills
                                })
                except Exception as e:
                    self.logger.error(f"Error processing skills category {category}: {str(e)}")
            
            # Search education
            if any(edu_keyword in query_lower for edu_keyword in ['education', 'degree', 'university', 'masters', 'bachelors']):
                relevant_content['education'] = [
                    RESUME_DATA['education']['masters'],
                    RESUME_DATA['education']['bachelors']
                ]
            
            # Search certifications
            for cert in RESUME_DATA.get('certifications', []):
                if any(cert_word in query_lower for cert_word in cert.lower().split()):
                    relevant_content['certifications'].append(cert)
            
            return relevant_content
            
        except Exception as e:
            self.logger.error(f"Error in find_relevant_content: {str(e)}", exc_info=True)
            return {
                "experiences": [],
                "projects": [],
                "skills": [],
                "education": [],
                "certifications": []
            }
    
    def find_relevant_projects(self, query: str) -> List[Dict[str, str]]:
        """Find projects related to the user's query"""
        relevant_links = []
        query_lower = query.lower()
        
        # Check each project
        for project in RESUME_DATA['projects']:
            # Skip if no GitHub URL
            if not project.get('github_url'):
                continue
                
            # Check if any technology in the project matches the query
            for tech in project['technologies']:
                if tech.lower() in query_lower:
                    relevant_links.append({
                        "name": project['name'],
                        "url": project['github_url']
                    })
                    break
            
            # Also check project name and description
            if (project['name'].lower() in query_lower or 
                any(word in project['description'].lower() for word in query_lower.split())):
                if not any(link['url'] == project['github_url'] for link in relevant_links):
                    relevant_links.append({
                        "name": project['name'],
                        "url": project['github_url']
                    })
        
        return relevant_links
    
    def create_system_prompt(self, user_message: str):
        """Create a detailed prompt with resume information and user query"""
        # Get relevant content first
        relevant_content = self.find_relevant_content(user_message)
        
        prompt = f"""You are an AI assistant representing {RESUME_DATA['personal']['name']}'s professional portfolio.
You have access to Shardul's complete resume and should answer questions as if you ARE Shardul.

CRITICAL INSTRUCTIONS:
1. ALWAYS respond in FIRST PERSON as Shardul (use "I", "my", "me")
2. Be SPECIFIC with numbers, metrics, and achievements from the resume
3. When discussing technical work, mention 1-2 specific technologies used
4. Keep responses concise but impactful (2-3 paragraphs max)
5. If asked about a company/role, lead with the most impressive achievement
6. Include GitHub links only when specifically relevant to technical discussions

RESPONSE PATTERNS FOR COMMON QUERIES:

If asked about experience at a specific company:
"At [Company], I worked as [Role] from [Duration]. My most impactful contribution was [specific achievement with metrics]. I primarily used [2-3 key technologies] to [specific technical accomplishment]."

If asked about skills/technologies:
"I have extensive experience with [technology], particularly in my role at [Company] where I [specific achievement]. For example, [specific project or metric]. You can see my [technology] implementation in my [Project Name] project: [GitHub link]."

If asked about data engineering experience:
"I have [X years] of data engineering experience across companies like [list 2-3]. I've built scalable pipelines processing [specific volumes] using [key technologies]. My expertise includes [2-3 core areas with specific examples]."

If asked general/vague questions:
Start with the most relevant and impressive experience, then provide 1-2 supporting examples with metrics.

SHARDUL'S INFORMATION:

CONTACT:
- Email: {RESUME_DATA['personal']['email']}
- Phone: {RESUME_DATA['personal']['phone']}
- LinkedIn: {RESUME_DATA['personal']['linkedin']}
- GitHub: {RESUME_DATA['personal']['github']}

EDUCATION & CERTIFICATIONS:
- {RESUME_DATA['education']['masters']['degree']} from {RESUME_DATA['education']['masters']['university']} (GPA: {RESUME_DATA['education']['masters']['gpa']})
- AWS Certified Data Engineer Associate
- {', '.join([cert for cert in RESUME_DATA['certifications'] if 'AWS' not in cert])}

"""
        
        # Add context-specific sections based on query type
        query_lower = user_message.lower()
        
        # Determine query focus
        is_experience_query = any(word in query_lower for word in ['experience', 'work', 'role', 'company'])
        is_technical_query = any(word in query_lower for word in ['build', 'developed', 'technical', 'project', 'implement'])
        is_skills_query = any(word in query_lower for word in ['skills', 'technologies', 'proficient', 'expertise'])
        
        # Add relevant experiences with focus
        if relevant_content['experiences']:
            prompt += "\nMOST RELEVANT EXPERIENCE:\n"
            prompt += self._format_experiences(relevant_content['experiences'][:2])  # Limit to top 2
        elif is_experience_query:
            prompt += "\nALL WORK EXPERIENCE:\n"
            prompt += self._format_experiences()
        else:
            # Add just recent experience
            prompt += "\nRECENT EXPERIENCE HIGHLIGHTS:\n"
            prompt += self._format_experiences(RESUME_DATA['experiences'][:2])
        
        # Add relevant projects with focus
        if relevant_content['projects'] or is_technical_query:
            prompt += "\n\nKEY TECHNICAL PROJECTS:\n"
            projects_to_show = relevant_content['projects'][:3] if relevant_content['projects'] else RESUME_DATA['projects'][:3]
            prompt += self._format_projects(projects_to_show)
        
        # Add skills if relevant
        if is_skills_query or relevant_content['skills']:
            prompt += f"\n\nTECHNICAL SKILLS:\n{self._format_skills()}"
        
        prompt += f"\n\nUSER QUESTION: {user_message}\n\n"
        
        # Add specific instructions based on query type
        if any(company['company'].lower() in query_lower for company in RESUME_DATA['experiences']):
            prompt += "INSTRUCTION: Focus specifically on the experience at the mentioned company. Lead with the role, duration, and most impressive achievement. Be specific about technologies used and impact created."
        elif is_skills_query:
            prompt += "INSTRUCTION: Mention specific projects or experiences where you used these skills. Include metrics and outcomes where possible."
        elif is_technical_query:
            prompt += "INSTRUCTION: Provide technical details about your implementation, the scale of the solution, and the impact. Include relevant GitHub links."
        else:
            prompt += "INSTRUCTION: Provide a comprehensive but concise response highlighting your most relevant experience. Focus on achievements with quantifiable impact."
        
        prompt += "\n\nRemember: Speak as Shardul in first person, be specific with metrics, and keep the response professional yet conversational."
        
        return prompt

    def _format_experiences(self, experiences=None):
        """Format work experiences for the prompt"""
        experiences_list = experiences if experiences else RESUME_DATA['experiences']
        formatted = []
        for exp in experiences_list:
            exp_text = f"\n{exp['role']} at {exp['company']} ({exp['duration']}):"
            for achievement in exp['achievements']:
                exp_text += f"\n  - {achievement}"
            exp_text += f"\n  Technologies: {', '.join(exp['technologies'])}"
            formatted.append(exp_text)
        return "\n".join(formatted)

    def _format_projects(self, projects=None):
        """Format projects for the prompt"""
        projects_list = projects if projects else RESUME_DATA['projects']
        formatted = []
        for proj in projects_list:
            proj_text = f"\n{proj['name']}:"
            proj_text += f"\n  Description: {proj['description']}"
            if proj.get('github_url'):
                proj_text += f"\n  GitHub: {proj['github_url']}"
            proj_text += f"\n  Technologies: {', '.join(proj['technologies'])}"
            formatted.append(proj_text)
        return "\n".join(formatted)

    def _format_skills(self):
        """Format skills by category for the prompt"""
        skills = []
        for category, skill_list in RESUME_DATA['skills'].items():
            skills.append(f"{category}: {', '.join(skill_list)}")
        return "\n".join(skills)

    def get_fallback_responses(self) -> Dict[str, any]:
        """Define comprehensive fallback responses for different scenarios"""
        return {
            'experience': {
                'general': "I'm an AWS Certified Data Engineer with experience at companies like Crewasis, Skyworks Solutions, Northeastern University, and Accion Labs. I've built scalable data pipelines processing millions of records, reduced data latency by 60%, and implemented ML solutions that improved business metrics by 15-40%. My expertise spans cloud platforms (AWS, Azure, GCP), big data technologies (Spark, Hadoop), and modern data engineering tools (Airflow, dbt, Snowflake).",
                'recent': f"Currently, I'm working as an AI Software Engineer Intern at Crewasis (ending {RESUME_DATA['experiences'][0]['duration'].split('—')[1].strip()}), where I've designed ETL pipelines processing 2M+ records and built ML pipelines with AWS Lambda. Previously, I was a Data Analytics Engineer at Skyworks Solutions, where I engineered 25+ Airflow pipelines and reduced data refresh latency by 60%.",
                'highlights': "Throughout my career, I've consistently delivered high-impact solutions: processing 2M+ unstructured records at Crewasis, building 25+ production pipelines at Skyworks, creating SnowGPT (40% analytics adoption increase) at Northeastern, and reducing ServiceNow ticket escalations by 40% at Accion Labs."
            },
            'skills': {
                'programming': "I'm proficient in Python (my primary language), SQL, and have experience with Scala, Java, and C++. I write clean, efficient code and have built everything from ETL pipelines to REST APIs to ML models.",
                'cloud': "I'm AWS Certified and experienced with AWS (S3, EC2, Lambda, Redshift, Glue), Azure (Data Factory, Synapse, SQL), and GCP (BigQuery, Cloud Run). I use Terraform for infrastructure as code and Kubernetes for container orchestration.",
                'data_engineering': "My data engineering toolkit includes Apache Airflow (25+ pipelines built), dbt, PySpark (processed 1.2M+ records), Databricks, and various data platforms like Snowflake, Redshift, and PostgreSQL. I've implemented both batch and real-time data pipelines.",
                'general': "My technical skills span programming (Python, SQL), cloud platforms (AWS, Azure, GCP), data engineering tools (Airflow, Spark, dbt), and ML frameworks. I'm experienced in building scalable systems that process millions of records efficiently."
            },
            'projects': {
                'overview': "I've built several impactful projects including NewsSphere (personalized news platform with RAG), SnowGPT (NL-to-SQL tool increasing analytics adoption by 40%), AWS YouTube Analytics (processing 80K+ interactions), and enterprise retail analytics pipelines. All my projects focus on scalability, real-world impact, and clean architecture.",
                'technical': "My projects demonstrate expertise in: cloud-native architectures (Docker, Kubernetes), real-time data processing (Spark, Lambda), ML/AI integration (RAG, LLMs), and modern web technologies (FastAPI, React). Check out my GitHub for implementation details.",
                'impact': "My projects have delivered measurable impact: 35% faster content retrieval (NewsSphere), 40% increase in analytics adoption (SnowGPT), real-time analysis of 80K+ interactions (YouTube Analytics), and processing of 25M+ retail records (Iowa Sales Intelligence)."
            },
            'education': {
                'details': f"I hold a Master of Science in Information Systems from Northeastern University (GPA: {RESUME_DATA['education']['masters']['gpa']}) and a Bachelor of Engineering in Computer Science from the University of Mumbai. My education combined with hands-on experience gives me both theoretical knowledge and practical skills.",
                'certifications': "I'm AWS Certified Data Engineer Associate and currently pursuing AWS Solutions Architect certification. I also hold certifications in Salesforce AI Associate and dbt Fundamentals, demonstrating my commitment to continuous learning."
            },
            'introduction': {
                'short': "I'm Shardul Chavan, an AWS Certified Data Engineer specializing in building scalable data solutions. With experience at companies from startups to Fortune 500, I've processed millions of records, reduced latencies by 60%, and increased analytics adoption by 40%.",
                'detailed': "I'm Shardul Chavan, an AWS Certified Data Engineer with a Master's from Northeastern University. I've worked across diverse environments - from Crewasis (startup) to Skyworks Solutions (Fortune 500), building data pipelines that process millions of records. My expertise includes cloud platforms (AWS, Azure), big data (Spark, Hadoop), and ML integration (LangChain, LLMs). I'm passionate about creating efficient, scalable solutions that deliver real business value.",
                'elevator': "AWS Certified Data Engineer with experience building pipelines processing 2M+ records, reducing latencies by 60%, and implementing ML solutions. Skilled in Python, cloud platforms, and modern data stack. Currently at Crewasis, previously at Skyworks Solutions."
            }
        }
    
    def _generate_fallback_response(self, user_message: str, classification: Dict, 
                                   relevant_content: Dict, relevant_links: List) -> Dict:
        """Generate intelligent fallback response based on query classification"""
        
        fallbacks = self.get_fallback_responses()
        response = ""
        
        # Handle different query types with rich responses
        if classification['type'] == 'company_specific' and classification['specific_company']:
            # Find the specific company experience
            for exp in RESUME_DATA['experiences']:
                if exp['company'] == classification['specific_company']:
                    response = f"At {exp['company']}, I worked as a {exp['role']} from {exp['duration']}. "
                    
                    # Add top 2 achievements
                    for i, achievement in enumerate(exp['achievements'][:2]):
                        if i == 0:
                            response += f"\n\nMy key achievement was: {achievement} "
                        else:
                            response += f"\n\nI also {achievement[0].lower()}{achievement[1:]} "
                    
                    response += f"\n\nTechnologies I used: {', '.join(exp['technologies'][:5])}."
                    break
            
            if not response:
                response = fallbacks['experience']['general']
        
        elif classification['type'] == 'skills' and classification['specific_technology']:
            tech = classification['specific_technology']
            response = f"Yes, I have strong experience with {tech}. "
            
            # Find where this tech was used
            experiences_with_tech = []
            projects_with_tech = []
            
            for exp in RESUME_DATA['experiences']:
                if any(t.lower() == tech for t in exp['technologies']):
                    experiences_with_tech.append(exp)
            
            for proj in RESUME_DATA['projects']:
                if any(t.lower() == tech for t in proj['technologies']):
                    projects_with_tech.append(proj)
            
            if experiences_with_tech:
                exp = experiences_with_tech[0]
                response += f"\n\nI used {tech} extensively at {exp['company']} where "
                response += f"{exp['achievements'][0]} "
            
            if projects_with_tech:
                proj = projects_with_tech[0]
                response += f"\n\nYou can see my {tech} skills in my {proj['name']} project: {proj['description']}"
                if proj.get('github_url'):
                    response += f" Check it out at {proj['github_url']}"
            
            if not experiences_with_tech and not projects_with_tech:
                # Use general skills fallback
                if any(tech in ' '.join(skills).lower() for skills in RESUME_DATA['skills'].values()):
                    response += fallbacks['skills']['general']
        
        elif classification['type'] == 'skills' and not classification['specific_technology']:
            # General skills query
            response = fallbacks['skills']['general']
            response += f"\n\nKey areas: {', '.join(list(RESUME_DATA['skills'].keys()))}."
        
        elif classification['type'] == 'education':
            response = fallbacks['education']['details']
            if 'certif' in user_message.lower():
                response = fallbacks['education']['certifications']
        
        elif classification['type'] == 'contact':
            response = f"I'd be happy to connect! You can reach me through:\n"
            response += f"📧 Email: {RESUME_DATA['personal']['email']}\n"
            response += f"📱 Phone: {RESUME_DATA['personal']['phone']}\n"
            response += f"💼 LinkedIn: {RESUME_DATA['personal']['linkedin']}\n"
            response += f"🐙 GitHub: {RESUME_DATA['personal']['github']}"
        
        elif classification['type'] == 'projects':
            if relevant_content['projects']:
                # Specific project mentioned
                proj = relevant_content['projects'][0]
                response = f"Let me tell you about {proj['name']} - {proj['description']}\n\n"
                if 'key_achievements' in proj and proj['key_achievements']:
                    response += f"Key highlights:\n{proj['key_achievements'][0]}\n"
                response += f"\nTech stack: {', '.join(proj['technologies'][:5])}"
                if proj.get('github_url'):
                    response += f"\n\nView the code: {proj['github_url']}"
            else:
                # General projects query
                response = fallbacks['projects']['overview']
                response += "\n\nSome highlights:\n"
                for i, proj in enumerate(RESUME_DATA['projects'][:3], 1):
                    response += f"{i}. {proj['name']} - {proj['description'][:100]}...\n"
        
        elif classification['type'] == 'general_intro':
            # Choose appropriate introduction based on query length
            if len(user_message.split()) < 5:
                response = fallbacks['introduction']['short']
            else:
                response = fallbacks['introduction']['detailed']
        
        elif classification['type'] == 'experience':
            # Experience queries
            if 'recent' in user_message.lower() or 'current' in user_message.lower():
                response = fallbacks['experience']['recent']
            elif 'highlight' in user_message.lower() or 'achievement' in user_message.lower():
                response = fallbacks['experience']['highlights']
            else:
                response = fallbacks['experience']['general']
        
        # Default fallback if no specific response generated
        if not response:
            response = fallbacks['introduction']['short']
            response += f"\n\nFeel free to ask me about specific experiences, projects, or skills!"
        
        # Add a personal touch at the end
        if not classification['type'] == 'contact':
            response += f"\n\nWant to discuss further? Reach me at {RESUME_DATA['personal']['email']}"
        
        return {
            "response": response,
            "relevant_links": relevant_links if relevant_links else [{
                "name": "GitHub Profile",
                "url": RESUME_DATA['personal']['github']
            }]
        }
    
    def process_query(self, user_message: str) -> Dict[str, any]:
        """Process user query and return response with relevant links"""
        
        start_time = datetime.now()
        query_id = f"query_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"[{query_id}] Processing query: {user_message[:100]}...")
        
        try:
            # Classify the query first
            query_classification = self.classify_query(user_message)
            self.logger.info(f"[{query_id}] Classification: {json.dumps(query_classification)}")
            
            # Find relevant project links
            relevant_links = self.find_relevant_projects(user_message)
            self.logger.info(f"[{query_id}] Found {len(relevant_links)} relevant project links")
            
            # Find all relevant content
            relevant_content = self.find_relevant_content(user_message)
            content_summary = {
                'experiences': len(relevant_content['experiences']),
                'projects': len(relevant_content['projects']),
                'skills': len(relevant_content['skills'])
            }
            self.logger.info(f"[{query_id}] Relevant content found: {json.dumps(content_summary)}")
            
            # If no API key or model not initialized, return intelligent fallback
            if not self.api_key or not self.model:
                self.logger.info(f"[{query_id}] Using fallback response (no API key or model)")
                response = self._generate_fallback_response(
                    user_message, 
                    query_classification, 
                    relevant_content, 
                    relevant_links
                )
                self._log_response_metrics(query_id, start_time, "fallback", response)
                return response
            
            try:
                # Create prompt with user message and classification context
                prompt = self.create_system_prompt(user_message)
                
                # Add classification context to prompt
                if query_classification['specific_company']:
                    prompt += f"\n\nNOTE: User is specifically asking about experience at {query_classification['specific_company']}."
                if query_classification['specific_technology']:
                    prompt += f"\n\nNOTE: User is interested in {query_classification['specific_technology']} skills/experience."
                
                self.logger.info(f"[{query_id}] Sending request to Gemini API")
                
                # Generate response using Gemini with timeout
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=600
                    )
                )
                
                if not response or not response.text:
                    raise ValueError("Empty response from Gemini API")
                
                result = {
                    "response": response.text,
                    "relevant_links": relevant_links
                }
                
                self._log_response_metrics(query_id, start_time, "gemini", result)
                self.logger.info(f"[{query_id}] Successfully generated Gemini response")
                
                return result
                
            except Exception as api_error:
                self.logger.error(f"[{query_id}] Gemini API error: {str(api_error)}")
                
                # Use classification for better fallback
                response = self._generate_fallback_response(
                    user_message, 
                    query_classification, 
                    relevant_content, 
                    relevant_links
                )
                self._log_response_metrics(query_id, start_time, "fallback_after_error", response)
                return response
                
        except Exception as e:
            self.logger.error(f"[{query_id}] Unexpected error in process_query: {str(e)}", exc_info=True)
            
            # Emergency fallback
            emergency_response = {
                "response": f"I'm Shardul Chavan, an AWS Certified Data Engineer. I apologize for the technical difficulty. "
                           f"Please feel free to contact me directly at {RESUME_DATA['personal']['email']} "
                           f"or explore my projects at {RESUME_DATA['personal']['github']}",
                "relevant_links": [{
                    "name": "GitHub Profile",
                    "url": RESUME_DATA['personal']['github']
                }]
            }
            
            self._log_response_metrics(query_id, start_time, "emergency", emergency_response)
            return emergency_response
    
    def _log_response_metrics(self, query_id: str, start_time: datetime, 
                            response_type: str, response: Dict[str, any]):
        """Log metrics about the response"""
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            "query_id": query_id,
            "response_type": response_type,
            "elapsed_time_seconds": elapsed_time,
            "response_length": len(response.get("response", "")),
            "num_links": len(response.get("relevant_links", [])),
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Response metrics: {json.dumps(metrics)}")
        
        # Log slow responses
        if elapsed_time > 3.0:
            self.logger.warning(f"[{query_id}] Slow response: {elapsed_time:.2f}s")