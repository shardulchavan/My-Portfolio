# resume_data.py

RESUME_DATA = {
    "personal": {
        "name": "Shardul Chavan",
        "title": "Software Engineer • Data & AI (AWS/Azure/Snowflake)",
        "email": "shardulc36@gmail.com",
        "phone": "+1-857-313-5138",
        "location": "Boston, MA (Open to Relocate)",
        "linkedin": "https://www.linkedin.com/in/shardulchavan36/",
        "github": "https://github.com/shardulchavan",
        "portfolio": "https://shardulchavan.github.io/My-Portfolio/"
    },

    "education": {
        "masters": {
            "degree": "Master of Science, Information Systems",
            "university": "Northeastern University",
            "gpa": "3.76/4.00",
            "duration": "Sep 2022 – Dec 2024",
            "location": "Boston, MA"
        },
        "bachelors": {
            "degree": "Bachelor of Engineering, Computer Science and Engineering",
            "university": "University of Mumbai",
            "duration": "Aug 2019 – May 2022",
            "location": "Mumbai, India"
        }
    },

    "certifications": [
        "AWS Certified – Data Engineer Associate",
        "Salesforce AI Associate",
        "dbt Fundamentals",
        "AWS Solutions Architect – Associate (Pursuing)"
    ],

    "experiences": [
        {
            "company": "Crewasis",
            "location": "Dallas, TX (Remote)",
            "role": "AI Software Engineer Intern",
            "duration": "Jan 2025 – Jun 2025",
            "current": False,
            "achievements": [
                "When Crewasis needed to process millions of unstructured social and news records daily (Situation), I was tasked with building a scalable pipeline to enable real-time NLP and sentiment analysis (Task). I architected ETL workflows using AWS Lambda and EC2 (Action), which successfully processed 2M+ records and delivered insights that informed product-market strategy, improving decision-making accuracy (Result).",
                "Crewasis struggled with lengthy manual retraining cycles for topic models (Situation). My responsibility was to reduce this bottleneck (Task). I implemented an EC2-hosted ML pipeline orchestrated with AWS Lambda that automated Latent Semantic Indexing (Action). This reduced manual retraining efforts by 70%, freeing engineers to focus on higher-value work (Result).",
                "To ensure stable and reproducible deployments (Situation), I was tasked with standardizing AWS infrastructure (Task). I wrote Terraform scripts to define EC2, S3, and Lambda components (Action), which allowed the team to deploy reliable, high-availability environments repeatably, cutting down environment setup time by days (Result).",
                "The marketing team lacked visibility into NLP insights from data streams (Situation). I was asked to make insights accessible to business users (Task). I developed Tableau dashboards combining CRM segmentation with NLP-driven results (Action), which enabled targeted marketing decisions and increased campaign conversions by 15% (Result).",
                "During a critical project, Crewasis experienced inefficiencies in model deployment and QA (Situation). I was responsible for enhancing DevOps practices (Task). I introduced CI/CD pipelines with GitHub Actions and Jenkins and built automated QA test harnesses (Action), reducing deployment times and cutting manual QA by 85% (Result)."
            ],
            "technologies": [
                "Python", "AWS (EC2, S3, Lambda, Bedrock)", "Terraform",
                "Scikit-learn", "FastAPI", "Tableau", "GitHub Actions",
                "Jenkins", "Docker", "React"
            ]
        },
        {
            "company": "Skyworks Solutions Inc. (DAS BU)",
            "location": "Boston, MA",
            "role": "Software Engineer – Data Platforms (Co-op)",
            "duration": "Jan 2024 – Jun 2024",
            "current": False,
            "achievements": [
                "The analytics teams faced high latency when refreshing RF testing data (Situation). My task was to improve refresh rates and reliability (Task). I built and orchestrated 25+ pipelines with Airflow and dbt across Azure SQL and data lakes (Action), reducing latency by 60% and giving teams access to fresher data (Result).",
                "Engineers were manually extracting BOM data from vendor PDFs, costing significant time (Situation). I was assigned to automate this process (Task). I developed APIs and SDKs to parse and structure data into analytics-ready formats (Action), saving over 5 engineering hours weekly and accelerating validation workflows (Result).",
                "RF test data ingestion could not scale with growing volumes (Situation). My responsibility was to re-architect the ingestion design (Task). I built a distributed ingestion system to process millions of test records concurrently (Action), tripling throughput and supporting faster NPI cost tracking (Result).",
                "The company experienced downtime issues with containerized data services (Situation). I was asked to improve reliability (Task). I containerized data services with Docker and introduced Prometheus monitoring (Action), cutting downtime by 45% and improving observability (Result).",
                "Different business units lacked consistent access to dimensional models (Situation). I was tasked to build data models and reporting (Task). I created scalable dimensional data models and developed Power BI dashboards (Action), enabling data-driven decision-making across 3 business units (Result)."
            ],
            "technologies": [
                "Airflow", "Azure (Synapse, Data Factory, Data Lake, SQL)",
                "Power BI", "Docker", "Prometheus", "Data Modeling", "Python"
            ]
        },
        {
            "company": "Northeastern University",
            "location": "Boston, MA",
            "role": "Graduate Teaching Assistant & AI Researcher",
            "duration": "Sep 2023 – Dec 2024",
            "current": False,
            "achievements": [
                "Students and staff needed a way to query large Snowflake datasets in natural language (Situation). I was tasked with creating a solution (Task). I built SnowGPT, a LangChain-powered app with FastAPI and Streamlit (Action), enabling NL-to-SQL translation and increasing self-service analytics adoption by 40% (Result).",
                "Insurance data pipelines were slow and manual (Situation). I was asked to improve efficiency (Task). I developed PySpark pipelines on Hadoop to process 1.2M+ records (Action), which enabled faster feature engineering and improved predictive model accuracy by 20% (Result).",
                "Graduate students lacked mentorship in AI/BI practices (Situation). I took on the role of mentoring 50+ students (Task). I guided them in integrating LLMs and AI techniques into enterprise workflows (Action), resulting in several successful student-led prototypes adopted in coursework and research (Result).",
                "Deployment of ML models was error-prone and time-intensive (Situation). I was tasked with automating this (Task). I containerized ML models with Docker and automated CI/CD via GitHub Actions (Action), reducing manual deployment time by 60% (Result)."
            ],
            "technologies": [
                "PySpark", "Hadoop", "LangChain", "Snowflake", "Docker",
                "GitHub Actions", "LLMs", "Machine Learning", "FastAPI"
            ]
        },
        {
            "company": "Accion Labs",
            "location": "Mumbai, India",
            "role": "Software Engineer – Data & API Integration",
            "duration": "Jan 2022 – Jul 2022",
            "current": False,
            "achievements": [
                "The support team struggled with repetitive ServiceNow workflows (Situation). I was asked to streamline API integrations (Task). I engineered REST APIs connecting ServiceNow with external systems (Action), improving support efficiency and reducing ticket escalations by 40% (Result).",
                "The QA and product teams lacked reliable reporting automation (Situation). I was tasked with automating SQL reporting (Task). I built parameterized SQL scripts and validation pipelines (Action), which reduced manual report preparation and improved KPI/SLA tracking accuracy (Result)."
            ],
            "technologies": [
                "ServiceNow", "REST APIs", "LLM Integration", "MySQL",
                "Snowflake", "JavaScript", "Python", "SQL", "Automation"
            ]
        }
    ],

    "projects": [
        {
            "name": "NewsSphere: Personalized News Digest Platform",
            "description": "Real-time personalized news with optimized RAG indexing and vector search.",
            "technologies": ["Google Cloud", "Cloud Run", "Docker", "Pinecone", "Azure SQL", "FastAPI", "CI/CD"],
            "github_url": "https://github.com/shardulchavan/NewsSphereAI",
            "key_achievements": [
                "The challenge was delivering personalized news at scale with minimal latency (Situation). I was tasked with building a real-time platform (Task). I engineered FastAPI services and optimized RAG indexing with Pinecone (Action), cutting retrieval latency by 35% and ensuring fresh content every 8 minutes (Result).",
                "Users required a reliable delivery pipeline (Situation). I containerized services with Docker and deployed them on GCP Cloud Run (Task & Action), resulting in reliable personalized news delivery without downtime (Result)."
            ]
        },
        {
            "name": "RAGDoc: SEC PDF Processing",
            "description": "RAG system over SEC filings with secure app layer and orchestrated ingestion.",
            "technologies": ["Streamlit", "FastAPI (JWT)", "OpenAI", "Airflow", "Docker", "Azure"],
            "github_url": "https://github.com/shardulchavan/CFA-PDFs-RAG",
            "key_achievements": [
                "Analysts struggled with extracting insights from SEC filings (Situation). I was tasked with building an AI-powered Q&A system (Task). I built a RAG pipeline integrating Nougat OCR and OpenAI LLMs (Action), enabling structured question answering and saving analysts hours of manual work (Result).",
                "Deployment needed to be secure and scalable (Situation). I containerized services, added JWT authentication, and orchestrated ingestion with Airflow (Action), delivering a reliable and secure solution (Result)."
            ]
        },
        {
            "name": "SnowGPT (Natural-Language to SQL on Snowflake)",
            "description": "Self-service analytics assistant for business users.",
            "technologies": ["LangChain", "Snowflake", "FastAPI", "Streamlit", "Docker", "GitHub Actions"],
            "github_url": "https://github.com/shardulchavan/SnowGPT",
            "key_achievements": [
                "Non-technical teams found querying Snowflake difficult (Situation). I was tasked to democratize data access (Task). I developed SnowGPT with LangChain and FastAPI (Action), which allowed NL-to-SQL translation and increased analytics adoption by 40% (Result).",
                "The app needed observability and reliability (Situation). I set up CI/CD with GitHub Actions and containerized deployments (Action), ensuring consistent and observable production runs (Result)."
            ]
        },
        {
            "name": "AWS-Based YouTube Data Analytics",
            "description": "Serverless ELT to analyze multi-regional YouTube interactions at scale.",
            "technologies": ["AWS Lambda", "Glue", "Redshift", "S3", "Athena", "Spark", "Terraform", "QuickSight", "EKS"],
            "github_url": "https://github.com/shardulchavan/youtube-analytics",
            "key_achievements": [
                "Marketing required real-time analysis of user engagement (Situation). I was tasked to design a scalable ELT system (Task). I built a serverless pipeline with AWS Lambda, Glue, and Redshift (Action), enabling analysis of 80K+ interactions in near real-time (Result).",
                "To scale workloads efficiently (Situation), I automated deployments with Terraform and orchestrated workloads with Kubernetes EKS (Action), which allowed the system to handle growing demand seamlessly (Result)."
            ]
        },
        {
            "name": "Retail Sales Data Pipeline",
            "description": "Large-scale Spark/Databricks pipeline for trend analysis and forecasting.",
            "technologies": ["Apache Spark", "Databricks", "Azure Blob Storage", "Azure SQL", "Grafana", "OpenTelemetry"],
            "github_url": "https://github.com/shardulchavan/retail-pipeline",
            "key_achievements": [
                "The retail company needed to process millions of transactions for analytics (Situation). I was tasked to architect the pipeline (Task). I built Spark-based pipelines on Databricks (Action), which processed 1M+ transactions and enabled accurate forecasting (Result).",
                "The pipeline needed monitoring (Situation). I integrated Grafana dashboards and OpenTelemetry tracing (Action), which gave early error detection and ensured reliable data delivery (Result)."
            ]
        },
        {
            "name": "Iowa Retail Sales Intelligence & Reporting",
            "description": "Enterprise BI for retail sales with Kimball-style models.",
            "technologies": ["Alteryx", "Talend", "Azure Data Factory", "Power BI", "Tableau"],
            "github_url": "https://github.com/shardulchavan/iowa-sales-intelligence",
            "key_achievements": [
                "The business had 25M+ retail sales records that weren’t analytics-ready (Situation). I was tasked with designing ETL workflows (Task). I built workflows in Azure Data Factory and Alteryx (Action), which reduced ingestion time by 60% and provided fast insights (Result).",
                "Executives needed business-friendly reports (Situation). I designed Kimball-style data models and built Power BI dashboards (Action), which gave clear reporting and supported decision-making (Result)."
            ]
        },
        {
            "name": "File System & Process Monitoring Daemon (macOS)",
            "description": "Lightweight C++ daemon for FS/process monitoring with CLI telemetry.",
            "technologies": ["C++", "macOS", "LLDB", "IPC", "Concurrency"],
            "github_url": None,
            "key_achievements": [
                "I identified the need for better system monitoring (Situation). I set out to design a lightweight daemon for macOS (Task). I built a process and file monitoring service in C++ with IPC and concurrency features (Action), enabling real-time event tracking (Result).",
                "To validate the daemon (Situation), I practiced debugging with LLDB and tuned performance (Action), which improved stability and gave me hands-on experience with memory management and system calls (Result)."
            ]
        }
    ]
}
