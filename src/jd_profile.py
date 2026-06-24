"""Keyword banks and weights derived from the job description."""

# Core of the role: retrieval / ranking / search / recommendation.
RETRIEVAL_RANKING = {
    "ranking", "rank", "ranker", "learning-to-rank", "learning to rank", "ltr",
    "lambdamart", "retrieval", "retriever", "search", "semantic search",
    "recommendation", "recommender", "recommendation systems", "recsys",
    "relevance", "relevance labeling", "discovery feed", "query", "matching",
    "information retrieval", "bm25", "re-ranking", "reranking",
}

EMBEDDINGS = {
    "embedding", "embeddings", "sentence transformers", "sentence-transformers",
    "bge", "nv-embed", "vector search", "nearest neighbor",
    "dense retrieval", "hybrid search", "hybrid retrieval",
}

VECTOR_DB = {
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "haystack", "vector database", "vector db",
}

ML_CORE = {
    "machine learning", "deep learning", "pytorch", "tensorflow", "keras",
    "transformers", "hugging face", "huggingface", "nlp",
    "natural language processing", "scikit-learn", "sklearn", "xgboost",
    "lightgbm", "feature engineering", "mlops", "mlflow", "kubeflow",
    "weights & biases", "model serving", "bentoml", "distributed training",
    "reinforcement learning",
}

# Recent GenAI alone (no pre-LLM depth) is a negative; handled in scoring.
LLM_GENAI = {
    "llm", "llms", "rag", "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "prompt engineering", "langchain", "llamaindex", "llama-index",
    "generative ai", "genai", "instruction tuning", "rlhf",
}

EVAL = {
    "ndcg", "mrr", "mean average precision", "a/b test", "a/b testing",
    "a/b experimentation", "ab test", "ab testing", "offline-online",
    "offline online", "offline-to-online", "experimentation framework",
    "evaluation framework", "relevance labeling", "click-through",
    "click through", "holdout", "offline metrics",
}

ENGINEER_TITLE_TERMS = {
    "machine learning engineer", "ml engineer", "ai engineer",
    "applied ml", "applied scientist", "applied machine learning",
    "data scientist", "data engineer", "software engineer", "backend engineer",
    "search engineer", "recommendation systems engineer", "nlp engineer",
    "ml research", "research engineer", "research scientist",
    "platform engineer", "mlops engineer", "full stack developer",
    "full-stack developer", ".net developer", "java developer",
    "cloud engineer", "devops engineer", "frontend engineer", "mobile developer",
    "qa engineer",
}

# Roles that count as genuine applied-ML tenure (narrower than the set above).
ML_ROLE_TERMS = {
    "machine learning engineer", "ml engineer", "ai engineer",
    "applied ml", "applied scientist", "applied machine learning",
    "data scientist", "research engineer", "research scientist",
    "nlp engineer", "recommendation systems engineer", "search engineer",
    "ml research", "deep learning", "recommendation",
}

# Off-the-keyboard management/architecture titles (kept narrow: ICs not hit).
NON_CODING_LEADER_TERMS = {
    "engineering manager", "director", "vice president", "vp of", "head of",
    "chief", "cto", "architect",
}

# Non-engineering self-identification -> AI skills in the list are noise.
NON_ENGINEER_TITLE_TERMS = {
    "marketing manager", "operations manager", "accountant", "hr manager",
    "customer support", "civil engineer", "mechanical engineer",
    "graphic designer", "sales executive", "content writer", "project manager",
    "business analyst",
}

# Phrases the keyword-stuffer traps use to describe themselves.
TRAP_SELF_ID_PHRASES = {
    "i've spent my career in marketing manager",
    "my professional background is in marketing manager",
    "i'm a marketing manager",
    "professional with",
    "apply my domain expertise alongside emerging ai",
    "experimented with chatgpt",
    "curious about how ai tools could augment my work",
}

ENGINEER_SELF_ID_PHRASES = {
    "machine learning engineer with", "ml engineer", "software engineer with",
    "shipped", "in production", "built and shipped", "production load",
    "embedding-based retrieval", "ranking models", "learning-to-rank",
    "recommendation system", "search product", "retrieval", "feature pipeline",
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree",
}

PRODUCT_COMPANIES = {
    "swiggy", "zomato", "uber", "ola", "flipkart", "razorpay", "cred",
    "meesho", "phonepe", "nykaa", "juspay", "sarvam ai", "krutrim",
    "mad street den", "infilect", "suki ai", "google", "meta", "microsoft",
    "amazon", "nvidia", "openai", "microsoft research", "fractal analytics",
    "byju's", "pied piper", "hooli",
}

PRODUCT_INDUSTRIES = {
    "food delivery", "fintech", "e-commerce", "transportation", "ai/ml",
    "software",
}
SERVICES_INDUSTRIES = {"it services", "consulting"}

# Primary CV/speech/robotics without NLP/IR is a negative.
CV_SPEECH_ROBOTICS = {
    "computer vision", "opencv", "yolo", "image classification",
    "object detection", "gans", "cnn", "speech recognition", "tts", "asr",
    "robotics", "diffusion",
}

# Pune/Noida preferred; rest of India relocatable; outside India = no visa.
PREFERRED_CITIES = {"pune", "noida"}
WELCOME_CITIES = {
    "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "noida", "pune",
    "bangalore", "bengaluru", "ncr", "chandigarh",
}

# Technical-fit weights (applied after the coherence gate); sum to 1.0.
FIT_WEIGHTS = {
    "retrieval_ranking": 0.26,
    "ml_depth": 0.14,
    "vector_db": 0.10,
    "evaluation": 0.08,
    "product_company": 0.12,
    "experience_band": 0.06,
    "ml_tenure": 0.08,
    "skill_depth": 0.10,
    "location": 0.06,
}
