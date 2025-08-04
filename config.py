import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration class"""
    
    # IBM watsonx.ai settings
    IBM_WATSONX_APIKEY = os.getenv('IBM_WATSONX_APIKEY')
    IBM_WATSONX_URL = os.getenv('IBM_WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
    IBM_WATSONX_PROJECT_ID = os.getenv('IBM_WATSONX_PROJECT_ID')
    
    # IBM Cloud Object Storage settings
    COS_ENDPOINT = os.getenv('COS_ENDPOINT')
    COS_API_KEY = os.getenv('COS_API_KEY')
    COS_RESOURCE_INSTANCE_ID = os.getenv('COS_RESOURCE_INSTANCE_ID')
    BUCKET_NAME = os.getenv('BUCKET_NAME', 'research-agent-bucket')
    
    # Cloudant settings
    CLOUDANT_URL = os.getenv('CLOUDANT_URL')
    CLOUDANT_APIKEY = os.getenv('CLOUDANT_APIKEY')
    
    # Flask settings
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    PORT = int(os.getenv('PORT', 5000))
    
    @classmethod
    def validate(cls):
        """Validate that all required environment variables are set"""
        required_vars = [
            'IBM_WATSONX_APIKEY',
            'IBM_WATSONX_PROJECT_ID'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
