from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from config import Config
import logging

logger = logging.getLogger(__name__)

class GraniteModels:
    """Wrapper class for IBM Granite models"""
    
    def __init__(self):
        """Initialize with IBM watsonx.ai credentials"""
        Config.validate()
        
        self.credentials = {
            "url": Config.IBM_WATSONX_URL,
            "apikey": Config.IBM_WATSONX_APIKEY
        }
        self.project_id = Config.IBM_WATSONX_PROJECT_ID
        
        logger.info("Initialized GraniteModels with watsonx.ai credentials")
    
    def get_llm(self, temperature=0.1, max_tokens=1000):
        """Get the Granite LLM model"""
        try:
            parameters = {
                GenParams.MAX_NEW_TOKENS: max_tokens,
                GenParams.TEMPERATURE: temperature,
                GenParams.TOP_P: 0.9,
                GenParams.REPETITION_PENALTY: 1.1,
            }
            
            llm = WatsonxLLM(
                model_id="ibm/granite-13b-chat-v2",
                url=self.credentials["url"],
                apikey=self.credentials["apikey"],
                project_id=self.project_id,
                params=parameters
            )
            
            logger.info("Successfully initialized Granite LLM")
            return llm
            
        except Exception as e:
            logger.error(f"Error initializing Granite LLM: {str(e)}")
            raise
    
    def get_embeddings(self):
        """Get the Granite embeddings model"""
        try:
            embeddings = WatsonxEmbeddings(
                model_id="ibm/slate-125m-english-rtrvr",
                url=self.credentials["url"],
                apikey=self.credentials["apikey"],
                project_id=self.project_id
            )
            
            logger.info("Successfully initialized Granite Embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error initializing Granite Embeddings: {str(e)}")
            raise
    
    def test_connection(self):
        """Test connection to watsonx.ai"""
        try:
            llm = self.get_llm()
            test_response = llm.invoke("Hello, this is a test.")
            logger.info("Successfully tested watsonx.ai connection")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to watsonx.ai: {str(e)}")
            return False
