
from dotenv import load_dotenv
import os

class Config:
    """应用配置类"""
    # 基本配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    UPLOAD_FOLDER = 'uploads'
    CHARTS_FOLDER = 'charts'  # 确保图表文件夹配置在类属性中
    MODELS_FOLDER = 'models'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # 允许的文件扩展名
    ALLOWED_EXTENSIONS = {'csv'}
    
    # DeepSeek API配置
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    
    @staticmethod
    def init_app(app):
        """初始化应用配置"""
        # 确保上传目录存在
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)
            print(f"创建上传目录: {Config.UPLOAD_FOLDER}")
        else:
            print(f"上传目录已存在: {Config.UPLOAD_FOLDER}")
            
        # 确保模型存储目录存在
        if not os.path.exists(Config.MODELS_FOLDER):
            os.makedirs(Config.MODELS_FOLDER)
            print(f"创建模型目录: {Config.MODELS_FOLDER}")
        else:
            print(f"模型目录已存在: {Config.MODELS_FOLDER}")
            
        # 确保图表存储目录存在
        if not os.path.exists(Config.CHARTS_FOLDER):
            os.makedirs(Config.CHARTS_FOLDER)
            print(f"创建图表目录: {Config.CHARTS_FOLDER}")
        else:
            print(f"图表目录已存在: {Config.CHARTS_FOLDER}")