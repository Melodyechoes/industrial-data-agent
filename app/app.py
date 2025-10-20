# 标准库
import os
import warnings

# 第三方库
## Flask相关
from flask import Flask
from flask_cors import CORS

## 环境变量
from dotenv import load_dotenv

# 初始化警告设置
warnings.filterwarnings('ignore')

# 加载.env文件中的环境变量
load_dotenv()

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 导入配置
from config import Config
app.config.from_object(Config)

# 初始化应用配置
Config.init_app(app)

def create_app():
    """应用工厂函数"""
    # 注册蓝图
    from routes.auth_routes import bp as auth_bp
    app.register_blueprint(auth_bp)
    
    from routes.file_routes import bp as file_bp
    app.register_blueprint(file_bp)
    
    from routes.analysis_routes import bp as analysis_bp
    app.register_blueprint(analysis_bp)
    
    return app

# 创建应用实例
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)