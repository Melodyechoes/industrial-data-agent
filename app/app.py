from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import pandas as pd
from werkzeug.utils import secure_filename

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
ALLOWED_EXTENSIONS = {'csv'}

# 确保上传目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 路由：首页
@app.route('/')
def hello():
    return '工业数据分析Agent系统已启动!'

# 路由：获取文件列表
@app.route('/api/files', methods=['GET'])
def list_files():
    try:
        files = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath) and allowed_file(filename):
                # 获取文件基本信息
                file_stats = os.stat(filepath)
                files.append({
                    'name': filename,
                    'size': file_stats.st_size,
                    'upload_time': file_stats.st_ctime
                })
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 路由：上传文件
@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        # 检查是否有文件部分
        if 'file' not in request.files:
            return jsonify({'error': '没有文件部分'}), 400
        
        file = request.files['file']
        
        # 检查是否选择了文件
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 检查文件类型
        if file and allowed_file(file.filename):
            # 安全地获取文件名
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 保存文件
            file.save(filepath)
            
            # 读取CSV文件获取列信息
            try:
                df = pd.read_csv(filepath)
                columns = list(df.columns)
                
                return jsonify({
                    'message': '文件上传成功',
                    'filename': filename,
                    'columns': columns,
                    'row_count': len(df)
                })
            except Exception as e:
                return jsonify({'error': f'文件处理错误: {str(e)}'}), 500
        else:
            return jsonify({'error': '文件类型不允许，仅支持CSV'}), 400
    except Exception as e:
        return jsonify({'error': f'上传过程中出错: {str(e)}'}), 500

# 路由：删除文件
@app.route('/api/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        # 安全检查，防止路径遍历攻击
        filename = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'message': '文件删除成功'})
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)