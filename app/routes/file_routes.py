import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

# 导入主应用中的配置和函数
from utils.util import allowed_file

bp = Blueprint('files', __name__, url_prefix='/api')

@bp.route('/files', methods=['GET'])
def list_files():
    try:
        files = []
        for filename in os.listdir(current_app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath) and allowed_file(filename, current_app.config['ALLOWED_EXTENSIONS']):
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

@bp.route('/upload', methods=['POST'])
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
        if file and allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
            # 安全地获取文件名
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            
            # 保存文件
            file.save(filepath)
            
            # 读取CSV文件获取列信息
            try:
                import pandas as pd
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

@bp.route('/files/<filename>', methods=['GET'])
def get_file_content(filename):
    try:
        filename = secure_filename(filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        # 安全检查，防止路径遍历攻击
        filename = secure_filename(filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'message': '文件删除成功'})
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500