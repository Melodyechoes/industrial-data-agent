import os
import openai
import sys
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
import docker
import tempfile
import json

# 导入主应用中的配置和函数
from services.service import read_large_csv, statistical_analysis, \
    correlation_analysis, anomaly_detection, apply_ml_model, save_model_to_disk, \
    load_model_from_disk, get_model_list, execute_python_code_in_container

bp = Blueprint('analysis', __name__, url_prefix='/api')

client = None

def get_client():
    global client
    if client is None:
        if current_app.config['DEEPSEEK_API_KEY']:
            client = openai.OpenAI(
                api_key=current_app.config['DEEPSEEK_API_KEY'],
                base_url="https://api.deepseek.com"
            )
    return client

@bp.route('/models', methods=['GET'])
def list_models():
    try:
        models = get_model_list(current_app.config['MODELS_FOLDER'])
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/models/<model_filename>', methods=['DELETE'])
def delete_model(model_filename):       
    """
    删除指定的模型文件及其关联的信息文件。

    该接口会安全地删除上传的模型文件（.pkl）和其对应的元信息文件（_info.json）。
    文件名经过安全处理，防止路径穿越等安全风险。
    
    参数:
        model_filename (str): 要删除的模型文件名，通过URL路径传入。

    返回:
        JSON 响应对象：
        - 成功时返回状态码200，并包含已删除文件的消息。
        - 若文件不存在，返回状态码404和错误信息。
        - 若发生异常，返回状态码500和异常描述。
    """
    try:
        # 安全检查
        model_filename = secure_filename(model_filename)
        info_filename = model_filename.replace('.pkl', '_info.json')
        
        model_path = os.path.join(current_app.config['MODELS_FOLDER'], model_filename)
        info_path = os.path.join(current_app.config['MODELS_FOLDER'], info_filename)
        
        deleted_files = []
        if os.path.exists(model_path):
            os.remove(model_path)
            deleted_files.append(model_filename)
        
        if os.path.exists(info_path):
            os.remove(info_path)
            deleted_files.append(info_filename)
            
        if deleted_files:
            return jsonify({'message': f'模型文件 {", ".join(deleted_files)} 删除成功'})
        else:
            return jsonify({'error': '模型文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/models/<model_filename>/apply', methods=['POST'])
def apply_saved_model(model_filename):
    try:
        data = request.get_json()
        filename = data.get('filename')
        model_filename = secure_filename(model_filename)
        
        if not filename:
            return jsonify({'error': '请指定文件名'}), 400
            
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404
            
        # 读取数据
        df = read_large_csv(filepath)
        
        # 加载模型
        model = load_model_from_disk(model_filename, current_app.config['MODELS_FOLDER'])
        
        # 准备数据进行预测
        X = df.copy()
        if hasattr(model, 'feature_names_in_'):
            # 确保特征列匹配
            missing_cols = set(model.feature_names_in_) - set(X.columns)
            extra_cols = set(X.columns) - set(model.feature_names_in_)
            
            # 删除多余列
            if extra_cols:
                X = X.drop(columns=extra_cols)
            
            # 添加缺失列（用0填充）
            for col in missing_cols:
                X[col] = 0
            
            # 重新排序列以匹配模型
            X = X[model.feature_names_in_]
        
        # 进行预测
        predictions = model.predict(X)
        
        return jsonify({
            'success': True,
            'predictions': predictions.tolist(),
            'model_filename': model_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#核心交互逻辑
@bp.route('/chat', methods=['POST'])
def chat_with_deepseek():
    """
    处理与DeepSeek API的聊天交互，支持工业数据分析和模型保存功能。

    该接口接收用户消息、文件名、会话ID等参数，维护会话历史，并在请求包含文件时进行数据分析。
    若用户请求分析任务，则读取指定CSV文件并生成统计图表及分析报告。同时支持将训练好的机器学习模型保存到磁盘。

    参数:
        message (str): 用户发送的消息内容。
        filename (str, optional): 用户上传的CSV文件名，用于数据分析。
        conversation_id (str, optional): 会话标识符，用于维持多轮对话状态；若未提供则创建新会话。
        save_model (bool): 是否保存模型的标志位。
        model_name (str): 要保存的模型名称，默认为'model'。
        model_training_result (dict, optional): 包含待保存模型对象及其元数据的结果字典。

    返回:
        JSON object: 包含以下字段的响应：
            - reply (str): AI返回的回复内容。
            - conversation_id (str): 当前会话的唯一标识符。
            - analysis (dict, optional): 分析结果，包含matplotlib/plotly代码、报告和图表URL。
            - saved_model (dict, optional): 模型保存信息，包括路径和文件名等。

    异常:
        若API密钥未配置或处理过程中发生错误，返回包含错误信息的JSON及相应HTTP状态码。
    """
    # 获取客户端实例
    client = get_client()
    
    if not client:
        return jsonify({'error': 'DeepSeek API密钥未配置'}), 500
        
    try:
        # 获取请求数据
        data = request.get_json()
        user_message = data.get('message', '')
        filename = data.get('filename', '')
        conversation_id = data.get('conversation_id', '')
        save_model = data.get('save_model', False)
        model_name = data.get('model_name', 'model')
        
        # 初始化会话历史
        if not hasattr(bp, 'conversations'):
            bp.conversations = {}
            
        # 获取或创建会话
        if not conversation_id or conversation_id not in bp.conversations:
            conversation_id = str(len(bp.conversations) + 1)
            bp.conversations[conversation_id] = [
                {"role": "system", "content": "你是一个专业的工业数据分析助手。用户会向你询问关于工业数据的分析问题。当用户请求数据分析时,请生成图表代码和分析报告。如果用户要求建立机器学习模型，请在分析完成后询问用户是否要保存模型，并提供保存选项。如果你的回复中有代码部分，请先对数据进行处理并生成相对应的图表代码,而不是在代码中引用csv文件,此处代码不要直接import引用csv文件,cvs文件仍然正常供你查看,注意,图表代码不要附加报告代码,仅为图表代码。"}
            ]
        
        # 获取当前会话
        messages = bp.conversations[conversation_id]
        
        file_context = ""
        analysis_results = {}
        if filename:
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(filename))
            if os.path.exists(filepath):
                try:
                    # 读取文件
                    df = read_large_csv(filepath)
                    file_context = f"\n\n用户选择的文件是: {filename}\n文件的前5行示例如下:\n{df.head(5).to_string()}"
                    
                    # 添加基本统计信息
                    stats = statistical_analysis(df)
                    file_context += f"\n\n基本统计信息:\n{stats}"
                    
                    # 检查是否为数据分析任务
                    if "分析" in user_message or "统计" in user_message or "报告" in user_message:
                        # 生成图表代码
                        import matplotlib.pyplot as plt
                        import plotly.express as px
                        
                        # 生成matplotlib图表（保存到charts文件夹）
                        plt.figure(figsize=(10, 6))
                        df.hist()
                        chart_path = os.path.join(current_app.config['CHARTS_FOLDER'], f'{filename}_hist.png')
                        plt.savefig(chart_path)
                        plt.close()
                        
                        # 生成plotly图表（保存到charts文件夹）
                        fig = px.scatter(df, x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else df.columns[0])
                        html_path = os.path.join(current_app.config['CHARTS_FOLDER'], f'{filename}_plot.html')
                        fig.write_html(html_path)
                        
                        # 生成分析报告
                        report = {
                            'summary': statistical_analysis(df),
                            'correlation': correlation_analysis(df),
                            'anomalies': anomaly_detection(df)
                        }
                        
                        analysis_results = {
                            'matplotlib_code': 'plt.figure(figsize=(10, 6))\ndf.hist()\nplt.savefig("hist.png")',
                            'plotly_code': 'fig = px.scatter(df, x=df.columns[0], y=df.columns[1])\nfig.write_html("plot.html")',
                            'report': report,
                            'chart_urls': {
                                'histogram': f'/api/files/{filename}_hist.png',
                                'scatter': f'/api/files/{filename}_plot.html'
                            }
                        }
                except Exception as e:
                    file_context = f"\n\n用户选择的文件是: {filename}，但在读取文件时出现错误: {str(e)}"

        # 添加用户消息到会话历史
        messages.append({"role": "user", "content": user_message + file_context})

        # 调用DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )

        # 获取AI回复并添加到会话历史
        ai_message = response.choices[0].message
        messages.append(ai_message)
        
        # 更新会话历史
        bp.conversations[conversation_id] = messages
        
        # 如果需要保存模型，检查回复中是否包含模型训练结果
        saved_model_info = None
        if save_model and "model_training_result" in data:
            model_result = data["model_training_result"]
            if "model" in model_result:
                # 保存模型
                model_info = {
                    'model_type': model_result.get('model_type', 'unknown'),
                    'target_column': model_result.get('target_column', 'unknown'),
                    'features': model_result.get('features', []),
                    'score': model_result.get('score', 0)
                }
                saved_model_info = save_model_to_disk(
                    model_result["model"], 
                    model_info, 
                    model_name,
                    current_app.config['MODELS_FOLDER']
                )
        
        return jsonify({
            'reply': ai_message.content,
            'conversation_id': conversation_id,
            'analysis': analysis_results if analysis_results else None,
            'saved_model': saved_model_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/analyze', methods=['POST'])
def analyze_data():
    """
    数据分析API接口
    支持多种数据分析功能
    """
    try:
        data = request.get_json()
        print("收到分析请求，请求数据:", data)  # 添加调试信息
        
        filename = data.get('filename')
        analysis_type = data.get('analysis_type')
        save_model = data.get('save_model', False)
        model_name = data.get('model_name', 'model')
        
        if not filename:
            return jsonify({'error': '请指定文件名'}), 400
            
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404
            
        # 读取数据
        df = read_large_csv(filepath)
        print(f"数据加载成功，形状: {df.shape}")  # 添加调试信息
        
        result = None
        
        # 根据分析类型调用不同功能
        if analysis_type == 'statistical':
            result = statistical_analysis(df)
        elif analysis_type == 'correlation':
            target_column = data.get('target_column')
            result = correlation_analysis(df, target_column)
        elif analysis_type == 'anomaly':
            method = data.get('method', 'isolation_forest')
            result = anomaly_detection(df, method=method)
        elif analysis_type == 'predict':
            target_column = data.get('target_column')
            print(f"预测分析，目标列: {target_column}")  # 添加调试信息
            
            if not target_column:
                return jsonify({'error': '请指定目标列'}), 400
                
            # 检查目标列是否存在
            if target_column not in df.columns:
                return jsonify({'error': f'目标列 "{target_column}" 在数据中不存在。可用列: {list(df.columns)}'}), 400
                
            model_type = data.get('model_type', 'linear_regression')
            print(f"模型类型: {model_type}")  # 添加调试信息
            
            try:
                model_result = apply_ml_model(df, target_column, model_type)
                print("模型训练完成，结果:", model_result.keys() if model_result else "无结果")  # 添加调试信息
            except Exception as e:
                print("模型训练时出错:", str(e))  # 添加调试信息
                import traceback
                traceback.print_exc()  # 打印详细错误信息
                return jsonify({'error': f'模型训练失败: {str(e)}'}), 500
            
            # 如果需要保存模型
            if save_model:
                print("开始保存模型")  # 添加调试信息
                model_info = {
                    'model_type': model_type,
                    'target_column': target_column,
                    'features': model_result.get('features', []),
                    'score': model_result.get('score', 0)
                }
                try:
                    saved_model_info = save_model_to_disk(
                        model_result["model"], 
                        model_info, 
                        model_name,
                        current_app.config['MODELS_FOLDER']
                    )
                    # 创建不包含模型对象的返回结果
                    result = {
                        'model_type': model_result['model_type'],
                        'score': model_result['score'],
                        'features': model_result['features'],
                        'coefficients': model_result.get('coefficients'),
                        'saved_model': saved_model_info
                    }
                    print("模型保存完成，保存信息:", saved_model_info)  # 添加调试信息
                except Exception as e:
                    print("模型保存时出错:", str(e))  # 添加调试信息
                    import traceback
                    traceback.print_exc()  # 打印详细错误信息
                    return jsonify({'error': f'模型保存失败: {str(e)}'}), 500
            else:
                # 创建不包含模型对象的返回结果
                result = {
                    'model_type': model_result['model_type'],
                    'score': model_result['score'],
                    'features': model_result['features'],
                    'coefficients': model_result.get('coefficients')
                }
                
            print("返回结果:", list(result.keys()))  # 添加调试信息
        else:
            return jsonify({'error': '不支持的分析类型'}), 400
            
        return jsonify({
            'success': True,
            'result': result,
            'analysis_type': analysis_type
        })
        
    except Exception as e:
        print("处理分析请求时出错:", str(e))  # 添加调试信息
        import traceback
        traceback.print_exc()  # 打印详细错误信息
        return jsonify({'error': str(e)}), 500
    
@bp.route('/execute_code', methods=['POST'])
def execute_code():
    """
    在Docker容器中执行生成的Python代码
    """
    try:
        data = request.get_json()
        code = data.get('code', '')
        filename = data.get('filename', '')
        
        if not code:
            return jsonify({'error': '请提供要执行的Python代码'}), 400
            
        # 如果提供了文件名，则获取文件路径
        filepath = None
        if filename:
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(filename))
            if not os.path.exists(filepath):
                return jsonify({'error': '指定的文件不存在'}), 404
        
        # 在Docker容器中执行代码
        result = execute_python_code_in_container(code, filepath)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@bp.route('/generate_charts', methods=['POST'])
def generate_charts():
    """
    为指定数据文件生成常见图表并以Base64编码形式返回(二进制传输放法)
    """
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': '请指定文件名'}), 400
            
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404
            
        # 读取数据
        df = read_large_csv(filepath)
        
        # 生成图表
        charts = {}
        
        # 生成直方图
        plt.figure(figsize=(12, 8))
        df.hist(bins=30, figsize=(12, 8))
        plt.suptitle('数据分布直方图')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        hist_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        charts['histogram'] = f'data:image/png;base64,{hist_base64}'
        
        # 生成散点图（仅前两列）
        if len(df.columns) >= 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.7)
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[1])
            plt.title(f'{df.columns[0]} vs {df.columns[1]} 散点图')
            plt.grid(True, alpha=0.3)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            scatter_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            charts['scatter'] = f'data:image/png;base64,{scatter_base64}'
        
        # 生成箱线图
        plt.figure(figsize=(12, 8))
        df.plot(kind='box', vert=False)
        plt.title('数据箱线图')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        box_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        charts['box'] = f'data:image/png;base64,{box_base64}'
        
        # 生成相关性热力图（如果列数不过多）
        if len(df.columns) <= 20:
            plt.figure(figsize=(10, 8))
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('特征相关性热力图')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            corr_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            charts['correlation'] = f'data:image/png;base64,{corr_base64}'
        
        return jsonify({
            'success': True,
            'charts': charts
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@bp.route('/test_connection', methods=['GET'])
def test_connection():
    """
    测试各种连接
    """
    try:
        # 测试Docker连接
        docker_success = False
        docker_error = None
        try:
            client = docker.from_env()
            client.ping()
            docker_success = True
        except Exception as e:
            docker_error = str(e)
        
        # 测试API密钥
        client = get_client()
        api_status = "可用" if client else "不可用"
        
        return jsonify({
            'docker': {
                'success': docker_success,
                'error': docker_error
            },
            'api_key': api_status,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500