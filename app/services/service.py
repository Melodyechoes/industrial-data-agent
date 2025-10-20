import os
import pandas as pd
import numpy as np
import pickle
import json
import subprocess
import tempfile
import sys
import docker
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def read_large_csv(filepath, chunksize=10000):
    """
    分块读取大型CSV文件
    """
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunks.append(chunk)
    return pd.concat(chunks, axis=0)

def statistical_analysis(df):
    """
    统计分析模块
    返回描述性统计信息
    """
    return df.describe().to_dict()

def correlation_analysis(df, target_column=None):
    """
    相关性分析模块
    如果指定目标列，则计算与目标列的相关性
    """
    corr = df.corr()
    if target_column and target_column in df.columns:
        return corr[target_column].sort_values(ascending=False).to_dict()
    return corr.to_dict()

def anomaly_detection(df, method='isolation_forest', **kwargs):
    """
    异常检测模块
    支持多种异常检测方法
    """
    if method == 'isolation_forest':
        clf = IsolationForest(**kwargs)
    elif method == 'svm':
        clf = OneClassSVM(**kwargs)
    else:
        raise ValueError("不支持的异常检测方法")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    pred = clf.fit_predict(X)
    return {'anomaly_flags': [int(x == -1) for x in pred.tolist()]}

def apply_ml_model(df, target_column, model_type='linear_regression', test_size=0.2):
    """
    应用机器学习模型进行预测分析
    支持线性回归、决策树、随机森林等模型
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 数据预处理
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 选择模型
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor()
    elif model_type == 'random_forest':
        model = RandomForestRegressor()
    else:
        raise ValueError("不支持的模型类型")
    
    # 训练和评估
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    return {
        'model': model,
        'model_type': model_type,
        'score': score,
        'features': list(X.columns),
        'coefficients': dict(zip(X.columns, model.coef_)) if hasattr(model, 'coef_') else None
    }

def save_model_to_disk(model, model_info, model_name, models_folder):
    """
    保存训练好的模型到磁盘
    """
    print(f"开始保存模型到磁盘: model_name={model_name}, models_folder={models_folder}")
    
    # 确保模型文件夹路径是绝对路径
    if not os.path.isabs(models_folder):
        models_folder = os.path.abspath(models_folder)
        print(f"转换为绝对路径: {models_folder}")
    
    # 确保模型存储目录存在
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
        print(f"创建模型目录: {models_folder}")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename_pkl = f"{model_name}_{timestamp}.pkl"
    model_filename_dat = f"{model_name}_{timestamp}.dat"
    model_path_pkl = os.path.join(models_folder, model_filename_pkl)
    model_path_dat = os.path.join(models_folder, model_filename_dat)
    
    print(f"模型文件路径: {model_path_pkl}")
    print(f"模型文件路径: {model_path_dat}")
    
    try:
        # 保存模型为.pkl格式
        with open(model_path_pkl, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存为PKL格式: {model_path_pkl}")
        
        # 保存模型为.dat格式
        with open(model_path_dat, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存为DAT格式: {model_path_dat}")
        
        # 保存模型信息
        model_info['model_path_pkl'] = model_path_pkl
        model_info['model_path_dat'] = model_path_dat
        model_info['model_name'] = model_name
        model_info['created_at'] = timestamp
        model_info['filename_pkl'] = model_filename_pkl
        model_info['filename_dat'] = model_filename_dat
        
        info_filename = f"{model_name}_{timestamp}_info.json"
        info_path = os.path.join(models_folder, info_filename)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print(f"模型信息已保存: {info_path}")
        
        # 添加调试信息
        print(f"当前工作目录: {os.getcwd()}")
        print(f"模型文件夹路径: {models_folder}")
        
        result = {
            'model_path_pkl': model_path_pkl,
            'model_path_dat': model_path_dat,
            'info_path': info_path,
            'model_filename_pkl': model_filename_pkl,
            'model_filename_dat': model_filename_dat,
            'info_filename': info_filename
        }
        print(f"保存结果: {result}")
        return result
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def load_model_from_disk(model_filename, models_folder):
    """
    从磁盘加载模型
    """
    model_path = os.path.join(models_folder, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_filename} 不存在")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def get_model_list(models_folder):
    """
    获取所有已保存的模型列表
    """
    models = []
    for filename in os.listdir(models_folder):
        if filename.endswith('_info.json'):
            info_path = os.path.join(models_folder, filename)
            with open(info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                model_info['info_filename'] = filename
                model_info['model_filename'] = filename.replace('_info.json', '.pkl')
                models.append(model_info)
    return models

def execute_python_code_in_container(code, data_file=None):
    """
    在Docker容器中安全执行Python代码
    
    Args:
        code (str): 要执行的Python代码
        data_file (str): 数据文件路径（可选）
        
    Returns:
        dict: 执行结果，包括输出和可能的错误信息
    """
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建代码文件
        code_file = os.path.join(temp_dir, "script.py")
        
        # 准备执行环境
        with open(code_file, "w", encoding="utf-8") as f:
            # 添加必要的导入
            f.write("import pandas as pd\n")
            f.write("import numpy as np\n")
            f.write("import json\n")
            f.write("import sys\n")
            f.write("import os\n\n")
            
            # 如果有数据文件，添加读取代码
            if data_file and os.path.exists(data_file):
                # 只复制文件名，因为文件会在Docker中挂载
                data_filename = os.path.basename(data_file)
                f.write(f"df = pd.read_csv('/data/{data_filename}')\n\n")
            
            f.write(code)
            
            # 添加结果输出代码
            f.write("\n\n")
            f.write("# 将结果保存到文件\n")
            f.write("result = locals().copy()\n")
            f.write("output_data = {}\n")
            f.write("for key, value in result.items():\n")
            f.write("    if not key.startswith('_') and key not in ['pd', 'np', 'json', 'sys', 'os']:\n")
            f.write("        try:\n")
            f.write("            json.dumps(value)  # 检查是否可以序列化\n")
            f.write("            output_data[key] = value\n")
            f.write("        except:\n")
            f.write("            output_data[key] = str(value)\n")
            f.write("with open('/output/result.json', 'w') as f:\n")
            f.write("    json.dump(output_data, f, default=str)\n")
        
        try:
            # 连接到Docker守护进程
            client = docker.from_env()
            
            # 获取数据文件目录和文件名
            data_dir = None
            data_filename = None
            if data_file and os.path.exists(data_file):
                data_dir = os.path.dirname(os.path.abspath(data_file))
                data_filename = os.path.basename(data_file)
            
            # 准备挂载卷
            volumes = {
                temp_dir: {'bind': '/code', 'mode': 'ro'},
                temp_dir: {'bind': '/output', 'mode': 'rw'}
            }
            
            # 如果有数据文件，添加数据目录挂载
            if data_dir:
                volumes[data_dir] = {'bind': '/data', 'mode': 'ro'}
            
            # 首先尝试使用包含必要依赖的镜像
            try:
                container = client.containers.run(
                    "python:3.9",
                    command=["python", "/code/script.py"],
                    volumes=volumes,
                    working_dir="/code",
                    detach=True,
                    remove=True,
                    network_disabled=True,  # 禁用网络以提高安全性
                )
                
                # 等待容器执行完成
                exit_code = container.wait()
                logs = container.logs().decode('utf-8')
                
            except docker.errors.ImageNotFound:
                # 如果镜像不存在，则使用slim版本并安装依赖
                container = client.containers.run(
                    "python:3.9-slim",
                    command=["sh", "-c", "pip install pandas numpy && python /code/script.py"],
                    volumes=volumes,
                    working_dir="/code",
                    detach=True,
                    remove=True,
                    network_disabled=True,  # 禁用网络以提高安全性
                )
                
                # 等待容器执行完成
                exit_code = container.wait()
                logs = container.logs().decode('utf-8')
            
            # 读取结果
            result_file = os.path.join(temp_dir, "result.json")
            output_data = {}
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    output_data = json.load(f)
            
            return {
                "success": exit_code == 0,
                "stdout": logs,
                "stderr": "" if exit_code == 0 else logs,
                "results": output_data,
                "error": None if exit_code == 0 else logs
            }
            
        except docker.errors.ContainerError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            stdout_msg = e.stdout.decode('utf-8') if e.stdout else ""
            return {
                "success": False,
                "stdout": stdout_msg,
                "stderr": error_msg,
                "results": {},
                "error": error_msg
            }
        except docker.errors.ImageNotFound:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Docker镜像未找到",
                "results": {},
                "error": "Docker镜像未找到"
            }
        except docker.errors.APIError as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Docker API错误: {str(e)}",
                "results": {},
                "error": f"Docker API错误: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "results": {},
                "error": str(e)
            }