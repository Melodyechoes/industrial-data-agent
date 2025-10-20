from flask import Blueprint, request, jsonify, send_file

bp = Blueprint('auth', __name__, url_prefix='/')

@bp.route('/')
def index():
    return send_file('templates/index.html')

@bp.route('/login')
def login():
    return send_file('templates/login.html')

@bp.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if username == 'admin' and password == '00000000':
        return jsonify({'success': True, 'message': '登录成功'})
    else:
        return jsonify({'success': False, 'message': '用户名或密码错误'})

@bp.route('/main')
def main():
    return send_file('templates/main.html')