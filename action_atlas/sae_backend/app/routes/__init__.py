# /root/autodl-tmp/learning/sae_backend/app/routes/__init__.py

from flask import Blueprint

# 导入蓝图
from .input import input_bp
from .explore import explore_bp
from .validate import validate_bp
# from .validate import validate_bp

# 导出蓝图，使其可以通过 from app.routes import xxx_bp 的方式导入
__all__ = ['input_bp', 'explore_bp', 'validate_bp']