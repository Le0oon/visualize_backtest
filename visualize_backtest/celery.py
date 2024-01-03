# celery.py
from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from django.conf import settings


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'visualize_backtest.settings')

app = Celery('visualize_backtest',broker='redis://127.0.0.1:6379/0',backend='redis://127.0.0.1:6379/0')

app.config_from_object('django.conf:settings')

# app.conf.beat_schedule = {
#     'autosc': {  # 取个名字
#         'task': 'user.tasks.auto_sc',  # 设置是要将哪个任务进行定时
#         'schedule': crontab(),  # 调用crontab进行具体时间的定义
#     },
# }
# 自动从所有已注册的django app中加载任务
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)
 
