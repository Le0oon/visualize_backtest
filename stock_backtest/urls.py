from django.urls import path
from . import views


app_name = 'stock_backtest'

urlpatterns = [
    path('', views.login, name='login'),
    path('backtest/', views.backtest, name='backtest'),
    path('write_factor/', views.write_factor, name='write_factor'),
    path('info/', views.info, name='info'),
    # path('pending/<str:factor_name>',views.pending, name='pending_evaluation'),
    path('evaluation_result/<str:factor_name>', views.evaluation_result, name='evaluation_result'),
    path('evaluating/', views.evaluating, name='evaluating'),
    path('reg/', views.reg, name='check_usr'),
    
    # path('grading/<int:grader_id>/',views.gradeEmployee, name = 'grading'),
    # path('reg/', views.reg, name='write_grader'),
    # path('thanks/<int:grader_id>',views.thanks,name='thanks')

]