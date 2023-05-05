from django.urls import path
from . import views

urlpatterns = [
    path('index', views.func_index, name='index_page'),
    path('shrimp_trend', views.func_shrimp_trend, name='shrimp_trend_page'),
    path('results', views.result_plots, name='results_page'),
    path('ypr_shrimp_details', views.YPR_shrimp_details, name='shrimps_details_page'),
    path('ypr_seaweed_details', views.YPR_seaweed_details, name='seaweeds_details_page'),
    path('shrimp_or_seaweed', views.func_shrimp_seaweed, name='shrimps_seaweeds_sel_page'),
    path('ypr_results', views.func_YPR_results, name="ypr_results_page")
]
