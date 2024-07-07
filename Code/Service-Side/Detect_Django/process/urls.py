from django.urls import path
from .views import Germinate,GetInfo,DeleteItemView,ClearItemView,Mango,RecordTime

urlpatterns = [
    # 其他 URL 模式...
    path('germinate/', Germinate.as_view(), name='upload_germination'),
    path('mango/', Mango.as_view(), name='upload_mango'),
    path('info/',GetInfo.as_view(),name='upload_getInfo'),
    path('recordtime/',RecordTime.as_view(),name='upload_time'),
    path('delete/', DeleteItemView.as_view(), name='delete_item'),  # 设置删除项的URL
    path('clear/', ClearItemView.as_view(), name='clear_item'),
    # 其他 URL 模式...
]