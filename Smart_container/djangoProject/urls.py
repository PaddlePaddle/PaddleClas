"""djangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from app01 import views
from django.conf import settings
from django.contrib import admin
from django.urls import include, path, re_path
#from tyadmin_api.views import AdminIndexView
from django.views.static import serve
from djangoProject import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login_in/', views.login_in),
    path('search/', views.search),
    path('record/', views.record),
    path('delete/', views.delete),
    path('replace/', views.replace),
    path('find/', views.find),
    path('reference/', views.reference),
    path('reference_client/',views.reference_client),
    re_path('media/(?P<path>.*)', serve, {"document_root": settings.MEDIA_ROOT}),
  #  re_path('^xadmin/.*', AdminIndexView.as_view()),
 #   path('api/xadmin1/', include('tyadmin_api.urls')),
]
