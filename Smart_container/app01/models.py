from django.db import models


# Create your models here.
class TContainer(models.Model):
    number = models.IntegerField(primary_key=True,verbose_name="number")
    openid = models.CharField(max_length=255, blank=True, null=True,verbose_name="openid")
    container_name = models.CharField(max_length=255, blank=True, null=True,verbose_name="name")
    container_price = models.CharField(max_length=255, blank=True, null=True,verbose_name="price")
    picture_address = models.CharField(max_length=255, blank=True, null=True,verbose_name="address")
    stock = models.IntegerField(blank=True, null=True,verbose_name="stock")

    class Meta:
        managed = True
        verbose_name = "container"
       	verbose_name_plural = verbose_name
       	db_table = 't_container'
            
    def __str__(self):
        return self.openid


class TUser(models.Model):
    openid = models.CharField(primary_key=True, max_length=255,verbose_name="openid")
    nickname = models.CharField(max_length=255, blank=True, null=True,verbose_name="nickname")
    session_key = models.CharField(max_length=255, blank=True, null=True,verbose_name="session_key")

    class Meta:
        managed = True
        verbose_name = "user"
       	verbose_name_plural = verbose_name
       	db_table = 't_user'
            
    def __str__(self):
        return self.openid
