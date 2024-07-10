# apiData/apps/models.py
from django.db import models

class Opinion(models.Model):
    texto = models.TextField()

    def __str__(self):
        return self.texto

class ResultadoLDA(models.Model):
    texto_procesado = models.TextField()
    resultado = models.CharField(max_length=100)
    fecha_creacion = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.resultado

class Observacion(models.Model):
    sede = models.CharField(max_length=50)
    facultad = models.CharField(max_length=50)
    carrera = models.CharField(max_length=50)
    observaciones = models.TextField()
    texto_token = models.JSONField()
    lemma = models.JSONField()
    score = models.FloatField()
    topic = models.IntegerField()
    name_topic = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.sede} - {self.facultad} - {self.carrera}"
