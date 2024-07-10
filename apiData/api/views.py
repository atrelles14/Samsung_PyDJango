# apiData/api/views.py
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from .inicio import inicio
import pandas as pd
import json

class FileUploadView(View):
    @csrf_exempt
    def post(self, request, *args, **kwargs):
        if request.method == 'POST' and request.FILES.get('file'):
            file = request.FILES['file']
            
            with open('temp_file.csv', 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            
            datos_final = inicio('temp_file.csv', 4, 8)
            
            if not isinstance(datos_final, pd.DataFrame):
                return JsonResponse({'error': 'Processing failed'}, status=500)
            
            data = datos_final[['TOPIC', 'name_topic', 'SCORE']].to_dict('records')
            
            return JsonResponse(data, safe=False)
        
        return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def lda_results(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        
        with open('temp_file.csv', 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        datos_final = inicio('temp_file.csv', 4, 8)
        
        if not isinstance(datos_final, pd.DataFrame):
            return JsonResponse({'error': 'Processing failed'}, status=500)
        
        data = datos_final[['TOPIC', 'name_topic', 'SCORE']].to_dict('records')
        
        return JsonResponse(data, safe=False)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)