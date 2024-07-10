<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload CSV</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <form id="upload-form" enctype="multipart/form-data">
        <input type="file" id="file-input" name="file" />
        <button type="submit">Upload</button>
    </form>

    <div id="chart"></div>

    <script>
        document.getElementById('upload-form').addEventListener('submit', async function(event) {
            event.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('file-input');
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('http://localhost:8000/api/lda_results/', {
            method: 'POST',
            body: formData,
            credentials: 'include',
            headers: {
                'X-CSRFToken': getCookie('csrftoken')
            }
        });


                if (!response.ok) {
                    throw new Error('Error al cargar el archivo');
                }

                const data = await response.json();
                console.log(data);  // Verifica los datos recibidos en la consola

                // Preparar datos para el gráfico
                const topics = data.map(d => d.name_topic);
                const scores = data.map(d => parseFloat(d.SCORE.replace('%', '')));  // Convertir SCORE a número

                // Configurar traza y diseño del gráfico con Plotly.js
                const trace = {
                    x: topics,
                    y: scores,
                    type: 'bar'
                };

                const layout = {
                    title: 'Resultados del Análisis LDA',
                    xaxis: { title: 'Temas' },
                    yaxis: { title: 'Puntuación (%)' }
                };

                // Crear el gráfico con Plotly.js en el contenedor 'chart'
                Plotly.newPlot('chart', [trace], layout);

            } catch (error) {
                console.error('Error:', error);
            }
        });

        // Función para obtener el valor del token CSRF
        function getCookie(name) {
            let cookieValue = null;
            if (document.cookie && document.cookie !== '') {
                const cookies = document.cookie.split(';');
                for (let i = 0; i < cookies.length; i++) {
                    const cookie = cookies[i].trim();
                    if (cookie.substring(0, name.length + 1) === (name + '=')) {
                        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                        break;
                    }
                }
            }
            return cookieValue;
        }
    </script>
</body>
</html>
