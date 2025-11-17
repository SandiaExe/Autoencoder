// Espera a que todo el HTML esté cargado antes de ejecutar el script
document.addEventListener("DOMContentLoaded", () => {
    
    // --- 1. Configuración de los Canvas ---
    const drawCanvas = document.getElementById('drawing-canvas');
    const ctx = drawCanvas.getContext('2d', { willReadFrequently: true }); // Optimización
    const reconCanvas = document.getElementById('reconstruction-canvas');
    const reconCtx = reconCanvas.getContext('2d');

    // Estilo del "lápiz" para dibujar
    ctx.strokeStyle = "white"; // Color del lápiz
    ctx.fillStyle = "white";
    ctx.lineWidth = 20;        // Grosor del lápiz (importante para que se parezca a MNIST)
    ctx.lineCap = "round";     // Puntas redondeadas
    ctx.lineJoin = "round";
    
    let drawing = false;

    // --- 2. Funciones de Dibujo ---
    function startDrawing(e) {
        drawing = true;
        draw(e); // Dibuja un punto al hacer clic
    }

    function stopDrawing() {
        drawing = false;
        ctx.beginPath(); // Levanta el lápiz
    }

    function draw(e) {
        if (!drawing) return;

        // Obtener coordenadas relativas al canvas
        const rect = drawCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }
    
    // Eventos del mouse para dibujar
    drawCanvas.addEventListener('mousedown', startDrawing);
    drawCanvas.addEventListener('mouseup', stopDrawing);
    drawCanvas.addEventListener('mouseout', stopDrawing); // Detener si el mouse sale
    drawCanvas.addEventListener('mousemove', draw);

    // --- 3. Lógica del Modelo ---
    let model;
    
    // Función asíncrona para cargar el modelo
    async function loadModel() {
        console.log("Cargando modelo...");
        try {
            // Ruta al 'model.json' que generaste
            model = await tf.loadLayersModel('model.json');
            console.log("¡Modelo cargado!");
            document.getElementById('reconstruct-btn').disabled = false;
            document.getElementById('reconstruct-btn').innerText = "Reconstruir";
        } catch (err) {
            console.error(err);
            alert("Error al cargar el modelo. Revisa la consola (F12).");
        }
    }
    
    // Deshabilitar el botón hasta que el modelo esté cargado
    document.getElementById('reconstruct-btn').disabled = true;
    document.getElementById('reconstruct-btn').innerText = "Cargando modelo...";
    loadModel(); // Inicia la carga del modelo

    // Función para preprocesar y predecir
    async function reconstruct() {
        if (!model) {
            alert("El modelo no está cargado todavía.");
            return;
        }

        // --- A. Preprocesamiento de la Imagen ---
        
        // 1. Crear un tensor desde el canvas de 280x280
        // Usamos '1' para obtener solo 1 canal (escala de grises)
        let tensor = tf.browser.fromPixels(drawCanvas, 1);

        // 2. Redimensionar la imagen a 28x28 (el tamaño de entrada de tu modelo)
        tensor = tf.image.resizeBilinear(tensor, [28, 28]);

        // 3. Normalizar los datos (rango de píxeles 0-255 -> 0-1)
        tensor = tensor.div(255.0);
        
        // 4. Ajustar la forma (shape) a [1, 28, 28] 
        // Tu modelo Keras tenía un Reshape((28, 28)) al final,
        // pero la entrada al autoencoder completo sigue siendo (None, 28, 28, 1)
        // Y la salida del autoencoder es (None, 28, 28).
        // PERO, el modelo guardado (stacked_auotencoder) espera la entrada original: [batch, alto, ancho, canales]
        const inputTensor = tensor.reshape([1, 28, 28, 1]); 
        
        // --- B. Realizar la Predicción ---
        console.log("Realizando predicción...");
        const prediction = model.predict(inputTensor);
        
        // --- C. Post-procesamiento y Visualización ---
        
        // La salida del modelo es [1, 28, 28]. 
        // 1. La normalizamos de 0-1 a 0-255
        // 2. Le quitamos la dimensión del batch (squeeze)
        // 3. La convertimos a tipo 'int32' para poder dibujarla
        const outputTensor = prediction.squeeze().mul(255).asType('int32');
        
        // 4. Convertir el tensor de salida a datos de imagen
        const outputData = await outputTensor.data();
        const imageData = new ImageData(28, 28);
        
        // Rellenamos el array de píxeles (R, G, B, A)
        for (let i = 0; i < 28 * 28; i++) {
            const val = outputData[i];
            imageData.data[i * 4 + 0] = val; // R
            imageData.data[i * 4 + 1] = val; // G
            imageData.data[i * 4 + 2] = val; // B
            imageData.data[i * 4 + 3] = 255; // Alpha (opacidad total)
        }

        // 5. Dibujar la imagen de 28x28 en el canvas de reconstrucción (escalándola a 280x280)
        
        // Creamos un canvas temporal de 28x28
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imageData, 0, 0);

        // Limpiamos el canvas de reconstrucción antes de dibujar
        reconCtx.clearRect(0, 0, reconCanvas.width, reconCanvas.height);
        
        // Deshabilitamos el suavizado de imagen para que se vean los píxeles
        reconCtx.imageSmoothingEnabled = false; 
        
        // Dibujamos el canvas temporal (28x28) en el canvas grande (280x280)
        reconCtx.drawImage(tempCanvas, 0, 0, 280, 280);

        // --- D. Limpiar Tensores de Memoria ---
        tensor.dispose();
        inputTensor.dispose();
        prediction.dispose();
        outputTensor.dispose();
        console.log("Reconstrucción completa.");
    }

    // --- 4. Conectar Botones ---
    document.getElementById('reconstruct-btn').addEventListener('click', reconstruct);
    
    document.getElementById('clear-btn').addEventListener('click', () => {
        // Limpiar ambos lienzos
        ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        reconCtx.clearRect(0, 0, reconCanvas.width, reconCanvas.height);
    });

});
