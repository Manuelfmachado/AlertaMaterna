<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlertaMaterna - Pitch Datos al Ecosistema 2025</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            overflow: auto;
        }

        .presentation {
            width: 100vw;
            height: 100vh;
            position: relative;
        }

        .slide {
            width: 100%;
            min-height: 100vh;
            position: absolute;
            top: 0;
            left: 0;
            display: none;
            padding: 60px;
            background: white;
            animation: slideIn 0.5s ease-out;
            overflow-y: auto;
        }

        .slide.active {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(100px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        h1 {
            font-size: 4em;
            color: #1a237e;
            margin-bottom: 30px;
            font-weight: 700;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }

        h2 {
            font-size: 2.5em;
            color: #e91e63;
            margin-bottom: 25px;
            font-weight: 600;
        }

        h3 {
            font-size: 1.8em;
            color: #424242;
            margin: 20px 0;
        }

        p, li {
            font-size: 1.4em;
            line-height: 1.6;
            color: #424242;
            margin: 10px 0;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }

        .highlight {
            color: #e53935;
            font-weight: bold;
            font-size: 1.2em;
        }

        .big-number {
            font-size: 5em;
            font-weight: 900;
            color: #e91e63;
            display: block;
            margin: 20px 0;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin: 30px 0;
        }

        .grid-4 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 30px;
            margin: 30px 0;
        }

        .card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .card h3 {
            color: white;
            font-size: 1.5em;
            margin-bottom: 15px;
        }

        .card p {
            color: white;
            font-size: 1.2em;
        }

        .funnel {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 30px 0;
        }

        .funnel-item {
            width: 100%;
            max-width: 800px;
            padding: 20px;
            margin: 10px 0;
            background: linear-gradient(90deg, #43a047 0%, #66bb6a 100%);
            color: white;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5em;
            font-weight: bold;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .funnel-item.excluded {
            background: linear-gradient(90deg, #e53935 0%, #ef5350 100%);
            max-width: 300px;
        }

        .navigation {
            position: fixed;
            bottom: 30px;
            right: 30px;
            display: flex;
            gap: 15px;
            z-index: 1000;
        }

        .nav-btn {
            background: #1a237e;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 1.2em;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }

        .nav-btn:hover {
            background: #e91e63;
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        }

        .slide-number {
            position: fixed;
            bottom: 30px;
            left: 30px;
            font-size: 1.5em;
            color: #1a237e;
            font-weight: bold;
        }

        .icon {
            font-size: 3em;
            margin-bottom: 15px;
        }

        .bar-chart {
            display: flex;
            align-items: flex-end;
            gap: 30px;
            height: 300px;
            margin: 40px 0;
        }

        .bar {
            flex: 1;
            background: linear-gradient(180deg, #e53935 0%, #ef5350 100%);
            border-radius: 10px 10px 0 0;
            position: relative;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            align-items: center;
            padding: 10px;
            color: white;
            font-weight: bold;
        }

        .bar-label {
            position: absolute;
            bottom: -40px;
            font-size: 1.2em;
            color: #424242;
        }

        .timeline {
            display: flex;
            justify-content: space-around;
            margin: 40px 0;
        }

        .timeline-item {
            text-align: center;
            flex: 1;
        }

        .timeline-circle {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            font-weight: bold;
            margin: 0 auto 15px;
        }

        .video-placeholder {
            width: 100%;
            max-width: 900px;
            height: 500px;
            background: #000;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 30px auto;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }

        .play-button {
            font-size: 6em;
            color: white;
            cursor: pointer;
            transition: transform 0.3s;
        }

        .play-button:hover {
            transform: scale(1.2);
        }

        ul {
            list-style: none;
            padding-left: 0;
        }

        ul li:before {
            content: "✓ ";
            color: #43a047;
            font-weight: bold;
            font-size: 1.5em;
            margin-right: 10px;
        }

        .emphasis {
            background: linear-gradient(135deg, #ffd54f 0%, #ffb300 100%);
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .logo-container {
            text-align: center;
            margin-bottom: 40px;
        }

        .logo-container img {
            max-width: 600px;
            width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <div class="presentation">
        <!-- SLIDE 1: GANCHO INICIAL -->
        <div class="slide active">
            <div class="logo-container">
                <h1 style="color: #e91e63; text-align: center;">🏥 AlertaMaterna</h1>
            </div>
            
            <h2 style="text-align: center; font-size: 3em; margin: 40px 0;">
                "162 bebés mueren por cada<br>1,000 nacimientos en Saravena"
            </h2>
            
            <div class="bar-chart">
                <div class="bar" style="height: 95%; background: linear-gradient(180deg, #e53935 0%, #c62828 100%);">
                    <span style="font-size: 2em;">162‰</span>
                    <span class="bar-label">Saravena</span>
                </div>
                <div class="bar" style="height: 5%; background: linear-gradient(180deg, #43a047 0%, #2e7d32 100%);">
                    <span style="font-size: 1.5em;">8‰</span>
                    <span class="bar-label">Colombia</span>
                </div>
            </div>
            
            <p style="text-align: center; font-size: 2em; margin-top: 50px;">
                Esto no es estadística.<br>
                <span class="highlight">Son 278 vidas perdidas en 2024.</span>
            </p>
        </div>

        <!-- SLIDE 2: EL PROBLEMA -->
        <div class="slide">
            <h1>El Problema a Resolver</h1>
            
            <div class="grid-4">
                <div class="card">
                    <div class="icon">🏥</div>
                    <h3>¿Qué?</h3>
                    <p>Crisis invisible en salud materno-infantil Orinoquía</p>
                </div>
                
                <div class="card">
                    <div class="icon">⚠️</div>
                    <h3>¿Por qué importa?</h3>
                    <p>137,780 nacimientos en zonas críticas</p>
                    <p><strong>49.7% muertes PREVENIBLES</strong></p>
                </div>
                
                <div class="card">
                    <div class="icon">🎯</div>
                    <h3>¿Cómo?</h3>
                    <p>5 años datos DANE<br>29 indicadores OMS<br>ML + medicina</p>
                </div>
                
                <div class="card">
                    <div class="icon">👥</div>
                    <h3>¿Quién sufre?</h3>
                    <p>• Madres rurales sin prenatal<br>• Bebés prematuros<br>• Autoridades sin datos</p>
                </div>
            </div>
        </div>

        <!-- SLIDE 3: QUIÉNES SOMOS -->
        <div class="slide">
            <h1>Equipo AlertaMaterna</h1>
            
            <div style="text-align: center; margin: 60px 0;">
                <h2 style="font-size: 2.5em;">🔬 Ciencia de Datos en Salud Pública</h2>
                <h2 style="font-size: 2.5em;">💻 Machine Learning & IA</h2>
                <h2 style="font-size: 2.5em;">📊 Visualización de Datos</h2>
            </div>
            
            <div class="emphasis">
                <h3 style="text-align: center; color: #1a237e;">Instituciones Colaboradoras</h3>
                <div style="display: flex; justify-content: space-around; margin-top: 30px;">
                    <div style="text-align: center;">
                        <p style="font-size: 2em;">🏛️</p>
                        <p><strong>DANE</strong><br>datos.gov.co</p>
                    </div>
                    <div style="text-align: center;">
                        <p style="font-size: 2em;">🏥</p>
                        <p><strong>Min. Salud</strong><br>REPS</p>
                    </div>
                    <div style="text-align: center;">
                        <p style="font-size: 2em;">🎓</p>
                        <p><strong>Academia</strong><br>Investigación</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- SLIDE 4: LA SOLUCIÓN -->
        <div class="slide">
            <h1>AlertaMaterna: La Solución</h1>
            
            <div class="grid-2" style="margin-top: 50px;">
                <div>
                    <h2>📊 DATOS</h2>
                    <p><strong>137,780</strong> nacimientos analizados (2020-2024)</p>
                    <p><strong>29</strong> indicadores OMS validados</p>
                    <p><strong>5</strong> departamentos</p>
                    <p><strong>55</strong> municipios</p>
                </div>
                
                <div>
                    <h2>🧠 ANÁLISIS ML</h2>
                    <p><strong>MODELO 1:</strong> Clasificación Híbrida</p>
                    <p>✓ 100% casos críticos detectados</p>
                    <p><strong>MODELO 2:</strong> XGBoost Predictivo</p>
                    <p>ROC-AUC: 0.77 | Accuracy: 87%</p>
                </div>
            </div>
            
            <div class="emphasis" style="margin-top: 40px;">
                <h2 style="text-align: center;">🚨 RESULTADOS</h2>
                <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                    <div style="text-align: center;">
                        <p class="big-number">63</p>
                        <p>Registros ALTO RIESGO (25%)</p>
                    </div>
                    <div style="text-align: center;">
                        <p class="big-number">40</p>
                        <p>Casos >50‰ detectados</p>
                    </div>
                    <div style="text-align: center;">
                        <p class="big-number">79%</p>
                        <p>Precision alertas</p>
                    </div>
                </div>
            </div>
            
            <h3 style="text-align: center; margin-top: 30px;">📱 Dashboard Interactivo - Decisiones en Tiempo Real</h3>
        </div>

        <!-- SLIDE 5: VIDEO DEMO -->
        <div class="slide">
            <h1 style="text-align: center;">Demostración</h1>
            
            <div class="video-placeholder">
                <div class="play-button">▶️</div>
            </div>
            
            <p style="text-align: center; font-size: 1.8em; margin-top: 30px;">
                <strong>Video 60 segundos:</strong><br>
                "De datos a alertas en tiempo real"
            </p>
            
            <div style="text-align: center; margin-top: 30px;">
                <p>00-10s: Mapa interactivo Orinoquía</p>
                <p>10-25s: Navegación dashboard</p>
                <p>25-40s: Caso Saravena (162‰)</p>
                <p>40-50s: Predictor en acción</p>
                <p>50-60s: Sistema desplegado</p>
            </div>
        </div>

        <!-- SLIDE 6: DATOS UTILIZADOS -->
        <div class="slide">
            <h1>100% Datos Abiertos Públicos</h1>
            
            <h2 style="text-align: center; margin: 30px 0;">📊 De 2.7M a 137K con Calidad</h2>
            
            <div class="funnel">
                <div class="funnel-item" style="max-width: 1000px; background: linear-gradient(90deg, #9e9e9e 0%, #757575 100%);">
                    2,789,391 nacimientos brutos<br>Orinoquía 2020-2024
                </div>
                
                <div style="margin: 20px 0; font-size: 2em;">⬇️ Filtro OMS: ≥10 nacimientos/año</div>
                
                <div class="funnel-item" style="max-width: 800px;">
                    ✅ 137,780 analizados<br>251 registros válidos (81%)
                </div>
                
                <div class="funnel-item excluded">
                    ❌ 59 registros excluidos (19%)
                </div>
            </div>
            
            <div class="grid-2" style="margin-top: 40px;">
                <div class="card">
                    <h3>🏥 REPS - Min. Salud</h3>
                    <p>142 instituciones salud</p>
                    <p>RIPS: Atenciones 2020-24</p>
                </div>
                
                <div class="card">
                    <h3>📍 DIVIPOLA</h3>
                    <p>55 municipios</p>
                    <p>5 departamentos</p>
                </div>
            </div>
            
            <p style="text-align: center; font-size: 1.5em; margin-top: 30px;">
                💡 "Preferimos 137K datos confiables que 2.7M con ruido"
            </p>
        </div>

        <!-- SLIDE 7: INNOVACIÓN -->
        <div class="slide">
            <h1>Innovación y Tecnología</h1>
            
            <div class="grid-2">
                <div class="card">
                    <h3>🔬 INNOVACIÓN 1: Sistema Híbrido Único</h3>
                    <p>Percentiles + Umbrales OMS = Detecta crisis aunque otros estén peor</p>
                    <p><strong>100% detección casos >50‰</strong></p>
                </div>
                
                <div class="card">
                    <h3>🧠 INNOVACIÓN 2: ML con Conocimiento Médico</h3>
                    <p>XGBoost + SMOTE + 29 features validadas por literatura</p>
                    <p><strong>No es "big data ciego"</strong></p>
                </div>
                
                <div class="card">
                    <h3>📊 INNOVACIÓN 3: Features Críticas Avanzadas</h3>
                    <p>• Mortalidad neonatal (0-7 días): Feature #1</p>
                    <p>• <strong>49.7% muertes PREVENIBLES</strong></p>
                    <p>• Índice fragilidad sistema</p>
                </div>
                
                <div class="card">
                    <h3>⚡ INNOVACIÓN 4: Predicción Preventiva</h3>
                    <p>Intervención ANTES de la tragedia</p>
                    <p><strong>ROC-AUC 0.77 (+9.2% vs baseline)</strong></p>
                </div>
            </div>
        </div>

        <!-- SLIDE 8: ANÁLISIS Y HALLAZGOS -->
        <div class="slide">
            <h1>Hallazgos Clave</h1>
            
            <div class="emphasis">
                <h2>📈 HALLAZGO 1: Retroceso 2024</h2>
                <div style="display: flex; justify-content: space-around; align-items: flex-end; height: 200px; margin: 30px 0;">
                    <div style="text-align: center;">
                        <div style="height: 120px; width: 60px; background: #43a047; border-radius: 5px;"></div>
                        <p style="margin-top: 10px;">2020<br>48.3‰</p>
                    </div>
                    <div style="text-align: center;">
                        <div style="height: 100px; width: 60px; background: #66bb6a; border-radius: 5px;"></div>
                        <p style="margin-top: 10px;">2021<br>42.1‰</p>
                    </div>
                    <div style="text-align: center;">
                        <div style="height: 85px; width: 60px; background: #81c784; border-radius: 5px;"></div>
                        <p style="margin-top: 10px;">2022<br>38.7‰</p>
                    </div>
                    <div style="text-align: center;">
                        <div style="height: 70px; width: 60px; background: #a5d6a7; border-radius: 5px;"></div>
                        <p style="margin-top: 10px;">2023<br>35.2‰</p>
                    </div>
                    <div style="text-align: center;">
                        <div style="height: 180px; width: 60px; background: #e53935; border-radius: 5px;"></div>
                        <p style="margin-top: 10px;">2024<br><strong>63.4‰</strong></p>
                    </div>
                </div>
                <p style="text-align: center; font-size: 1.5em;"><strong>+80% aumento en 2024 - Sistema alertó a tiempo</strong></p>
            </div>
            
            <div class="grid-2" style="margin-top: 30px;">
                <div>
                    <h3>🎯 HALLAZGO 2: Top 3 Predictores</h3>
                    <p>1️⃣ Mortalidad neonatal (24.17%)</p>
                    <p>2️⃣ Instituciones de salud (9.24%)</p>
                    <p>3️⃣ Mortalidad evitable (6.65%)</p>
                </div>
                
                <div>
                    <h3>🗺️ HALLAZGO 3: Geografía del Riesgo</h3>
                    <p><span class="highlight">Arauca:</span> 57% alto riesgo 🔴</p>
                    <p><span class="highlight">Vichada:</span> 100% alto riesgo 🔴🔴</p>
                    <p><span class="highlight">Meta:</span> 22% alto riesgo 🟡</p>
                </div>
            </div>
        </div>

        <!-- SLIDE 9: IMPACTO Y ESCALABILIDAD -->
        <div class="slide">
            <h1>Impacto y Escalabilidad</h1>
            
            <div class="emphasis">
                <h2>✅ IMPACTO HOY - ORINOQUÍA</h2>
                <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                    <div style="text-align: center;">
                        <p class="big-number">63</p>
                        <p>Registros alto riesgo identificados</p>
                    </div>
                    <div style="text-align: center;">
                        <p class="big-number">100%</p>
                        <p>Casos >50‰ detectados</p>
                    </div>
                    <div style="text-align: center;">
                        <p class="big-number">87%</p>
                        <p>Accuracy modelo</p>
                    </div>
                </div>
            </div>
            
            <h2 style="margin-top: 40px;">🚀 ESCALABILIDAD - FASES</h2>
            
            <div class="timeline">
                <div class="timeline-item">
                    <div class="timeline-circle">✅</div>
                    <h3>FASE 1</h3>
                    <p>Orinoquía (5 dptos)</p>
                    <p><strong>HOY</strong></p>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-circle" style="background: linear-gradient(135deg, #ffd54f 0%, #ffb300 100%);">📋</div>
                    <h3>FASE 2</h3>
                    <p>Amazonía + Pacífico</p>
                    <p><strong>6 meses</strong></p>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-circle" style="background: linear-gradient(135deg, #e91e63 0%, #c2185b 100%);">🚀</div>
                    <h3>FASE 3</h3>
                    <p>Nacional (32 dptos)</p>
                    <p><strong>12 meses</strong></p>
                </div>
            </div>
            
            <div class="card" style="margin-top: 30px;">
                <h3>💡 MODELO DE ADOPCIÓN</h3>
                <ul>
                    <li>Modelos exportables (.pkl)</li>
                    <li>API lista para integración</li>
                    <li>Dashboard autohospedable</li>
                    <li>Implementación: <strong>&lt;1 semana</strong></li>
                </ul>
            </div>
        </div>

        <!-- SLIDE 10: CIERRE -->
        <div class="slide" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <div style="text-align: center;">
                <h1 style="color: white; font-size: 5em; margin-bottom: 40px;">AlertaMaterna</h1>
                
                <h2 style="color: white; font-size: 3em; line-height: 1.4; margin: 50px 0;">
                    "Hoy los datos predicen.<br>
                    Mañana salvan vidas."
                </h2>
                
                <div style="background: rgba(255,255,255,0.2); padding: 40px; border-radius: 20px; margin: 50px 0; backdrop-filter: blur(10px);">
                    <p style="font-size: 2em; line-height: 1.6; color: white;">
                        AlertaMaterna puede prevenir la muerte de<br>
                        <strong style="font-size: 1.5em; color: #ffd54f;">40 bebés este año</strong> solo en Orinoquía.
                    </p>
                    <p style="font-size: 1.8em; margin-top: 30px; color: white;">
                        Cada alerta temprana es una vida salvada.<br>
                        Cada predicción correcta es una intervención a tiempo.
                    </p>
                </div>
                
                <div style="margin-top: 60px; font-size: 1.5em;">
                    <p style="color: white;">🔗 Demo en vivo | 📂 Código abierto | 📧 Contacto</p>
                    <p style="color: #ffd54f; margin-top: 20px; font-size: 1.3em;"><strong>Datos que salvan vidas</strong></p>
                </div>
            </div>
        </div>
    </div>

    <div class="navigation">
        <button class="nav-btn" onclick="previousSlide()">◀ Anterior</button>
        <button class="nav-btn" onclick="nextSlide()">Siguiente ▶</button>
    </div>

    <div class="slide-number">
        <span id="current-slide">1</span> / 10
    </div>

    <script>
        let currentSlide = 0;
        const slides = document.querySelectorAll('.slide');
        const totalSlides = slides.length;

        function showSlide(n) {
            slides[currentSlide].classList.remove('active');
            currentSlide = (n + totalSlides) % totalSlides;
            slides[currentSlide].classList.add('active');
            document.getElementById('current-slide').textContent = currentSlide + 1;
        }

        function nextSlide() {
            showSlide(currentSlide + 1);
        }

        function previousSlide() {
            showSlide(currentSlide - 1);
        }

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowRight' || e.key === ' ') {
                nextSlide();
            } else if (e.key === 'ArrowLeft') {
                previousSlide();
            }
        });

        // Click navigation
        document.addEventListener('click', function(e) {
            if (!e.target.classList.contains('nav-btn')) {
                if (e.clientX > window.innerWidth / 2) {
                    nextSlide();
                } else {
                    previousSlide();
                }
            }
        });
    </script>
</body>
</html>