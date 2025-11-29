#AlertaMaterna:SistemadeClasificacióndeRiesgoObstétricoyPrediccióndeMortalidadInfantilenlaRegiónOrinoquía

![AlertaMaternaBanner](alertamaterna_banner.png)

[![Python3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

##📋Descripción

**AlertaMaterna**esunsistemadeinteligenciaartificialqueidentificamunicipiosdelaregiónOrinoquíaconaltoriesgodemortalidadmaterno-infantil,utilizandodatosoficialesdelDANEdelperiodo2020-2024.

Elsistemaanaliza**24indicadoresdesalud**(atenciónprenatal,bajopesoalnacer,prematuridad,accesoaservicios)paraclasificar**55municipios**endoscategorías:**ALTORIESGO**o**BAJORIESGO**,ademásdepredecirlaprobabilidaddemortalidadinfantilencadamunicipio.

###🎯Objetivos

1.**Clasificar**municipiossegúnsunivelderiesgoobstétrico
2.**Predecir**probabilidaddemortalidadinfantil(<1año)
3.**Priorizar**intervencionesensaludpública
4.**Monitorear**evolucióntemporaldeindicadorescríticos

###🌍RegióndeAnálisis

**Orinoquíacolombiana**:Meta,Arauca,Casanare,GuaviareyVichada(55municipios,310registrosmunicipio-año2020-2024)

##🚀CaracterísticasPrincipales

-✅**Sistemahíbridodeclasificación**:Combinapercentilesestadísticos+umbralescríticosOMS/PAHO
-✅**100%dedeteccióndecasoscríticos**:Identificatodoslosmunicipiosconmortalidad>50‰
-✅**ModelopredictivoXGBoost**:ROC-AUC0.71,priorizasensibilidadsobreespecificidad
-✅**Dashboardinteractivo**:VisualizacionesentiemporealconStreamlityPlotly
-✅**Basadoendatosoficiales**:DANE-453,901nacimientosy21,250defuncionesfetales(2024)
-✅**Códigoabierto**:DisponibleenGitHubbajolicenciaMIT

##📊ResultadosPrincipales

|Métrica|Valor|
|---------|-------|
|**Registrosanalizados**|310municipio-año(2020-2024)|
|**Registrosválidos**|251(≥10nacimientos)|
|**Municipiosaltoriesgo**|53(21.1%)|
|**Casoscríticosdetectados**|40(mortalidad>50‰)-100%sensibilidad|
|**ROC-AUCModeloPredictivo**|0.71|
|**Accuracy**|66%|

##🧠ModelosImplementados

###Modelo1:ClasificacióndeRiesgoObstétrico

**Sistemahíbridodepuntaje(0-8puntos)**:

Unmunicipioesclasificadocomo**ALTORIESGO**sicumple:
-≥3puntosencriteriosdepercentil75(mortalidadfetal,atenciónprenatal,bajopeso,prematuridad,cesáreas,presiónobstétrica)
-**O**mortalidadfetal>50‰(clasificaciónautomática,+3puntos)

**Justificacióndelumbral50‰**:
-TasaglobalOMS:5‰
-Latinoamérica:10-15‰
-**50‰=10xlatasanormal**→crisisdesaludpública(PAHO2019)

###Modelo2:PrediccióndeMortalidadInfantil

**Algoritmo**:XGBoostconSMOTE(balanceodeclases)

**Features**:20variablessociosanitarias(excluyendoidentificadoresytargets)

**Performance**:
-ROC-AUC:**0.71**
-Recall(altamortalidad):**62%**(priorizadeteccióndecasoscríticos)
-Precision(bajamortalidad):**84%**

**Top3featuresmásimportantes**:
1.APGARbajopromedio(18.7%)
2.Porcentajebajopesoalnacer(7.4%)
3.Consultasprenatalespromedio(7.2%)


##🛠️InstalaciónyUso

###RequisitosPrevios

-Python3.8osuperior
-pip(gestordepaquetesPython)

###Instalación

```bash
#1.Clonarelrepositorio
gitclonehttps://github.com/Manuelfmachado/AlertaMaterna.git
cdAlertaMaterna

#2.Instalardependencias
pipinstall-rrequirements.txt
```

###EjecuciónRápida

**Opción1:Ejecutardashboarddirectamente**(modelosyaentrenados)

```bash
streamlitrunapp_simple.py
```

Eldashboardseabriráen`http://localhost:8501`

**Opción2:Ejecutarpipelinecompleto**(reentrenarmodelos)

```bash
#Paso1:Generarfeatures
cdsrc
pythonfeatures.py

#Paso2:Entrenarmodelos
pythontrain_model.py

#Paso3:Lanzardashboard
cd..
streamlitrunapp_simple.py
```

##📁EstructuradelProyecto

```
AlertaMaterna/
├──data/
│├──raw/#DatosoriginalesDANE
││├──BD-EEVV-Nacimientos-*.csv
││├──BD-EEVV-Defunciones*.csv
││└──codigos_*.csv
│└──processed/#Datosprocesados
│├──features_municipio_anio.csv#310registroscon24features
│└──features_alerta_materna.csv#Contargetsyclasificación
├──src/
│├──features.py#Generaciónde24features
│└──train_model.py#Entrenamientodemodelos
├──models/#Modelosentrenados(.pkl)
│├──modelo_mortalidad_xgb.pkl
│├──scaler_mortalidad.pkl
│├──umbral_mortalidad.pkl
│└──umbral_riesgo_obstetrico.pkl
├──app_simple.py#DashboardStreamlit
├──requirements.txt#DependenciasPython
├──DOCUMENTACION_TECNICA.md#Justificacióncientífica(60+páginas)
├──alertamaterna_banner.png#Bannerdelproyecto
└──README.md#Estearchivo
```

##📈UsodelDashboard

Eldashboardtiene**2pestañasprincipales**:

###1.📊PanoramaGeneral

-**Indicadoresprincipales**:Municipiosanalizados,altoriesgo,nacimientos,mortalidadfetal
-**Distribuciónderiesgo**:Gráficocomparativopordepartamento
-**Indicadoresclave**:Promediosdemortalidad,atenciónprenatal,bajopeso
-**Top10municipiosaltoriesgo**:Rankingconpuntajesdetallados

###2.🎯PredictordeRiesgo

**Herramientainteractiva**paraevaluarmunicipios:

1.Ingresa20indicadoresdelmunicipio(nacimientos,atenciónprenatal,APGAR,etc.)
2.Elsistemacalculaprobabilidaddealtamortalidad
3.Visualizaciónderiesgo:
-🟢**Verde(<30%)**:Riesgobajo
-🟡**Amarillo(30-60%)**:Riesgomedio
-🔴**Rojo(>60%)**:Riesgoalto

##🔢FeaturesGeneradas(24variables)

###📊Demográficas(5)
-`total_nacimientos`:Totaldenacimientosregistrados
-`edad_materna_promedio`:Edadpromediodemadres
-`pct_madres_adolescentes`:%madres<18años
-`pct_madres_edad_avanzada`:%madres≥35años
-`pct_bajo_nivel_educativo`:%madresconeducaciónbásica

###🏥Clínicas(8)
-`total_defunciones`:Defuncionesinfantiles(<1año)
-`defunciones_fetales`:Muertesfetales(≥22semanas)
-`tasa_mortalidad_fetal`:Defuncionesfetalespor1,000nacimientos
-`tasa_mortalidad_infantil`:Defunciones<1añopor1,000nacimientos
-`pct_bajo_peso`:%nacimientos<2,500g
-`pct_embarazo_multiple`:%embarazosmúltiples
-`pct_cesarea`:%partosporcesárea
-`pct_prematuro`:%nacimientos<37semanas
-`apgar_bajo_promedio`:PromedioAPGAR<7

###🏢Institucionales(3)
-`num_instituciones`:Númerodeinstitucionesdesalud
-`presion_obstetrica`:Nacimientosporinstitución
-`pct_instituciones_publicas`:%institucionespúblicas

###💰Socioeconómicas(3)
-`pct_sin_seguridad_social`:%sinafiliaciónasalud
-`pct_regimen_subsidiado`:%enrégimensubsidiado
-`pct_area_rural`:%nacimientosenzonarural

###🩺AtenciónPrenatal(2)
-`pct_sin_control_prenatal`:%sincontrolprenatal
-`consultas_promedio`:Promediodeconsultasprenatales

###🎯Targets(3)
-`riesgo_obstetrico`:ALTO/BAJO(Modelo1)
-`puntos_riesgo`:Puntaje0-8(Modelo1)
-`alta_mortalidad`:0/1(Modelo2)

##📚MetodologíaCientífica

###JustificacióndeParámetros

Todoslosparámetrosestánrespaldadosporliteraturacientífica.Ver**`DOCUMENTACION_TECNICA.md`**(60+páginas)con:

-16referenciasbibliográficas(OMS,PAHO,estudiosepidemiológicos)
-Justificacióndelumbral50‰(10xtasanormal)
-Análisisdesensibilidaddelumbral≥3puntos
-ExplicacióndeSMOTEparabalanceodeclases
-ValidacióndehiperparámetrosXGBoost
-Coherenciaconconocimientodeldominiomédico

###FiltradodeDatos

-**Umbralmínimo**:10nacimientos/añopormunicipio
-**Justificación**:Evitarvarianzaextremapornúmerospequeños
-**Resultado**:310registros→251válidos(81%)

###UmbralesCríticos

|Indicador|Umbral|Justificación|
|-----------|--------|---------------|
|Mortalidadfetalcrítica|>50‰|10xtasanormal(OMS:5‰)|
|Sinatenciónprenatal|>50%|Fallasistémica(OMSrecomienda100%)|
|Clasificaciónaltoriesgo|≥3puntos|Detecta100%casoscríticos,21%clasificados|
|Targetmortalidadinfantil|>Percentil75|6.42‰(50%sobrepromedionacional~4‰)|


##🎯CasosdeUso

1.**Planificaciónestratégicaensaludpública**:Identificarmunicipiosquerequiereninversiónprioritaria
2.**Asignacióneficientederecursos**:Priorizardepartamentossegúnnivelderiesgo
3.**Monitoreotemporal**:Evaluarevolucióndeindicadorescríticos(2020-2024)
4.**Análisisdeimpacto**:Simularefectosdemejoraseninfraestructurasanitaria
5.**Sistemadealertastempranas**:Detectardeteriorodeindicadoresentiemporeal
6.**Evaluacióndepolíticaspúblicas**:Medirefectividaddeintervenciones

##🏆ResultadosDestacados

###PorDepartamento(2024)

|Departamento|Municipios|AltoRiesgo|%AltoRiesgo|MortalidadPromedio|
|--------------|------------|-------------|---------------|---------------------|
|**Vichada**|2|2|100%|86.5‰|
|**Arauca**|7|4|57%|99.6‰|
|**Guaviare**|4|1|25%|85.2‰|
|**Meta**|18|4|22%|25.1‰|
|**Casanare**|14|2|14%|24.8‰|

###MunicipiosCríticos(Mortalidad>50‰,2024)

|Municipio|Departamento|Nacimientos|Defunciones|Mortalidad|Estado|
|-----------|--------------|-------------|-------------|------------|--------|
|Saravena|Arauca|1,716|278|162.0‰|🔴CRÍTICO|
|PuertoRondón|Arauca|21|2|95.2‰|🔴CRÍTICO|
|PuertoCarreño|Vichada|513|47|91.6‰|🔴CRÍTICO|
|Arauca|Arauca|1,188|107|90.1‰|🔴CRÍTICO|
|SanJosédelGuaviare|Guaviare|1,009|86|85.2‰|🔴CRÍTICO|

**Totalpoblaciónafectada**:4,811nacimientosenmunicipioscríticos(38%deltotal2024)

##🔧TecnologíasUtilizadas

###MachineLearning
-**XGBoost**1.7+:Gradientboostingoptimizado
-**Scikit-learn**1.3+:Preprocessing,métricas,validación
-**Imbalanced-learn**0.11+:SMOTEparabalanceodeclases

###AnálisisdeDatos
-**Pandas**2.0+:Manipulaciónyanálisisdedatos
-**NumPy**1.24+:Operacionesnuméricas

###Visualización
-**Streamlit**1.28+:Dashboardwebinteractivo
-**Plotly**5.11+:Gráficosinteractivos
-**Matplotlib**3.7+:Visualizacionesestáticas

##📖DocumentaciónAdicional

-**DOCUMENTACION_TECNICA.md**:Justificacióncientíficacompleta(60+páginas,16referencias)
-Marcoteóricoycontextoepidemiológico
-Justificacióndecadaparámetroconliteraturamédica
-Análisisdesensibilidaddeumbrales
-Validaciónycoherenciaderesultados
-Limitacionesytrabajofuturo

##🤝Contribuciones

Lascontribucionessonbienvenidas.Porfavor:

1.Forkelrepositorio
2.Creaunaramaparatufeature(`gitcheckout-bfeature/AmazingFeature`)
3.Committuscambios(`gitcommit-m'AddsomeAmazingFeature'`)
4.Pushalarama(`gitpushoriginfeature/AmazingFeature`)
5.AbreunPullRequest

##📄Licencia

Esteproyectoesde**códigoabierto**bajolicenciaMITparausoensaludpública.

##👥Autores

**ProyectoAlertaMaterna**-SistemadeClasificacióndeRiesgoObstétricoyPrediccióndeMortalidadInfantil

##🙏Agradecimientos

Datosproporcionadospor:
-**DANE**(DepartamentoAdministrativoNacionaldeEstadística)
-**MinisteriodeSaludyProtecciónSocialdeColombia**
-Registrosvitalesdenacimientosydefunciones(2020-2024)

Referenciascientíficas:
-**OMS**(OrganizaciónMundialdelaSalud)
-**PAHO**(PanAmericanHealthOrganization)
-**UNICEF**-Estudiossobresaludmaterno-infantil

##📞Contacto

Parapreguntas,sugerenciasocolaboraciones:
-GitHub:[@Manuelfmachado](https://github.com/Manuelfmachado)
-Repositorio:[AlertaMaterna](https://github.com/Manuelfmachado/AlertaMaterna)

##📌CómoCitar

Siutilizasesteproyectoentuinvestigaciónotrabajo,porfavorcítalocomo:

```
AlertaMaterna(2025).SistemadeClasificacióndeRiesgoObstétricoyPredicción
deMortalidadInfantilenlaRegiónOrinoquía.GitHub:Manuelfmachado/AlertaMaterna
```

---

<divalign="center">

**AlertaMaternav1.0**|2024-2025
*AnticipacióndelriesgoobstétricoenlaregiónOrinoquía*

[🏠Inicio](#alertamaterna-sistema-de-clasificación-de-riesgo-obstétrico-y-predicción-de-mortalidad-infantil-en-la-región-orinoquía)•[📊Dashboard](#-uso-del-dashboard)•[📚Documentación](#-documentación-adicional)•[🤝Contribuir](#-contribuciones)

</div>
