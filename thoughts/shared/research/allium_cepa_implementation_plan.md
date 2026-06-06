# Allium Cepa — ML Pipeline: Plan de Implementación

> Documento de arquitectura funcional para guiar la implementación con Claude Code.
> Cubre todos los componentes, sus responsabilidades, integraciones y comportamiento por entorno.
>
> **v2** — Actualizado para reflejar el repositorio existente (CLAUDE.md), las decisiones
> de storage HuggingFace vs S3, y la eliminación de VAE/ControlNet/Diffusers del scope.

---

## Índice

1. [Visión general del sistema](#1-visión-general-del-sistema)
2. [Qué ya existe vs qué se construye](#2-qué-ya-existe-vs-qué-se-construye)
3. [Stack tecnológico por capa](#3-stack-tecnológico-por-capa)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Componentes funcionales](#5-componentes-funcionales)
6. [DAGs de Airflow](#6-dags-de-airflow)
7. [Pipeline de DVC](#7-pipeline-de-dvc)
8. [Flujo de datos end-to-end](#8-flujo-de-datos-end-to-end)
9. [Entorno de desarrollo](#9-entorno-de-desarrollo)
10. [Entorno productivo](#10-entorno-productivo)
11. [Model Validation Gate](#11-model-validation-gate)
12. [Active Learning Loop](#12-active-learning-loop)
13. [Matriz de integración entre componentes](#13-matriz-de-integración-entre-componentes)

---

## 1. Visión general del sistema

El sistema implementa un pipeline de **Continuous Training con Human-in-the-Loop** para clasificación celular toxicológica en imágenes de microscopía de *Allium cepa* (UTN/INA).

Pipeline de inferencia de dos etapas (ya implementado):
1. **Detección**: modelo YOLO detecta células individuales en imágenes completas de microscopio.
2. **Clasificación**: backbone EfficientNet/ResNet/VGG clasifica cada crop como *mitosis* o *no_mitosis*.

Ambas etapas tienen calibración post-hoc ya implementada (vector scaling + isotonic regression).

El sistema nuevo agrega tres capas sobre este núcleo existente:

- **Ingesta de datos**: desde Zooniverse (clasificaciones de expertos) y desde imágenes crudas vía Google Drive.
- **Training reproducible y automático**: extensión del pipeline DVC existente con stages de training, evaluación, y validación. Re-entrenamiento disparado por datos nuevos validados.
- **Serving y feedback loop**: interfaz web para usuarios no técnicos que alimenta el sistema de etiquetado con imágenes nunca vistas.

```
┌──────────────────────────────────────────────────────────────────────┐
│                          FUENTES DE DATOS                            │
│  HuggingFace (ground truth)  Zooniverse (expertos)  Drive (raw)      │
└──────────┬───────────────────────────┬──────────────────┬────────────┘
           │                           │                  │
           ▼                           ▼                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      AIRFLOW — ORQUESTACIÓN                          │
│  DAG A: zooniverse_ingest    DAG B: raw_image_ingest                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
           ┌───────────────────┼──────────────────────┐
           ▼                   ▼                      ▼
  HuggingFace             S3 / MinIO              S3 / MinIO
  GIAR-UTN/               dataset/labeled/        dataset/review/
  allium-cepa-dataset     (validado)              (pendiente expertos)
  (ground truth curado)
           │
           └───────────────────┐
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 DVC PIPELINE — TRAINING REPRODUCIBLE                 │
│  download_dataset → coco_to_yolo → prepare_crops →                   │
│  train_detector → calibrate_detector →                               │
│  train_classifier → calibrate_classifier →                           │
│  evaluate → validate_model                                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│               MLFLOW — EXPERIMENT TRACKING & REGISTRY                │
│  Runs / Métricas / Model Registry (Staging → Production)             │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        SERVING — DOCKER                              │
│  FastAPI (AlliumCepaModel wrapper)    Streamlit (UI usuarios)        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
         Alta confianza               Baja confianza
         → S3 ui_logs/ (candidatos)   → Zooniverse (nueva tarea)
```

---

## 2. Qué ya existe vs qué se construye

Esta sección es crítica para Claude Code: **no reimplementar lo que ya existe**.

### Ya implementado — no tocar

| Componente | Ubicación | Estado |
|---|---|---|
| Pipeline DVC (stages de data prep) | `dvc.yaml` | Completo: `download_dataset`, `coco_to_yolo`, `prepare_crops` |
| Sistema de configs Pydantic v2 | `src/allium_cepa_classifier/config/` | Completo. Todos los configs nuevos extienden `BaseConfig` |
| Modelo de inferencia two-stage | `src/allium_cepa_classifier/data_models/allium_cepa_model.py` | Completo. `AlliumCepaModel.predict()` es la interfaz de inferencia |
| Calibración del clasificador | `src/allium_cepa_classifier/training/calibrator.py` | Completo (vector scaling) |
| Calibración del detector | `src/allium_cepa_classifier/training/detector_calibrator.py` | Completo (isotonic regression) |
| Scripts de training | `scripts/train_classifier.py`, `scripts/train_detector.py` | Completos. Los DVC stages los invocan como subprocesos |
| Script de calibración standalone | `scripts/calibrate_classifier.py`, `scripts/calibrate_detector.py` | Completos |
| TensorBoard integration | En cada trainer | Completo |
| Sistema de experimentos con snapshot | `used_config.yaml` por run | Completo |
| Dataset en HuggingFace | `GIAR-UTN/allium-cepa-dataset` | Público. Pinned via `--rev <sha>` en `dvc.yaml` |
| `uv` + `hatchling` + `ruff` + `pre-commit` | `pyproject.toml` | Completo. CI hereda esta configuración |

### A construir — trabajo nuevo

| Componente | Ubicación nueva | Descripción |
|---|---|---|
| DVC stages de training | Extensión de `dvc.yaml` | Agregar stages: `train_detector`, `calibrate_detector`, `train_classifier`, `calibrate_classifier`, `evaluate`, `validate_model` |
| Config de producción y validación | `src/allium_cepa_classifier/config/` | `ProductionConfig`, `ValidationConfig`, `ZooniverseConfig` — todas extienden `BaseConfig` |
| Model validation gate | `src/allium_cepa_classifier/validation/validate_model.py` | Compara `metrics.json` del run nuevo vs producción |
| DAGs de Airflow | `airflow/dags/` | Los tres DAGs descritos en sección 6 |
| Clientes de ingesta | `src/allium_cepa_classifier/ingestion/` | `zooniverse_client.py`, `drive_client.py`, `hf_publisher.py` |
| FastAPI wrapper | `app/api.py` | Thin wrapper sobre `AlliumCepaModel`. No contiene lógica de inferencia propia |
| Streamlit UI | `app/streamlit_app.py` | UI para usuarios no técnicos |
| Dockerfiles | Raíz del repo | `Dockerfile` (inferencia liviana), `Dockerfile.train` (training completo) |
| Docker Compose | Raíz del repo | `docker-compose.yml` (dev), `docker-compose.prod.yml` (prod) |
| DVC remotes config | `.dvc/config` | `local_minio` y `production` |
| CI workflow | `.github/workflows/ci.yml` | Corre `uv run pytest` + `uv run ruff check` en cada push |
| MLflow integration | Dentro de los scripts de training existentes | Agregar `mlflow.log_params/metrics/artifacts` calls. Mínima intervención |

### Descartado del scope

- VAE (`train_vae.py`, `vae_model.py`, `vae_trainer.py`, `vae_evaluator.py`) — no forma parte del pipeline productivo
- ControlNet / Diffusers (`train_controlnet.py`, `generate_controlnet_samples.py`, `scripts/vendor/`) — standalone, no se integra

---

## 3. Stack tecnológico por capa

| Capa | Tecnología | Rol |
|---|---|---|
| Package manager | `uv` + `hatchling` | Ya configurado. Todos los Dockerfiles usan `uv sync` |
| Linting / formatting | `ruff` | Ya configurado. CI lo corre en cada push |
| Orquestación | Apache Airflow | DAGs de ingesta y retraining |
| Versionado de datos (pipeline) | DVC | Versiona outputs del pipeline en S3/MinIO |
| Dataset ground truth | HuggingFace (`GIAR-UTN/allium-cepa-dataset`) | Público, pinned por SHA en `dvc.yaml`. Solo se actualiza con datos validados por expertos |
| Object storage (dev) | MinIO | Emula S3 en local. Compatible al 100% con la API de S3 |
| Object storage (prod) | AWS S3 | El único cambio vs MinIO es `endpoint_url` en `.dvc/config` y boto3 |
| Experiment tracking | MLflow | Métricas, parámetros, model registry. Se agrega a los scripts de training existentes |
| Training framework | PyTorch + `timm` + Ultralytics | Ya implementado. No se cambia |
| Detección | YOLO (Ultralytics) | Ya implementado |
| Clasificación | EfficientNet/ResNet/VGG via `timm` | Ya implementado |
| Calibración | Vector scaling + Isotonic regression | Ya implementado |
| Inferencia | `AlliumCepaModel` | Ya implementado. FastAPI lo envuelve sin agregar lógica |
| Serving API | FastAPI | Thin wrapper sobre `AlliumCepaModel` |
| UI | Streamlit | Interfaz para usuarios no técnicos |
| Containerización | Docker + Compose | Empaqueta todos los servicios |
| CI/CD | GitHub Actions | Tests + lint. Hereda config de `uv` y `ruff` existente |
| Etiquetado externo | Zooniverse | Clasificación por expertos. Tiene API REST |
| Demo pública | Hugging Face Spaces | Streamlit siempre online, $0 |

---

## 4. Estructura del repositorio

Lo que sigue es la estructura **final** del repo. Lo que ya existe se marca con `[existe]`. Lo nuevo con `[nuevo]`.

```
allium-cepa-cell-detection/
│
├── .github/
│   └── workflows/
│       └── ci.yml                          [nuevo] Tests + lint en cada push
│
├── airflow/                                [nuevo]
│   └── dags/
│       ├── zooniverse_ingest.py            [nuevo] DAG A
│       ├── raw_image_ingest.py             [nuevo] DAG B
│       └── retrain_pipeline.py             [nuevo] DAG C
│
├── app/                                    [nuevo]
│   ├── api.py                              [nuevo] FastAPI: thin wrapper sobre AlliumCepaModel
│   └── streamlit_app.py                    [nuevo] UI Streamlit
│
├── src/
│   └── allium_cepa_classifier/
│       ├── config/
│       │   ├── base_config.py              [existe] BaseConfig con from_yaml()
│       │   ├── allium_cepa_config.py       [existe] Config de inferencia
│       │   ├── experiment_config.py        [existe] Config de training del clasificador
│       │   ├── detector_config.py          [existe] Config de training del detector
│       │   ├── production_config.py        [nuevo]  Paths a modelo en S3, confidence thresholds
│       │   ├── validation_config.py        [nuevo]  Thresholds del model gate (min F1 delta, tolerancia por clase)
│       │   └── zooniverse_config.py        [nuevo]  Credenciales, project ID, consensus threshold
│       │
│       ├── data_models/
│       │   └── allium_cepa_model.py        [existe] AlliumCepaModel — interfaz de inferencia completa
│       │
│       ├── ingestion/                      [nuevo]
│       │   ├── zooniverse_client.py        [nuevo]  Descarga clasificaciones con filtro de consenso
│       │   ├── drive_client.py             [nuevo]  Descarga imágenes crudas desde Google Drive
│       │   └── hf_publisher.py             [nuevo]  Publica datos validados a HuggingFace + retorna nuevo SHA
│       │
│       ├── training/
│       │   ├── model_builder.py            [existe] build_model() → BackboneWithHead
│       │   ├── trainer.py                  [existe] Loop de training del clasificador
│       │   ├── calibrator.py               [existe] Vector scaling
│       │   └── detector_calibrator.py      [existe] Isotonic regression
│       │
│       └── validation/                     [nuevo]
│           └── validate_model.py           [nuevo]  Model validation gate
│
├── scripts/
│   ├── train_classifier.py                 [existe] Invocado por DVC stage
│   ├── train_detector.py                   [existe] Invocado por DVC stage
│   ├── calibrate_classifier.py             [existe] Invocado por DVC stage
│   ├── calibrate_detector.py               [existe] Invocado por DVC stage
│   └── sweep.py                            [existe] Sweep de configs
│
├── experiments/                            [existe] Configs y artifacts por experimento
│   ├── binary_classifier/
│   │   └── efficientnet_b1/
│   │       └── config.yaml
│   └── yolo/
│       └── yolo11n_200e/
│           └── config.yaml
│
├── datasets/                               [existe, ignorado por git]
│   ├── allium_cepa_full_images_merged_v3/  [existe] Raw COCO dataset (descargado por DVC)
│   ├── crops/
│   │   └── binary_classifier/             [existe] Crops para training del clasificador
│   └── yolo_dataset/                      [existe] Dataset en formato YOLO
│
├── tests/
│   ├── [tests existentes]                  [existe]
│   ├── test_validate_model.py              [nuevo]
│   └── test_ingestion_clients.py           [nuevo]
│
├── notebooks/                              [existe] Notebooks exploratorios
│
├── docs/
│   └── screenshots/                        [nuevo] MLflow UI, Airflow DAGs, demo — para README
│
├── dvc.yaml                                [existe → extender] Agregar stages de training y validación
├── dvc.lock                                [existe] Generado por DVC, no editar manualmente
├── .dvc/config                             [existe → modificar] Agregar remotes local_minio y production
├── pyproject.toml                          [existe] uv + hatchling + ruff. Agregar dependencias nuevas
├── docker-compose.yml                      [nuevo] Stack completo de desarrollo
├── docker-compose.prod.yml                 [nuevo] Solo API + Streamlit
├── Dockerfile                              [nuevo] Imagen de inferencia (liviana, solo runtime deps)
├── Dockerfile.train                        [nuevo] Imagen de training (uv sync --all-groups)
└── README.md                               [existe → reescribir] Optimizado para AI recruiters
```

---

## 5. Componentes funcionales

### 5.1 HuggingFace Dataset — Ground Truth

**Responsabilidad:** Almacenar el dataset de ground truth curado y con validación experta. Es la fuente de datos de entrenamiento principal y la única fuente de verdad para el test set fijo.

**Repo:** `GIAR-UTN/allium-cepa-dataset` (público)

**Política de actualización:** Solo se actualiza cuando DAG A confirma que nuevos datos han pasado el filtro de consenso de Zooniverse. Nunca recibe datos sin validación experta.

**Mecanismo de versionado:** SHA pinned en `dvc.yaml` con `--rev <sha>`. Cada actualización genera un nuevo SHA que se commitea en `dvc.yaml`. Esto mantiene la reproducibilidad completa: cualquier SHA anterior del repo reproduce exactamente el mismo dataset.

**Flujo de actualización desde el pipeline:**
```
DAG A confirma consenso de nuevos datos
        ↓
hf_publisher.py sube imágenes + annotations.json actualizado a HF
        ↓
HF genera nuevo commit SHA
        ↓
DAG A actualiza --rev en dvc.yaml con el nuevo SHA
        ↓
DAG A hace git commit + push de dvc.yaml
        ↓
El próximo dvc repro descarga el dataset actualizado
```

---

### 5.2 S3 / MinIO — Pipeline Artifacts

**Responsabilidad:** Almacenar todos los artefactos generados por el pipeline: outputs procesados de DVC, pesos de modelos, métricas de producción, imágenes en revisión, logs del UI.

**Política:** Nunca almacena ground truth anotado manualmente. Eso vive en HuggingFace. S3 almacena todo lo que es generado, procesado, o está pendiente de validación.

**Estructura de buckets:**

```
allium-cepa-ml/
│
├── dvc-cache/                          ← DVC remote cache (outputs de cada stage)
│   ├── files/md5/                      ← Blobs por hash (formato DVC)
│
├── dataset/
│   ├── raw/                            ← Imágenes completas sin procesar (de Drive)
│   ├── labeled/
│   │   └── auto/                       ← Crops auto-clasificados con alta confianza
│   ├── review/                         ← Crops pendientes de revisión en Zooniverse
│   └── test_fixed/                     ← Test set inmutable. Se inicializa una vez, nunca se modifica
│
├── models/
│   ├── experiments/                    ← Todos los pesos de runs de training (via DVC)
│   └── production/
│       ├── object_detection.pt         ← Detector YOLO activo
│       ├── classifier_calibrated.pt    ← Clasificador calibrado activo
│       ├── yolo_isotonic_calibrator.pkl ← Calibrador isotónico del detector
│       └── metrics.json                ← Métricas del modelo activo (fuente de verdad para el gate)
│
├── mlflow/                             ← Artifact store de MLflow
│
└── ui_logs/                            ← Predicciones del UI (imagen + confidence + timestamp)
```

**En desarrollo:** MinIO corre como container Docker en `localhost:9000`. UI en `localhost:9001`. Compatible al 100% con la API de S3 — el código no distingue entre MinIO y S3 real.

**En producción:** AWS S3. El único cambio es `endpoint_url` en `.dvc/config` y en la variable de entorno de boto3.

---

### 5.3 DVC — Versionado del pipeline

**Responsabilidad:** Orquestar el pipeline de training de forma reproducible. Versionar los outputs de cada stage en S3/MinIO. Detectar qué stages necesitan re-ejecutarse cuando cambian las dependencias.

**Principio clave:** DVC no versiona el ground truth (eso es HuggingFace + SHA). DVC versiona los **outputs generados** por el pipeline: crops procesados, pesos entrenados, métricas de evaluación.

**Remotes:**
- `local_minio`: `s3://allium-cepa-ml` con `endpointurl = http://localhost:9000` — desarrollo
- `production`: `s3://allium-cepa-ml` sin endpoint — apunta a AWS S3 real

El switch es `dvc remote default local_minio` o `dvc remote default production`. Todo el código es idéntico.

Ver sección 7 para los stages completos.

---

### 5.4 MLflow — Experiment Tracking y Model Registry

**Responsabilidad:** Registrar cada run de training con sus parámetros, métricas y artefactos. Gestionar el ciclo de vida de los modelos con el Model Registry (stages: `None → Staging → Production → Archived`).

**Integración con el repo existente:** Se agregan calls a `mlflow.log_params()`, `mlflow.log_metrics()`, y `mlflow.log_artifacts()` dentro de los scripts de training existentes (`train_classifier.py`, `train_detector.py`). Mínima intervención al código existente.

**El `used_config.yaml` ya existente** se loguea como artefacto de MLflow en cada run, manteniendo la trazabilidad de configuración que ya existe.

**En desarrollo:** Container Docker en `localhost:5000`. Artifact store apunta a MinIO.

**En producción:** Render.com free tier (URL pública). Artifact store apunta a S3. URL visible en el README como evidencia de experimentos reales.

---

### 5.5 Airflow — Orquestación

**Responsabilidad:** Ejecutar los DAGs de ingesta de datos y disparar el pipeline de retraining. Provee UI de monitoreo, retry logic, y alertas por email en caso de fallo.

**Relación con DVC:** Airflow no conoce los internals del pipeline de training. DAG C simplemente ejecuta `dvc repro` como subprocess. DVC se encarga del resto.

**En desarrollo:** Container Docker en `localhost:8080`. SQLite como backend. DAGs ejecutables manualmente desde la UI sin esperar el schedule.

**En producción:** Corre en la máquina local del investigador (o EC2 t2.micro si se necesita disponibilidad continua). No necesita URL pública. Los DAGs son nocturnos — se puede levantar el container, ejecutar los DAGs, y apagarlo.

---

### 5.6 FastAPI — Servicio de inferencia

**Responsabilidad:** Exponer un endpoint REST `/predict` que recibe una imagen y devuelve la clasificación con confidence scores e intervalos de confianza.

**Implementación:** Instancia `AlliumCepaModel(AlliumCepaConfig())` una sola vez al startup del servidor. El endpoint llama `model.predict()` y retorna `result.get_counts_with_ci()`. No contiene lógica de inferencia propia — toda la lógica ya está en `AlliumCepaModel`.

Al recibir cada imagen, loguea la imagen y el confidence score en `s3://ui_logs/` para alimentar el active learning loop.

Al iniciar, descarga los pesos del modelo desde `s3://models/production/` si no están presentes localmente.

**En desarrollo:** Container Docker en `localhost:8000`. Se conecta a MinIO para pesos y logs.

**En producción:** Container en Hugging Face Spaces o EC2. Se conecta a S3 real.

---

### 5.7 Streamlit — UI para usuarios no técnicos

**Responsabilidad:** Proveer una interfaz web simple donde investigadores pueden subir imágenes de microscopía completas y obtener el mitotic index con intervalos de confianza, sin necesidad de conocimiento técnico.

**En desarrollo:** Container Docker en `localhost:8501`. Se conecta a FastAPI por red interna de Docker (`http://api:8000`).

**En producción:** Hugging Face Spaces (siempre online, $0). Se conecta a la URL pública de FastAPI.

---

### 5.8 GitHub Actions — CI

**Responsabilidad:** Correr `uv run pytest` y `uv run ruff check` en cada push. Mantiene el badge de estado verde en el README.

**No hace training.** Los runners de GitHub Actions no tienen GPU y el tiempo de training excede los límites del free tier. El training corre siempre en la máquina local vía `dvc repro`.

El workflow hereda la configuración de `uv`, `ruff`, y `pre-commit` ya existente en `pyproject.toml`.

---

## 6. DAGs de Airflow

### DAG A — `zooniverse_ingest`

**Schedule:** Diario, 3:00 AM

**Propósito:** Descargar clasificaciones nuevas de Zooniverse, filtrar por consenso, y cuando hay suficientes datos validados, publicarlos en HuggingFace y actualizar el SHA en `dvc.yaml`.

```
Task 1: download_zooniverse_classifications
│  └─ Llama a la API de Zooniverse
│  └─ Descarga clasificaciones desde el último timestamp procesado
│  └─ Guarda clasificaciones crudas en archivo temporal
│
Task 2: filter_by_consensus
│  └─ Filtra clasificaciones con acuerdo entre expertos >= umbral (ZooniverseConfig.consensus_threshold)
│  └─ Descarta imágenes ambiguas
│  └─ Loguea cuántas fueron aceptadas vs descartadas
│
Task 3: normalize_to_coco_format
│  └─ Convierte formato Zooniverse al formato COCO interno del dataset
│  └─ Valida que cada imagen tenga exactamente un label
│  └─ Asigna flag source: "zooniverse" en las anotaciones
│
Task 4: upload_validated_to_s3
│  └─ Sube imágenes validadas a s3://dataset/labeled/zooniverse/ (staging area)
│
Task 5: publish_to_huggingface
│  └─ Ejecutado solo si el volumen de datos nuevos supera el umbral mínimo configurable
│  └─ hf_publisher.py agrega las nuevas imágenes al repo GIAR-UTN/allium-cepa-dataset
│  └─ Actualiza annotations.json con los nuevos labels
│  └─ Obtiene el nuevo commit SHA del repo HF
│
Task 6: update_dvc_yaml_sha
│  └─ Actualiza el --rev <sha> en dvc.yaml con el nuevo SHA de HF
│  └─ git commit + git push de dvc.yaml al repositorio
│  └─ Esto hace que el próximo dvc repro descargue el dataset actualizado
│
Task 7: notify_retrain
   └─ Notifica al DAG C que hay datos nuevos disponibles
   └─ DAG C decide si el volumen justifica un retraining
```

**En caso de fallo:** Retry automático en Tasks 1, 4, 5 (operaciones de red). Tasks 2 y 3 son deterministas, no se reintentan. Alerta por email al investigador en fallo persistente.

---

### DAG B — `raw_image_ingest`

**Schedule:** On-demand (trigger manual o detección de archivos nuevos en Drive)

**Propósito:** Procesar imágenes crudas de microscopía completas, detectar instancias celulares, cropearlas, clasificarlas con el modelo actual, y routearlas según confidence.

```
Task 1: download_from_drive
│  └─ drive_client.py conecta a Google Drive API
│  └─ Descarga imágenes nuevas de la carpeta compartida con investigadores
│  └─ Valida formato (TIF, PNG, JPG)
│  └─ Sube imágenes crudas a s3://dataset/raw/
│
Task 2: run_instance_detection
│  └─ Instancia AlliumCepaModel con el modelo de producción (desde s3://models/production/)
│  └─ Corre solo la etapa de detección YOLO sobre cada imagen completa
│  └─ Genera crops individuales de cada célula detectada
│  └─ Guarda crops en directorio temporal
│
Task 3: classify_and_score
│  └─ Corre la etapa de clasificación + calibración sobre los crops
│  └─ Genera (clase, confidence_score, ci_lower, ci_upper) por crop via get_counts_with_ci()
│
Task 4: route_by_confidence
│  └─ confidence >= ProductionConfig.high_confidence_threshold:
│      → s3://dataset/labeled/auto/ con flag source: "auto"
│  └─ confidence < ProductionConfig.high_confidence_threshold:
│      → s3://dataset/review/
│  └─ Loguea distribución de confidence scores
│
Task 5: send_low_confidence_to_zooniverse
│  └─ Para imágenes en review/: crea nuevas tareas en Zooniverse via API
│  └─ Las imágenes de confidence muy bajo se marcan como prioridad alta
│
Task 6: notify_retrain_if_threshold
   └─ Si los datos en labeled/auto/ superan el umbral de volumen: notifica al DAG C
```

---

### DAG C — `retrain_pipeline`

**Schedule:** Triggered por DAG A (Task 7) o DAG B (Task 6). También ejecutable manualmente.

**Propósito:** Re-entrenar el modelo completo con el dataset actualizado, validar contra producción, y promover si es mejor.

```
Task 1: check_dataset_changes
│  └─ Compara el SHA actual en dvc.yaml con el último SHA usado en training
│  └─ Si no hay cambios: marca tasks siguientes como skipped
│
Task 2: run_dvc_repro
│  └─ Ejecuta dvc repro como subprocess
│  └─ DVC detecta qué stages cambiaron (el SHA del dataset cambió en Task 6 de DAG A)
│  └─ Re-ejecuta solo los stages afectados:
│      download_dataset → coco_to_yolo → prepare_crops →
│      train_detector → calibrate_detector →
│      train_classifier → calibrate_classifier →
│      evaluate → validate_model
│  └─ MLflow loguea métricas y artefactos de cada run durante la ejecución
│
Task 3: read_validation_result
│  └─ Lee validation_result.json generado por el stage validate_model de DVC
│  └─ Determina el branch: approved o rejected
│
Task 4a (si approved): promote_model
│  └─ Copia los pesos del nuevo modelo a s3://models/production/
│      (object_detection.pt, classifier_calibrated.pt, yolo_isotonic_calibrator.pkl)
│  └─ Actualiza s3://models/production/metrics.json con las métricas del nuevo modelo
│  └─ Transiciona la versión en MLflow Model Registry a stage: Production
│  └─ Llama a la API de Hugging Face Spaces para reiniciar el container
│      (FastAPI descargará automáticamente el nuevo modelo al iniciar)
│  └─ Envía notificación de éxito con métricas comparativas al investigador
│
Task 4b (si rejected): archive_model
   └─ Registra el modelo en MLflow como stage: Archived
   └─ Loguea las métricas comparativas (nuevo vs producción)
   └─ Envía notificación de rechazo con detalle de por qué falló el gate
```

---

## 7. Pipeline de DVC

El `dvc.yaml` existente tiene tres stages. Se extiende con seis stages nuevos. DVC determina automáticamente qué re-ejecutar según las dependencias que cambiaron.

### Stages existentes (no modificar)

```
Stage: download_dataset
│  Dep:   --rev <sha> en dvc.yaml (el SHA de HuggingFace)
│  Out:   datasets/allium_cepa_full_images_merged_v3/
│  Hace:  Descarga el dataset desde GIAR-UTN/allium-cepa-dataset en HF

Stage: coco_to_yolo
│  Dep:   datasets/allium_cepa_full_images_merged_v3/
│  Out:   datasets/yolo_dataset/
│  Hace:  Convierte anotaciones COCO a formato YOLO

Stage: prepare_crops
   Dep:   datasets/allium_cepa_full_images_merged_v3/
   Out:   datasets/crops/binary_classifier/
   Hace:  Genera crops individuales para el clasificador binario
```

### Stages nuevos (a agregar)

```
Stage: train_detector
│  Dep:   datasets/yolo_dataset/, experiments/yolo/yolo11n_200e/config.yaml
│  Out:   experiments/yolo/yolo11n_200e/<timestamp>/weights/
│  Hace:  uv run python scripts/train_detector.py --config ...
│         Loguea en MLflow y TensorBoard

Stage: calibrate_detector
│  Dep:   experiments/yolo/yolo11n_200e/<timestamp>/weights/, datasets/yolo_dataset/val/
│  Out:   experiments/yolo/yolo11n_200e/<timestamp>/weights/yolo_isotonic_calibrator.pkl
│  Hace:  uv run python scripts/calibrate_detector.py --experiment ...

Stage: train_classifier
│  Dep:   datasets/crops/binary_classifier/, experiments/binary_classifier/efficientnet_b1/config.yaml
│  Out:   experiments/binary_classifier/efficientnet_b1/<timestamp>/weights/classifier.pt
│  Hace:  uv run python scripts/train_classifier.py --config ... --no-calibrate
│         Loguea en MLflow y TensorBoard

Stage: calibrate_classifier
│  Dep:   experiments/binary_classifier/efficientnet_b1/<timestamp>/weights/classifier.pt
│  Out:   experiments/binary_classifier/efficientnet_b1/<timestamp>/weights/classifier_calibrated.pt
│  Hace:  uv run python scripts/calibrate_classifier.py --experiment ...
│         Loguea ECE antes/después en MLflow

Stage: evaluate
│  Dep:   Ambos modelos calibrados, datasets/test_fixed/ (inmutable en S3)
│  Out:   metrics/evaluation_report.json
│  Hace:  Evalúa el modelo two-stage completo (AlliumCepaModel) sobre el test set fijo
│         Calcula: Macro F1, F1 por clase, Accuracy, ECE
│         El formato extiende el metrics.json existente del repo
│         Loguea en MLflow como artefacto del run

Stage: validate_model
   Dep:   metrics/evaluation_report.json, s3://models/production/metrics.json
   Out:   validation_result.json  {approved: bool, new_metrics: {...}, current_metrics: {...}}
   Hace:  validate_model.py compara métricas (ver sección 11)
          Exit code 0 = aprobado, 1 = rechazado (DAG C lee este archivo)
          Loguea comparación en MLflow
```

**Reproducibilidad garantizada:** `git clone` + `dvc pull` + `dvc repro` reproduce exactamente el mismo modelo en cualquier máquina con acceso al S3 y al HuggingFace repo.

---

## 8. Flujo de datos end-to-end

### Flujo A — Clasificaciones de expertos (Zooniverse → HuggingFace → Training)

```
Experto clasifica imágenes en Zooniverse
        ↓
DAG A se ejecuta (diario, 3 AM)
        ↓
Filtro de consenso (ZooniverseConfig.consensus_threshold)
        ↓
Datos validados → hf_publisher.py actualiza GIAR-UTN/allium-cepa-dataset
        ↓
Nuevo SHA commiteado en dvc.yaml
        ↓
DAG C disparado
        ↓
dvc repro detecta que download_dataset tiene nuevo SHA → re-ejecuta pipeline completo
        ↓
validate_model.py compara nuevo modelo vs producción
        ↓
Aprobado → s3://models/production/ actualizado + MLflow Registry + HF Space reiniciado
```

### Flujo B — Imágenes crudas (Drive → S3 → Posiblemente Zooniverse)

```
Investigador sube imagen completa a Google Drive
        ↓
DAG B se ejecuta (on-demand)
        ↓
AlliumCepaModel detecta células → genera crops
        ↓
Clasificación + confidence scoring
        ↓
Alta confianza  →  s3://dataset/labeled/auto/  →  (volumen suficiente → DAG C)
Baja confianza  →  s3://dataset/review/         →  nueva tarea en Zooniverse
                                                    (vuelve al Flujo A)
```

### Flujo C — Imágenes del UI (usuarios de Streamlit)

```
Usuario sube imagen en Streamlit
        ↓
FastAPI llama AlliumCepaModel.predict() → result.get_counts_with_ci()
        ↓
Resultado mostrado: mitotic index + CI al 95.45%
        ↓
Imagen + confidence logueados en s3://ui_logs/ con timestamp
        ↓
Revisión batch mensual (manual, investigador)
        ↓
Aprobadas       →  s3://dataset/labeled/auto/
Dudosas o bajas →  Zooniverse (vuelve al Flujo A)
```

---

## 9. Entorno de desarrollo

**Un solo comando para levantar todo el stack:**
```
docker-compose up
```

**Servicios y puertos:**

| Servicio | Puerto | Descripción |
|---|---|---|
| MinIO API | 9000 | S3 local. Usado por DVC, MLflow, boto3 |
| MinIO UI | 9001 | Consola web para inspeccionar buckets |
| MLflow | 5000 | Tracking UI y Model Registry |
| Airflow | 8080 | DAGs UI, scheduling, logs, retry |
| FastAPI | 8000 | Endpoint de inferencia |
| Streamlit | 8501 | UI de demo |

**Consideraciones clave:**

- Los Dockerfiles usan `uv sync` para instalar dependencias — coherente con el toolchain existente del repo.
- `Dockerfile` (inferencia): `uv sync --no-dev` — solo dependencias de runtime.
- `Dockerfile.train` (training): `uv sync --all-groups` — todas las dependencias incluyendo dev.
- DVC remote `local_minio` apunta a `http://localhost:9000`. Cambiar a `production` es una línea.
- El test set fijo (`s3://dataset/test_fixed/`) se inicializa una vez con `dvc push` al arrancar el proyecto. Nunca se vuelve a modificar.
- Los DAGs de Airflow pueden ejecutarse manualmente desde la UI sin esperar el schedule — útil durante desarrollo.
- Los scripts de training existentes se ejecutan directamente con `uv run python scripts/...` para desarrollo rápido sin pasar por DVC.

**Variables de entorno** (en `.env`, nunca commiteado al repo):

```
# MinIO (desarrollo)
MINIO_ROOT_USER
MINIO_ROOT_PASSWORD

# Google Drive
GOOGLE_DRIVE_CREDENTIALS_PATH

# Zooniverse
ZOONIVERSE_USERNAME
ZOONIVERSE_PASSWORD
ZOONIVERSE_PROJECT_ID

# HuggingFace
HF_TOKEN

# MLflow
MLFLOW_TRACKING_URI

# AWS (solo producción)
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
```

---

## 10. Entorno productivo

**Componentes y dónde corren:**

| Componente | Dónde | Costo | Siempre online |
|---|---|---|---|
| HuggingFace Dataset | HuggingFace | $0 | Sí |
| AWS S3 | AWS | ~$0.02/mes | Sí |
| MLflow | Render.com (free tier) | $0 | Sí (sleep por inactividad) |
| FastAPI | Hugging Face Spaces | $0 | Sí |
| Streamlit | Hugging Face Spaces | $0 | Sí |
| Airflow | Máquina local del investigador | $0 | Solo cuando corre DAGs |
| Training (`dvc repro`) | Máquina local del investigador | $0 | Solo cuando re-entrena |

**Decisiones de arquitectura productiva:**

- **Airflow no necesita estar 24/7 online.** Los DAGs son nocturnos o on-demand. Se levanta el container, se ejecutan los DAGs, se apaga.
- **El training corre en la máquina del investigador.** Tiene GPU/CPU disponible y es donde están los datos. No se necesita EC2 para training.
- **EC2 t2.micro es opcional.** Solo si se quiere FastAPI en una URL propia en lugar de HuggingFace Spaces.
- **El modelo en producción** son tres archivos en `s3://models/production/`. FastAPI los descarga al iniciar si no están en el filesystem local.

**Flujo de deploy de un nuevo modelo:**

```
Stage validate_model de DVC genera validation_result.json {approved: true}
        ↓
DAG C Task 4a: copia los tres archivos de pesos a s3://models/production/
        ↓
DAG C Task 4a: actualiza s3://models/production/metrics.json
        ↓
DAG C Task 4a: MLflow Model Registry → stage: Production
        ↓
DAG C Task 4a: llama API de HuggingFace Spaces para reiniciar el container
        ↓
FastAPI descarga nuevos pesos desde S3 al iniciar
        ↓
Nuevo modelo activo en producción
```

---

## 11. Model Validation Gate

**Propósito:** Garantizar que ningún modelo peor que el actual llegue a producción, independientemente de cuántos datos nuevos haya.

**Implementación:** `src/allium_cepa_classifier/validation/validate_model.py`, invocado por el DVC stage `validate_model`. Lee dos archivos `metrics.json` y aplica la lógica de decisión.

**Test set fijo:** El árbitro neutral. Se define una vez al inicio del proyecto desde el split `test` del dataset COCO. Ningún stage de augmentation o preprocessing lo toca. Vive en `s3://dataset/test_fixed/` y se referencia como dependencia del stage `evaluate` en `dvc.yaml`.

**Métricas comparadas:**

El `metrics.json` existente ya tiene `accuracy` y `ECE`. Se extiende para agregar:

| Métrica | Tipo | Justificación |
|---|---|---|
| Macro F1 | **Principal** | Dataset desbalanceado (más no_mitosis que mitosis). Accuracy sola es insuficiente |
| F1 por clase (mitosis / no_mitosis) | Secundaria | Detecta degradación en una clase específica aunque el Macro F1 mejore |
| Accuracy | Terciaria | Control de sanidad. Ya existe en metrics.json |
| ECE (Expected Calibration Error) | Complementaria | Crítico para el modelo de incertidumbre. Ya existe en metrics.json |

**Lógica de decisión** (umbrales configurables en `ValidationConfig`):

- El nuevo modelo debe superar el Macro F1 de producción por `ValidationConfig.min_f1_delta` (ej: 0.01).
- No se acepta un modelo que mejore el Macro F1 pero degrade el F1 de cualquier clase individual en más de `ValidationConfig.per_class_tolerance` (ej: 0.03).
- No se acepta un modelo con ECE significativamente peor que el de producción (umbral configurable).

**Fuente de verdad:** `s3://models/production/metrics.json` — se actualiza cada vez que un modelo es promovido. Es lo que el gate usa como baseline.

---

## 12. Active Learning Loop

**Propósito:** El sistema mejora con el tiempo sin requerir que los investigadores etiqueten datos manualmente. Las imágenes donde el modelo es más incierto son exactamente las que más valor aportan al dataset.

**Umbrales** (configurables en `ProductionConfig`):

| Rango de confidence | Acción |
|---|---|
| >= `high_confidence_threshold` (ej: 0.90) | Auto-label → `s3://dataset/labeled/auto/` con flag `source: auto` |
| < `high_confidence_threshold` | → `s3://dataset/review/` → nueva tarea en Zooniverse |

**Ciclo completo:**

```
1. Modelo en producción clasifica imágenes (DAG B o UI)
2. Crops de baja confianza van a Zooniverse como nuevas tareas
3. Expertos clasifican en Zooniverse a su ritmo
4. DAG A descarga clasificaciones, filtra por consenso
5. Datos validados → HuggingFace + nuevo SHA en dvc.yaml
6. DAG C dispara dvc repro → nuevo modelo entrenado
7. Model Validation Gate: ¿es mejor?
8. Sí → nuevo modelo en producción, más preciso
9. El modelo más preciso envía menos imágenes a Zooniverse
10. (Loop)
```

**Integridad científica:** Las imágenes con flag `source: auto` están identificadas en las anotaciones del dataset. En los experimentos de training se puede controlar si incluirlas o no (parámetro en `ExperimentConfig`), permitiendo medir el impacto real del active learning en las métricas comparando runs con y sin datos auto-etiquetados.

**El test set fijo nunca recibe imágenes auto-etiquetadas.** Solo datos con validación experta pueden entrar al test set, garantizando que el gate siempre evalúa contra ground truth de calidad conocida.

---

## 13. Matriz de integración entre componentes

| Componente | Se conecta con | Cómo |
|---|---|---|
| Airflow DAG A | Zooniverse API | HTTP REST (`panoptes-client` o requests) |
| Airflow DAG A | S3 / MinIO | boto3 |
| Airflow DAG A | `hf_publisher.py` | Python directo (`huggingface_hub`) |
| Airflow DAG A | Git repo | subprocess: `git commit`, `git push` (actualiza SHA en dvc.yaml) |
| Airflow DAG A | DAG C | Airflow trigger / sensor |
| Airflow DAG B | Google Drive API | `google-auth` + `google-api-python-client` |
| Airflow DAG B | S3 / MinIO | boto3 |
| Airflow DAG B | `AlliumCepaModel` | Python directo (instancia el modelo para detección + clasificación) |
| Airflow DAG B | Zooniverse API | HTTP REST (crea nuevas tareas para imágenes de baja confianza) |
| Airflow DAG B | DAG C | Airflow trigger / sensor |
| Airflow DAG C | DVC | subprocess: `dvc repro` |
| Airflow DAG C | MLflow | mlflow Python SDK (lee resultados del run) |
| Airflow DAG C | S3 / MinIO | boto3 (copia pesos a `production/`, actualiza `metrics.json`) |
| Airflow DAG C | HuggingFace Spaces API | HTTP REST (reinicia el container de FastAPI) |
| DVC stage `download_dataset` | HuggingFace | `huggingface_hub` (pinned SHA) |
| DVC stage `train_*` | MLflow | mlflow Python SDK (dentro de los scripts de training existentes) |
| DVC stage `train_*` | TensorBoard | Ya integrado en los trainers existentes |
| DVC stage `evaluate` | S3 / MinIO | boto3 (descarga `test_fixed/`) |
| DVC stage `validate_model` | S3 / MinIO | boto3 (descarga `production/metrics.json`) |
| `hf_publisher.py` | HuggingFace | `huggingface_hub` (`HfApi.upload_folder()`) |
| FastAPI | `AlliumCepaModel` | Python directo (instancia en startup) |
| FastAPI | S3 / MinIO | boto3 (descarga pesos al iniciar, loguea en `ui_logs/`) |
| Streamlit | FastAPI | HTTP REST (`http://api:8000` en dev, URL pública en prod) |
| MLflow | S3 / MinIO | MLflow artifact store configurado como `s3://mlflow/` |
| GitHub Actions | Repositorio | `git push` trigger — corre `uv run pytest` + `uv run ruff check` |

---

## Orden de implementación recomendado para Claude Code

Cada bloque es independiente y testeable por separado.

1. `.dvc/config` — agregar remotes `local_minio` y `production`
2. `docker-compose.yml` — stack completo de desarrollo (MinIO, MLflow, Airflow, API, Streamlit)
3. `Dockerfile` + `Dockerfile.train` — usando `uv sync`
4. `src/allium_cepa_classifier/config/` — `ProductionConfig`, `ValidationConfig`, `ZooniverseConfig`
5. Extensión de `dvc.yaml` — los seis stages nuevos
6. `src/allium_cepa_classifier/validation/validate_model.py` — el gate
7. MLflow integration — agregar calls en los scripts de training existentes
8. `src/allium_cepa_classifier/ingestion/` — `zooniverse_client.py`, `drive_client.py`, `hf_publisher.py`
9. `airflow/dags/` — los tres DAGs
10. `app/api.py` — FastAPI wrapper
11. `app/streamlit_app.py` — UI
12. `.github/workflows/ci.yml` — CI
13. `README.md` — reescritura optimizada para AI recruiters

---

*Este documento es el input para la implementación con Claude Code.*
*Cada sección es un bloque de trabajo independiente.*
*Lo que ya existe en el repo no se reimplementa — se extiende o se envuelve.*
