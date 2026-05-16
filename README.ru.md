# SmartUIBot

Фреймворк бота на основе компьютерного зрения в реальном времени: выбираете
область экрана → захватываете её → запускаете детекцию YOLO11 → движок
принятия решений на основе полезности выбирает поведение → очеловеченный
ввод мышью/клавиатурой выполняет его — всё это видно в живом окне отладки и
защищено переключателем безопасности **Arm/Disarm** (Взвести/Снять).

Кросс-платформенный (macOS / Windows / Linux), полностью построен на внедрении
зависимостей, поэтому весь конвейер запускается headless с фейками.
**Только для одиночной / офлайн-игры.**

> 🇬🇧 English version: [README.md](README.md)

## Быстрый старт
    python -m pip install -e ".[dev]"
    python run.py

При первом запуске (нет `configs/state.yaml`) появляется оверлей выбора ROI —
растяните прямоугольник, чтобы задать область захвата; она сохраняется между
перезапусками. Панель управления предлагает Start/Stop, Pause/Resume,
**Arm/Disarm**, ползунок уверенности, горячую перезагрузку модели и повторный
выбор ROI — всё в рантайме.

Бот запускается в режиме **Disarmed** (снят): ничего не вводится, пока вы его
не взведёте. Горячая клавиша аварийной остановки и предохранитель «угол экрана»
немедленно снимают бота с боевого взвода и прерывают действие. Разрешения
платформы и параметры настройки описаны в [SETUP.md](SETUP.md).

## Тестирование
    pytest -q -m "not model"     # быстро, headless, без GPU/экрана
    pytest -q -m model           # скачивает yolo11n.pt, реальная инференция

---

# Как это работает

## Конвейер в общих чертах

Четыре сервиса-воркера, каждый в своём потоке, общаются **только** через
синхронную внутрипроцессную шину событий. Данные текут в одном направлении;
backpressure — «отбросить старое, оставить самое свежее», поэтому бот всегда
работает с самым свежим кадром.

```
 экран ──grab──> CaptureService ──FrameCaptured──────┐
                                                      ▼
                                         DetectionService  (инференция YOLO11)
                                                      │
                                                      ├─DetectionsReady──> DebugWindow (живой превью)
                                                      ▼
                                         DecisionService   (политика полезности, только ARMED)
                                                      │
                                                      ├─ActionRequested──┐
                                                      ▼                   ▼
                                         ActionService ──move/click/key──> ввод ОС
                                                      │
                                                      └─ActionStarted / Completed / Aborted ─> DebugWindow
```

Между сервисами используется **очередь размера 1 «последний побеждает»**
([`core/latest_queue.py`](src/smartuibot/core/latest_queue.py) —
`LatestQueue` в `src/smartuibot/core/latest_queue.py:7`): `put()`
перезаписывает любой ожидающий элемент, поэтому медленный потребитель
(инференция на CPU) никогда не отстаёт от быстрого производителя (захват 60
FPS); он просто пропускает устаревшие кадры.

## Корень композиции — контейнер

Всё связывается в одном месте:
[`core/container.py`](src/smartuibot/core/container.py). `AppContainer`
(`src/smartuibot/core/container.py:26`) принимает конфиг плюс три платформенных
адаптера (бэкенд захвата, детектор, бэкенд ввода) как аргументы конструктора и
создаёт каждый синглтон: шину, четыре сервиса, FSM режима, политику полезности
и watchdog. `start()` (`src/smartuibot/core/container.py:80`) запускает
воркеров в обратном порядке конвейера (action → decision → detection →
capture), чтобы каждый потребитель был подписан раньше, чем его производитель
начнёт эмитить; `stop()` (`src/smartuibot/core/container.py:87`) сначала
снимает бота с взвода, затем разбирает всё.

Реальные адаптеры выбираются в [`app.py`](src/smartuibot/app.py):
`build_real_container` (`src/smartuibot/app.py:67`) вызывает фабрики `_make_*`,
а `main` (`src/smartuibot/app.py:79`) собирает Qt-оболочку, связывает панель
управления + окно отладки, устанавливает горячую клавишу аварийной остановки и
запускает цикл событий. Поскольку платформенные части внедряются, тесты
подставляют фейки из [`tests/fakes/`](tests/fakes) и прогоняют весь цикл без
экрана, GPU или реальной мыши.

## Фреймворк сервисов

Все воркеры наследуются от `Service`
([`core/service.py`](src/smartuibot/core/service.py),
`src/smartuibot/core/service.py:12`). Базовый класс владеет демон-потоком,
гоняет `run_once()` в цикле (`src/smartuibot/core/service.py:52`), обновляет
метку `last_heartbeat` на каждой итерации и поддерживает кооперативные
`pause()`/`resume()`. Любое необработанное исключение превращается в
**фатальное событие `ServiceError`**, и поток завершается аккуратно, а не
роняет процесс.

`Watchdog` ([`core/watchdog.py`](src/smartuibot/core/watchdog.py),
`src/smartuibot/core/watchdog.py:15`) надзирает за всеми четырьмя сервисами:
он опрашивает `is_alive` раз в секунду и перезапускает мёртвый воркер с
экспоненциальной задержкой, эскалируя до фатального `ServiceError` после
`max_retries` (`src/smartuibot/core/watchdog.py:45`).

## Шина событий

[`core/event_bus.py`](src/smartuibot/core/event_bus.py) — потокобезопасный
**синхронный** pub/sub (`src/smartuibot/core/event_bus.py:15`). `publish()`
вызывает подписчиков синхронно; исключение подписчика логируется и
проглатывается (`src/smartuibot/core/event_bus.py:33`), поэтому один плохой
обработчик никогда не сломает производителя или других подписчиков. Все типы
сообщений — это frozen-датаклассы в
[`core/events.py`](src/smartuibot/core/events.py): `FrameCaptured`,
`DetectionsReady`, `ActionRequested`, `ActionStarted/Completed/Aborted`,
`ModeChanged`, `FpsTick`, `LogRecord`, `ServiceError`, `StateChanged`.

Общие формы данных живут в
[`core/types.py`](src/smartuibot/core/types.py): `ROI`
(`src/smartuibot/core/types.py:10`, с валидацией, сериализуется в YAML через
`as_dict`/`from_dict`), `Frame` (BGR `np.ndarray` + seq + timestamp, `:44`),
`Detection` (метка, уверенность, bbox, опциональный track id, `:52`) и
`ActionStep` (исполняемый примитив: `move|click|key|wait`, `:75`).

## Этап 1 — Захват

[`vision/capture/service.py`](src/smartuibot/vision/capture/service.py)
(`CaptureService` в `src/smartuibot/vision/capture/service.py:15`) захватывает
текущий ROI через `CaptureBackend`, оборачивает его в `Frame` с монотонно
растущим `seq`, публикует `FrameCaptured` и спит, удерживая `target_fps`. ROI
можно менять в рантайме под локом (`set_roi`, `:31`), поэтому «Select ROI»
работает во время выполнения.

Бэкенд — это `Protocol`
([`vision/capture/backend.py`](src/smartuibot/vision/capture/backend.py),
`src/smartuibot/vision/capture/backend.py:21`). Поставляемая реализация —
[`mss_backend.py`](src/smartuibot/vision/capture/mss_backend.py); более быстрый
путь `dxcam` для Windows пока откатывается к `mss` (см.
`_make_capture_backend` в `src/smartuibot/app.py:35`).

## Этап 2 — Детекция

[`vision/detect/service.py`](src/smartuibot/vision/detect/service.py)
(`DetectionService` в `src/smartuibot/vision/detect/service.py:16`)
подписывается на `FrameCaptured`, кладёт кадры в свою `LatestQueue` и в своём
потоке берёт самый свежий кадр, запускает инференцию, фильтрует по
**настраиваемому в рантайме** порогу уверенности (`set_confidence`, `:38` —
управляется ползунком в UI), сглаживает результат и публикует
`DetectionsReady`.

- `Detector` — это `Protocol`
  ([`vision/detect/detector.py`](src/smartuibot/vision/detect/detector.py),
  `src/smartuibot/vision/detect/detector.py:11`); реальный — это
  [`yolo.py`](src/smartuibot/vision/detect/yolo.py) (`Yolo11Detector`,
  Ultralytics YOLO11), с горячей перезагрузкой через `reload()`.
- `SmoothingFilter`
  ([`vision/detect/smoothing.py`](src/smartuibot/vision/detect/smoothing.py),
  `src/smartuibot/vision/detect/smoothing.py:7`) держит метку видимой ещё
  `smoothing_frames` кадров после её исчезновения, уменьшая мерцание.

## Этап 3 — Принятие решений

[`ai/service.py`](src/smartuibot/ai/service.py) (`DecisionService` в
`src/smartuibot/ai/service.py:17`) тикает с частотой `decision.tick_hz`.
**Он бездействует, если режим не ARMED**
(`src/smartuibot/ai/service.py:41`). На каждом тике он делает снимок состояния
мира, просит политику выбрать действие и при попадании публикует
`ActionRequested` с конкретными `ActionStep`.

- **Состояние мира**
  ([`ai/world_state.py`](src/smartuibot/ai/world_state.py)):
  `WorldStateTracker` (`src/smartuibot/ai/world_state.py:43`) строит
  неизменяемый снимок `WorldState` на каждый тик и записывает выполненные
  поведения в кольцевой буфер. `best_match` (`:17`) находит детекцию с
  наибольшей уверенностью, совпадающую с метками поведения;
  `ticks_since`/`recent_count` обеспечивают логику кулдауна и анти-лупа.
- **Поведения**
  ([`ai/behavior.py`](src/smartuibot/ai/behavior.py)): `Behavior` — это
  `Condition` (набор меток + пороги) плюс `base_utility`, `cooldown_s` и
  декларативные `BehaviorStep`. `resolve_steps`
  (`src/smartuibot/ai/behavior.py:41`) превращает абстрактные шаги в конкретные
  `ActionStep`, разрешая `target: detection | roi_center | fixed` в
  пиксельные координаты. Поведения загружаются из
  [`configs/behaviors.yaml`](configs/behaviors.yaml) с помощью
  [`ai/registry.py`](src/smartuibot/ai/registry.py) (`load_behaviors` в
  `src/smartuibot/ai/registry.py:29`), который валидирует виды и значения.
- **Политика полезности**
  ([`ai/utility.py`](src/smartuibot/ai/utility.py), `UtilityPolicy` в
  `src/smartuibot/ai/utility.py:13`): оценивает каждое поведение, чьё условие
  выполнено, и выбирает argmax. Оценка (`choose`, `:34`) применяет
  масштабирование по уверенности, исключение по кулдауну, штраф анти-лупа,
  когда поведение слишком часто повторяется в окне, небольшой
  **детерминированный (seeded)** случайный джиттер и периодический пропуск
  «колебание» — чтобы игра выглядела человеческой и была воспроизводима из
  `decision.rng_seed`.

## Этап 4 — Действие (автоматизация ввода)

[`input/service.py`](src/smartuibot/input/service.py) (`ActionService` в
`src/smartuibot/input/service.py:28`) потребляет `ActionRequested` (только
самое свежее) и, только если всё ещё ARMED, выполняет шаги по одному, с
ограничением частоты `max_actions_per_second`. Он перепроверяет `_aborted()`
(`src/smartuibot/input/service.py:62`) **перед каждым шагом и каждой
подточкой движения**, поэтому снятие с взвода или аварийная остановка
прерывают действие в середине и эмитят `ActionAborted`; иначе `ActionStarted`
→ `ActionCompleted`.

- **Очеловеченное движение**
  ([`input/motion.py`](src/smartuibot/input/motion.py)): `bezier_path`
  (`src/smartuibot/input/motion.py:19`) рисует квадратичную кривую Безье со
  случайной контрольной точкой и джиттером в каждой точке; плюс случайные
  задержки реакции и опциональный перелёт (overshoot). Вся случайность — из
  внедрённого детерминированного `random.Random`.
- **Ограничение по ROI**: когда `roi_confine` включён, `_screen`/`_confine`
  (`src/smartuibot/input/service.py:57`) зажимают каждую цель внутри выбранной
  области перед маппингом в абсолютные координаты экрана — клики не могут выйти
  за рамку.
- **Бэкенды**: `InputBackend` — это `Protocol`
  ([`input/backend.py`](src/smartuibot/input/backend.py),
  `src/smartuibot/input/backend.py:8`). По умолчанию используется безопасный
  `NoOpInputBackend` (`:18`, ничего не вводит);
  [`pynput_backend.py`](src/smartuibot/input/pynput_backend.py) (mac/Linux) и
  [`pydirectinput_backend.py`](src/smartuibot/input/pydirectinput_backend.py)
  (Windows) выбираются `resolve_input_backend_name`.

## Безопасность: FSM режима

[`ai/mode.py`](src/smartuibot/ai/mode.py) (`ModeFSM` в
`src/smartuibot/ai/mode.py:13`) — потокобезопасный шлюз с тремя состояниями:
`DISARMED` (по умолчанию), `ARMED`, `PAUSED`. **Ввод возможен только в
ARMED**, и сервисы решений и действий проверяют это независимо. Кнопка
Arm/Disarm (`UiController.toggle_arm`, `src/smartuibot/ui/controls.py:67`) и
горячая клавиша аварийной остановки / предохранитель угла
(`src/smartuibot/app.py:110`) переключают его в Disarmed и прерывают действие.
Он стартует в DISARMED, если только не задано `input.start_armed: true`
(`src/smartuibot/core/container.py:53`).

## Слой UI (Qt, главный поток)

Правило Qt: вся работа с GUI — в главном потоке. Потоки-воркеры только
публикуют события.

- [`ui/controls.py`](src/smartuibot/ui/controls.py): `UiController`
  (`src/smartuibot/ui/controls.py:16`) — **чистый клей без импортов Qt** —
  каждое действие (start/stop, pause, confidence, reload, ROI, arm)
  юнит-тестируемо против фейкового контейнера. `ControlBar` (`:75`) — тонкий
  Qt-виджет поверх.
- [`ui/debug_window.py`](src/smartuibot/ui/debug_window.py): `DebugWindow`
  (`src/smartuibot/ui/debug_window.py:48`) подписывается на события из
  потоков-воркеров, буферизует их в `queue.Queue` и сливает по `QTimer` с
  частотой ~30 Гц (`_drain`, `:113`) — потокобезопасная передача. Рисует
  рамки детекций, FPS, логи, режим и журнал действий.
- [`ui/roi_selector.py`](src/smartuibot/ui/roi_selector.py): полноэкранный
  полупрозрачный оверлей, растянутый на `QScreen` сконфигурированного монитора
  (`_resolve_screen`). Drag-release переводит логические точки в пиксельное
  пространство бэкенда захвата в чистой функции `selection_to_roi`, масштабируя
  по **измеренному отношению** размера монитора mss к логическому размеру
  экрана Qt (не по `devicePixelRatio`, который на macOS+mss удваивает), с
  зажимом по границам монитора — корректно на Retina/HiDPI. Результат
  сохраняется в `configs/state.yaml` (`save_roi`). Esc или слишком маленькое
  выделение отменяют без изменения состояния.

## Конфигурация и логирование

- [`core/config.py`](src/smartuibot/core/config.py): `load_config`
  (`src/smartuibot/core/config.py:104`) загружает
  [`configs/default.yaml`](configs/default.yaml), глубоко мёржит опциональный
  пользовательский оверрайд и строит валидированный frozen `AppConfig`
  (`src/smartuibot/core/config.py:68`, fail-fast в `__post_init__`).
- [`core/logging_setup.py`](src/smartuibot/core/logging_setup.py):
  `setup_logging` (`src/smartuibot/core/logging_setup.py:50`) устанавливает три
  обработчика — цветную консоль, ротируемый **JSON**-файл в `logs/` и
  `_BusHandler`, который перепубликует записи логов как события `LogRecord`,
  чтобы они появлялись вживую в окне отладки.
- [`platform_support/detect.py`](src/smartuibot/platform_support/detect.py):
  `current_os` / `resolve_backend_name` / `resolve_input_backend_name`
  (`src/smartuibot/platform_support/detect.py:15`) превращают `auto` в нужный
  для каждой ОС бэкенд захвата и ввода.

## Стратегия тестирования

[`tests/`](tests) зеркалит пакет. Фейки в
[`tests/fakes/`](tests/fakes) (capture, detector, input) позволяют
[`tests/integration/test_closed_loop.py`](tests/integration/test_closed_loop.py)
и [`test_pipeline.py`](tests/integration/test_pipeline.py) прогонять полный
цикл capture→decision→action headless и детерминированно (seeded RNG).
Юнит-тесты покрывают каждый модуль; маркер `model` изолирует единственный
набор тестов, которому нужны реальные веса YOLO, поэтому запуск по умолчанию
остаётся офлайн и быстрым.

## Проектные документы

Более глубокое обоснование и планы слайсов находятся в
[`docs/superpowers/specs/`](docs/superpowers/specs) и
[`docs/superpowers/plans/`](docs/superpowers/plans) — Слайс A (CV-конвейер),
Слайс B (движок решений + ввод) и селектор области захвата.
